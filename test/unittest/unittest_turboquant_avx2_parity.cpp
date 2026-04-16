// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer authors
 *
 * @file   unittest_turboquant_avx2_parity.cpp
 * @date   16 April 2026
 * @brief  Parity tests between the scalar TurboQuant fallback and the AVX2
 *         path. Pins down behaviour the existing FP32-reference tests do
 *         not catch: (1) bit-exact 4-bit nibble decisions and (2) per-
 *         element numerical agreement within a tight FMA-reordering
 *         tolerance for the dot-product / accumulation kernels.
 *
 *         Built only on x86 hosts; on non-x86 the AVX2 entry points do
 *         not exist and the file is excluded by meson.
 *
 * @see    https://github.com/nntrainer/nntrainer
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <avx2_turboquant.h>
#include <fallback_internal.h>
#include <turboquant_utils.h>

namespace {

/// Tolerance for continuous outputs (scores, accumulated values, norms).
/// Differences below this come from FMA fusion + reduction-tree reordering;
/// anything above suggests an actual algorithmic divergence.
constexpr float kFmaAtol = 1e-4f;

/// Tolerance for the per-head L2 norm scalar.
constexpr float kNormAtol = 1e-4f;

struct KvFixture {
  int head_dim;
  int num_cache_head;
  int gqa_size;
  int num_rows;
  uint32_t seed;
};

static std::vector<float> make_random(size_t n, std::mt19937 &gen, float lo,
                                      float hi) {
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> out(n);
  for (auto &v : out)
    v = d(gen);
  return out;
}

static void quantize_all_rows_via_fallback(const KvFixture &f,
                                           const std::vector<float> &raw_kv,
                                           const std::vector<float> &rot_signs,
                                           std::vector<uint8_t> &packed,
                                           std::vector<float> &norms) {
  const int packed_row_bytes = f.num_cache_head * f.head_dim / 2;
  packed.assign(f.num_rows * packed_row_bytes, 0);
  norms.assign(f.num_rows * f.num_cache_head, 0.0f);
  for (int r = 0; r < f.num_rows; ++r) {
    nntrainer::__fallback_quantize_kv_turboquant(
      raw_kv.data() + r * f.num_cache_head * f.head_dim,
      packed.data() + r * packed_row_bytes,
      norms.data() + r * f.num_cache_head, rot_signs.data(), f.head_dim,
      f.num_cache_head);
  }
}

} // namespace

/**
 * @brief Bit-exact equality on the packed nibble stream and tight numeric
 *        equality on the per-head L2 norm. The quantizer is expected to
 *        produce identical 4-bit indices because (a) the FWHT butterfly
 *        order does not differ between scalar and AVX2 stages, and (b) the
 *        boundary comparison `val > b_i` is order-independent. The norm is
 *        computed via different reduction trees, so a small absolute
 *        tolerance is allowed.
 */
TEST(turboquant_avx2_parity, quantize_kv_bitexact_nibbles) {
  for (int head_dim : {64, 128}) {
    for (int num_heads : {1, 4, 13}) {
      std::mt19937 gen(0xC0FFEEu ^ (head_dim * 31 + num_heads));
      auto input = make_random(num_heads * head_dim, gen, -2.0f, 2.0f);
      std::vector<float> rot_signs(head_dim);
      nntrainer::generate_random_signs(rot_signs.data(), head_dim,
                                       0xBADC0DEu);

      const int packed_bytes = num_heads * head_dim / 2;
      std::vector<uint8_t> p_fb(packed_bytes), p_av(packed_bytes);
      std::vector<float> n_fb(num_heads), n_av(num_heads);

      nntrainer::__fallback_quantize_kv_turboquant(
        input.data(), p_fb.data(), n_fb.data(), rot_signs.data(), head_dim,
        num_heads);
      nntrainer::avx2::quantize_kv_turboquant(input.data(), p_av.data(),
                                              n_av.data(), rot_signs.data(),
                                              head_dim, num_heads);

      for (int b = 0; b < packed_bytes; ++b) {
        ASSERT_EQ(p_fb[b], p_av[b])
          << "Nibble mismatch at byte " << b << " (head_dim=" << head_dim
          << ", num_heads=" << num_heads << ")";
      }
      for (int h = 0; h < num_heads; ++h) {
        EXPECT_NEAR(n_fb[h], n_av[h], kNormAtol)
          << "Norm mismatch at head " << h << " (head_dim=" << head_dim
          << ", num_heads=" << num_heads << ")";
      }
    }
  }
}

/**
 * @brief Per-element parity for compute_kcaches_packed4 against the scalar
 *        fallback. Uses a tight absolute tolerance that allows FMA fusion
 *        and 8-way reduction-tree reordering but would still catch any
 *        single-bin nibble misalignment (which would change a score by
 *        ~ centroid_step * |rq[d]|, far above kFmaAtol).
 */
TEST(turboquant_avx2_parity, compute_kcaches_matches_fallback) {
  const std::vector<KvFixture> cases = {
    {64, 4, 2, 17, 0x1111u},
    {64, 8, 4, 64, 0x2222u},
    {128, 2, 1, 9, 0x3333u},
    {128, 6, 4, 33, 0x4444u},
  };

  for (const auto &f : cases) {
    std::mt19937 gen(f.seed);
    const int num_q_heads = f.num_cache_head * f.gqa_size;
    auto query = make_random(num_q_heads * f.head_dim, gen, -1.5f, 1.5f);
    auto raw_k =
      make_random(f.num_rows * f.num_cache_head * f.head_dim, gen, -1.0f, 1.0f);

    std::vector<float> rot_signs(f.head_dim);
    nntrainer::generate_random_signs(rot_signs.data(), f.head_dim, f.seed ^ 0xA5);

    std::vector<uint8_t> packed;
    std::vector<float> norms;
    quantize_all_rows_via_fallback(f, raw_k, rot_signs, packed, norms);

    const size_t out_n = (size_t)f.num_rows * num_q_heads;
    std::vector<float> out_fb(out_n, 0.0f);
    std::vector<float> out_av(out_n, 0.0f);

    nntrainer::__fallback_compute_kcaches_packed4(
      query.data(), packed.data(), norms.data(), out_fb.data(), f.num_rows,
      f.num_cache_head, f.head_dim, f.gqa_size, /*tile=*/1, rot_signs.data(),
      /*window=*/(size_t)f.num_rows, /*head_start=*/0, /*head_end=*/-1);

    nntrainer::avx2::compute_kcaches_packed4(
      query.data(), packed.data(), norms.data(), out_av.data(), f.num_rows,
      f.num_cache_head, f.head_dim, f.gqa_size, /*tile=*/1, rot_signs.data(),
      /*window=*/(size_t)f.num_rows, /*head_start=*/0, /*head_end=*/-1);

    float max_abs_diff = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < out_n; ++i) {
      float d = std::fabs(out_fb[i] - out_av[i]);
      if (d > max_abs_diff) {
        max_abs_diff = d;
        worst = i;
      }
    }
    EXPECT_LT(max_abs_diff, kFmaAtol)
      << "head_dim=" << f.head_dim << " num_cache_head=" << f.num_cache_head
      << " gqa=" << f.gqa_size << " rows=" << f.num_rows
      << " worst_idx=" << worst << " fb=" << out_fb[worst]
      << " av=" << out_av[worst];
  }
}

/**
 * @brief Per-element parity for compute_vcache_packed4. The accumulator is
 *        per-output-position, so the ordering of additions is identical
 *        between scalar and AVX2 — empirically bit-exact. We still use
 *        kFmaAtol to remain robust against future micro-optimisations
 *        (e.g. partial reduction across rows).
 */
TEST(turboquant_avx2_parity, compute_vcache_matches_fallback) {
  const std::vector<KvFixture> cases = {
    {64, 4, 2, 12, 0x5555u},
    {64, 8, 4, 50, 0x6666u},
    {128, 2, 1, 7, 0x7777u},
    {128, 4, 4, 41, 0x8888u},
  };

  for (const auto &f : cases) {
    std::mt19937 gen(f.seed);
    const int num_q_heads = f.num_cache_head * f.gqa_size;
    auto raw_v =
      make_random(f.num_rows * f.num_cache_head * f.head_dim, gen, -1.0f, 1.0f);

    // Attention weights with shape (num_rows, num_cache_head, gqa_size),
    // softmax-like (positive, normalised per query head).
    std::vector<float> attn(f.num_rows * num_q_heads);
    for (auto &v : attn) {
      std::uniform_real_distribution<float> d(0.0f, 1.0f);
      v = d(gen);
    }
    for (int qh = 0; qh < num_q_heads; ++qh) {
      float sum = 0.0f;
      for (int r = 0; r < f.num_rows; ++r)
        sum += attn[r * num_q_heads + qh];
      for (int r = 0; r < f.num_rows; ++r)
        attn[r * num_q_heads + qh] /= sum;
    }

    std::vector<float> rot_signs(f.head_dim);
    nntrainer::generate_random_signs(rot_signs.data(), f.head_dim, f.seed ^ 0x5A);

    std::vector<uint8_t> packed;
    std::vector<float> norms;
    quantize_all_rows_via_fallback(f, raw_v, rot_signs, packed, norms);

    const size_t out_n = (size_t)num_q_heads * f.head_dim;
    std::vector<float> out_fb(out_n, 0.0f);
    std::vector<float> out_av(out_n, 0.0f);
    const int row_num = f.num_rows - 1;

    nntrainer::__fallback_compute_vcache_packed4(
      row_num, attn.data(), packed.data(), norms.data(), out_fb.data(),
      f.num_cache_head, f.gqa_size, f.head_dim, rot_signs.data(),
      /*window=*/(size_t)f.num_rows, /*head_start=*/0, /*head_end=*/-1);

    nntrainer::avx2::compute_vcache_packed4(
      row_num, attn.data(), packed.data(), norms.data(), out_av.data(),
      f.num_cache_head, f.gqa_size, f.head_dim, rot_signs.data(),
      /*window=*/(size_t)f.num_rows, /*head_start=*/0, /*head_end=*/-1);

    float max_abs_diff = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < out_n; ++i) {
      float d = std::fabs(out_fb[i] - out_av[i]);
      if (d > max_abs_diff) {
        max_abs_diff = d;
        worst = i;
      }
    }
    EXPECT_LT(max_abs_diff, kFmaAtol)
      << "head_dim=" << f.head_dim << " num_cache_head=" << f.num_cache_head
      << " gqa=" << f.gqa_size << " rows=" << f.num_rows
      << " worst_idx=" << worst << " fb=" << out_fb[worst]
      << " av=" << out_av[worst];
  }
}

/**
 * @brief End-to-end parity: a sliding-window decode pass that exercises
 *        head_start/head_end and local_window_size, which the simpler
 *        cases above leave at defaults.
 */
TEST(turboquant_avx2_parity, kcaches_with_window_and_head_range) {
  const int head_dim = 128;
  const int num_cache_head = 8;
  const int gqa_size = 2;
  const int num_rows = 200;
  const size_t window = 64;
  const int head_start = 2;
  const int head_end = 6;

  std::mt19937 gen(0xDEADu);
  const int num_q_heads = num_cache_head * gqa_size;
  auto query = make_random(num_q_heads * head_dim, gen, -1.0f, 1.0f);
  auto raw_k = make_random(num_rows * num_cache_head * head_dim, gen, -1.0f,
                           1.0f);

  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xC0DEu);

  KvFixture f{head_dim, num_cache_head, gqa_size, num_rows, 0};
  std::vector<uint8_t> packed;
  std::vector<float> norms;
  quantize_all_rows_via_fallback(f, raw_k, rot_signs, packed, norms);

  const size_t out_n = (size_t)num_rows * num_q_heads;
  std::vector<float> out_fb(out_n, 0.0f);
  std::vector<float> out_av(out_n, 0.0f);

  nntrainer::__fallback_compute_kcaches_packed4(
    query.data(), packed.data(), norms.data(), out_fb.data(), num_rows,
    num_cache_head, head_dim, gqa_size, 1, rot_signs.data(), window, head_start,
    head_end);
  nntrainer::avx2::compute_kcaches_packed4(
    query.data(), packed.data(), norms.data(), out_av.data(), num_rows,
    num_cache_head, head_dim, gqa_size, 1, rot_signs.data(), window, head_start,
    head_end);

  // Only the [head_start, head_end) x window tile is written by both impls;
  // compare only those slots.
  const int row_cnt = (int)std::min<size_t>(window, (size_t)num_rows);
  float max_abs_diff = 0.0f;
  for (int t_row = 0; t_row < row_cnt; ++t_row) {
    for (int n = head_start; n < head_end; ++n) {
      for (int g = 0; g < gqa_size; ++g) {
        size_t idx = (size_t)t_row * num_cache_head * gqa_size +
                     (size_t)n * gqa_size + g;
        max_abs_diff =
          std::max(max_abs_diff, std::fabs(out_fb[idx] - out_av[idx]));
      }
    }
  }
  EXPECT_LT(max_abs_diff, kFmaAtol);
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return result;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}
