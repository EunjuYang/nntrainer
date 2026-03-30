// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_turboquant_bench.cpp
 * @brief  Micro-benchmark for TurboQuant v2 hot-path functions.
 *         Measures per-function latency to identify optimization targets.
 */

#include <cpu_backend.h>
#include <gtest/gtest.h>
#include <turboquant_utils.h>

#include <chrono>
#include <cmath>
#include <random>
#include <vector>

static double bench(std::function<void()> fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i)
    fn();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
    fn();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

TEST(turboquant_bench, profile_all_functions) {
  // Qwen3-0.6B config
  constexpr int num_heads_Q = 16;
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int q_width = num_heads_Q * head_dim;
  constexpr int packed_row = kv_width / 2;

  std::mt19937 gen(42);
  std::normal_distribution<float> normal(0.0f, 0.3f);

  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  // ---- Benchmark quantize_kv_turboquant_v2 ----
  std::vector<float> kv_data(kv_width);
  for (auto &v : kv_data) v = normal(gen);
  std::vector<uint8_t> packed(packed_row);
  std::vector<float> norms(num_heads_KV);

  double t_quant = bench([&]() {
    nntrainer::quantize_kv_turboquant_v2(
      kv_data.data(), packed.data(), norms.data(), rot_signs.data(),
      head_dim, num_heads_KV);
  }, 100, 5000);

  // ---- Benchmark compute_kcaches_packed4_v2 (single-token decode) ----
  for (int ctx : {32, 128, 512, 2048}) {
    std::vector<uint8_t> tq_kc(ctx * packed_row);
    std::vector<float> tq_kn(ctx * num_heads_KV);
    for (auto &v : tq_kc) v = gen() % 256;
    for (auto &v : tq_kn) v = normal(gen);

    std::vector<float> query(q_width);
    for (auto &v : query) v = normal(gen);
    std::vector<float> scores(ctx * num_heads_Q, 0.0f);

    double t_kcache = bench([&]() {
      std::fill(scores.begin(), scores.end(), 0.0f);
      nntrainer::compute_kcaches_packed4_v2(
        query.data(), tq_kc.data(), tq_kn.data(), scores.data(),
        ctx, num_heads_KV, head_dim, gqa_size, 4, rot_signs.data());
    }, 10, 500);

    printf("  compute_kcaches_packed4_v2  ctx=%4d: %8.1f us\n", ctx, t_kcache);
  }

  // ---- Benchmark compute_vcache_packed4_v2 (single-token decode) ----
  for (int ctx : {32, 128, 512, 2048}) {
    std::vector<uint8_t> tq_vc(ctx * packed_row);
    std::vector<float> tq_vn(ctx * num_heads_KV);
    for (auto &v : tq_vc) v = gen() % 256;
    for (auto &v : tq_vn) v = normal(gen);

    std::vector<float> attn(ctx * num_heads_Q);
    float w = 1.0f / ctx;
    for (auto &v : attn) v = w;

    std::vector<float> vout(q_width, 0.0f);

    double t_vcache = bench([&]() {
      std::fill(vout.begin(), vout.end(), 0.0f);
      nntrainer::compute_vcache_packed4_v2(
        ctx - 1, attn.data(), tq_vc.data(), tq_vn.data(), vout.data(),
        num_heads_KV, gqa_size, head_dim, rot_signs.data());
    }, 10, 500);

    printf("  compute_vcache_packed4_v2   ctx=%4d: %8.1f us\n", ctx, t_vcache);
  }

  // ---- Benchmark sub-components ----
  // Hadamard transform alone
  std::vector<float> had_buf(head_dim);
  for (auto &v : had_buf) v = normal(gen);
  double t_hadamard = bench([&]() {
    nntrainer::hadamard_transform(had_buf.data(), head_dim);
  }, 1000, 50000);

  // Lloyd-Max quantize alone (128 elements)
  std::vector<float> qbuf(head_dim);
  for (auto &v : qbuf) v = normal(gen) * 0.2f;
  const auto &cb = nntrainer::get_codebook(head_dim);
  std::vector<uint8_t> qout(head_dim);
  double t_lloydmax = bench([&]() {
    for (int i = 0; i < head_dim; ++i)
      qout[i] = nntrainer::lloydmax_quantize(qbuf[i], cb);
  }, 1000, 50000);

  // Centroid lookup dot product (inner loop of kcache)
  std::vector<uint8_t> packed_head(head_dim / 2);
  for (auto &v : packed_head) v = gen() % 256;
  std::vector<float> rq(head_dim);
  for (auto &v : rq) v = normal(gen);
  double t_dotprod = bench([&]() {
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d += 2) {
      uint8_t byte = packed_head[d / 2];
      sum += rq[d] * cb.centroids[byte & 0x0F];
      sum += rq[d + 1] * cb.centroids[(byte >> 4) & 0x0F];
    }
    volatile float sink = sum;
    (void)sink;
  }, 1000, 50000);

  printf("\n=== TurboQuant v2 Micro-Benchmark (Qwen3-0.6B config) ===\n");
  printf("  quantize_kv (1 row, 4 heads):    %8.1f us\n", t_quant);
  printf("  hadamard_transform (dim=128):     %8.3f us\n", t_hadamard);
  printf("  lloydmax_quantize (128 elems):    %8.3f us\n", t_lloydmax);
  printf("  centroid dot product (128 elems): %8.3f us\n", t_dotprod);
}

GTEST_API_ int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
