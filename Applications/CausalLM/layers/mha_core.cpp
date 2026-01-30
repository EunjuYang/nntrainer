void MHACoreLayer::compute_kcaches(
  nntrainer::Tensor &in, nntrainer::Tensor &cache, nntrainer::Tensor &out,
  unsigned int from, size_t sequence_len, unsigned int num_head,
  unsigned int group_size, unsigned int head_dim, BS::thread_pool<> &pool) {

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (sequence_len == 1) {
      unsigned int num_heads_kv = num_head / group_size;
      unsigned int num_threads = pool.get_thread_count();
      if (num_threads > 1 && num_heads_kv > 1) {
        unsigned int heads_per_thread =
          (num_heads_kv + num_threads - 1) / num_threads;
        std::vector<std::future<void>> futures;

        for (unsigned int h = 0; h < num_heads_kv; h += heads_per_thread) {
          unsigned int current_chunk_size =
            std::min(heads_per_thread, num_heads_kv - h);

          float *input_addr = in.getData<float>() + h * group_size * head_dim;
          uint16_t *cache_addr = cache.getData<uint16_t>() + h * head_dim;
          float *output_addr = out.getData<float>() + h * group_size;

          futures.emplace_back(pool.submit_task([=]() {
            nntrainer::compute_kcaches<uint16_t>(
              input_addr, cache_addr, output_addr, from + 1, current_chunk_size,
              head_dim, group_size, tile_size, local_window_size, num_heads_kv);
          }));
        }
        for (auto &fut : futures)
          fut.get();
      } else {
        nntrainer::compute_kcaches<uint16_t>(
          in.getData<float>(), cache.getData<uint16_t>(), out.getData<float>(),
          from + 1, num_head / group_size, head_dim, group_size, tile_size,
          local_window_size);
      }
    } else {
      std::vector<std::future<void>> futures;
      int seq =
        sequence_len < local_window_size ? sequence_len : local_window_size;

      for (int i = 0; i < seq; ++i) {
        float *input_addr = in.getData<float>() + num_head * head_dim * i;
        uint16_t *cache_addr = cache.getData<uint16_t>();
        int row_to_compute = from + i + 1;
        size_t out_start_row =
          calc_attn_index(from + i) - calc_attn_index(from);
        float *output_addr = out.getData<float>() + out_start_row * num_head;

        futures.emplace_back(pool.submit_task([=]() {
          nntrainer::compute_kcaches<uint16_t>(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, tile_size,
            local_window_size);
        }));
      }
      for (auto &fut : futures)
        fut.get();
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (sequence_len == 1) {
      unsigned int num_heads_kv = num_head / group_size;
      unsigned int num_threads = pool.get_thread_count();
      if (num_threads > 1 && num_heads_kv > 1) {
        unsigned int heads_per_thread =
          (num_heads_kv + num_threads - 1) / num_threads;
        std::vector<std::future<void>> futures;
        for (unsigned int h = 0; h < num_heads_kv; h += heads_per_thread) {
          unsigned int current_chunk_size =
            std::min(heads_per_thread, num_heads_kv - h);
          _FP16 *input_addr = in.getData<_FP16>() + h * group_size * head_dim;
          _FP16 *cache_addr = cache.getData<_FP16>() + h * head_dim;
          _FP16 *output_addr = out.getData<_FP16>() + h * group_size;
          futures.emplace_back(pool.submit_task([=]() {
            nntrainer::compute_kcaches(input_addr, cache_addr, output_addr,
                                       from + 1, current_chunk_size, head_dim,
                                       group_size, tile_size, local_window_size,
                                       num_heads_kv);
          }));
        }
        for (auto &fut : futures)
          fut.get();
      } else {
        nntrainer::compute_kcaches(in.getData<_FP16>(), cache.getData<_FP16>(),
                                   out.getData<_FP16>(), from + 1,
                                   num_head / group_size, head_dim, group_size,
                                   tile_size, local_window_size);
      }
    } else {
      std::vector<std::future<void>> futures;
      unsigned int seq_start =
        sequence_len < local_window_size ? 0 : sequence_len - local_window_size;
      for (unsigned int i = seq_start; i < sequence_len; ++i) {
        _FP16 *input_addr = in.getData<_FP16>() + num_head * head_dim * i;
        _FP16 *cache_addr = cache.getData<_FP16>();
        int row_to_compute = from + i + 1;
        size_t out_start_row =
          calc_attn_index(from + i) - calc_attn_index(from);

        _FP16 *output_addr = out.getData<_FP16>() + out_start_row * num_head;

        futures.emplace_back(pool.submit_task([=]() {
          nntrainer::compute_kcaches(input_addr, cache_addr, output_addr,
                                     row_to_compute, num_head / group_size,
                                     head_dim, group_size, tile_size,
                                     local_window_size);
        }));
      }
      for (auto &fut : futures)
        fut.get();
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}
