subdir('fake_data_gen')
subdir('benchmark_application')

benchmark_attn_src = files('benchmark_attention_kernels.cpp')

benchmark_attn_deps = [nntrainer_dep]
if get_option('enable-openmp')
  benchmark_attn_deps += dependency('openmp', required: true)
endif

executable('benchmark_attention_kernels',
  benchmark_attn_src,
  dependencies: benchmark_attn_deps,
  install: false,
)
