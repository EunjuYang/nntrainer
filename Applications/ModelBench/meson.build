bench_src = [
    meson.current_source_dir() / 'main.cpp',
]

e = executable('nntr_model_bench',
    bench_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep],
)
