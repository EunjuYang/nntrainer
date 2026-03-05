causallm_src = [
    meson.current_source_dir() / 'huggingface_tokenizer.cpp',
    meson.current_source_dir() / 'llm_util.cpp',
]

executable_src = [
    meson.current_source_dir() / 'main.cpp',
]

causallm_inc_abs = [meson.current_source_dir()]
causallm_inc = [include_directories('.')]

# Build layer dependency
causallm_layer_dependencies = []
subdir('layers')

# Add common layers to dependencies
causallm_layer_dependencies += [
    causallm_rms_norm_dep,
    causallm_tie_word_embedding_dep,
    causallm_swiglu_dep,
    causallm_mha_core_dep,
    causallm_embedding_layer_dep,
    causallm_lm_head_dep,
    causallm_reshaped_rms_norm_dep,
    causallm_qkv_layer_dep,
    causallm_embedding_pooling_layer_dep,
    causallm_embedding_normalize_layer_dep,
]

subdir('models')

if (get_option('platform') == 'windows') and (build_machine.system() == 'windows')
    run_command(powershell_prog, '-ExecutionPolicy', 'Bypass', '-File', join_paths(meson.source_root(), 'jni', 'prepare_encoder.ps1'), meson.build_root(), '0.2', check: true)
elif get_option('platform') != 'tizen'
    run_command([meson.source_root() / 'jni' / 'prepare_encoder.sh', meson.build_root(), '0.2'], check: true)
endif

tokenizer_path = meson.current_source_dir() / 'lib' / 'libtokenizers_c.a'

# libtokenizers_c.a is a Rust-built static library that depends on oniguruma
onig_dep = dependency('oniguruma', required: true)
thread_dep = dependency('threads')
dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
tokenizer_dependencies = [onig_dep, thread_dep, dl_dep]

cpp_args = []

if get_option('default_library') == 'shared'
    cpp_args += [ '-DPLUGGABLE' ]
endif

causallm = shared_library('causallm',
    causallm_src,
    dependencies: [
        nntrainer_dep,
        nntrainer_ccapi_dep,
        causallm_layer_dependencies, 
        tokenizer_dependencies,
        openmp_dep
    ],
    include_directories: causallm_inc,
    install: true,
    install_dir: application_install_dir,
    link_args: [tokenizer_path],
    cpp_args: cpp_args
)

causallm_dep = declare_dependency(
    link_with: [causallm],
    include_directories: causallm_inc
)

e = executable('nntr_causallm',
    executable_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep],
)