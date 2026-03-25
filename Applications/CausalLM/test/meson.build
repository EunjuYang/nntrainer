causallm_test_name = 'test_causallm_qwen3'

causallm_test_src = [
    meson.current_source_dir() / 'test_causallm_qwen3.cpp',
]

causallm_test_exe = executable(
    causallm_test_name,
    causallm_test_src,
    include_directories: causallm_inc,
    dependencies: [
        gtest_dep,
        nntrainer_dep,
        nntrainer_ccapi_dep,
        causallm_layer_dependencies,
        causallm_dep,
    ],
    install: get_option('enable-test'),
    install_dir: application_install_dir,
)

test(causallm_test_name, causallm_test_exe,
    args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), causallm_test_name),
    timeout: 600,
    suite: 'causallm_tests',
    is_parallel: false,
)
