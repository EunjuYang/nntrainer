unittest_causallm_api = executable(
  'unittest_causallm_api',
  'unittest_causallm_api.cpp',
  dependencies: [
    gtest_dep,
    gtest_main_dep,
    nntrainer_dep,
    nntrainer_ccapi_dep,
    causallm_layer_dependencies,
    causallm_dep,
    causallm_api_dep,
  ],
  include_directories: causallm_inc,
  install: get_option('enable-test'),
  install_dir: application_install_dir,
)

test('unittest_causallm_api', unittest_causallm_api,
  args: ['--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_causallm_api')],
  timeout: test_timeout,
  suite: 'unittests',
)
