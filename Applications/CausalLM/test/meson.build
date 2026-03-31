unittest_causallm = executable(
  'unittest_causallm',
  'unittest_causallm.cpp',
  include_directories: causallm_inc,
  dependencies: [
    nntrainer_dep,
    nntrainer_ccapi_dep,
    gtest_dep,
    causallm_layer_dependencies,
    causallm_dep
  ],
)

test('unittest_causallm', unittest_causallm,
  timeout: test_timeout,
  suite: 'unittests'
)
