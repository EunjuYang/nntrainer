unittest_causallm = executable('unittest_causallm',
  'unittest_causallm.cpp',
  include_directories: causallm_inc,
  dependencies: [
    nntrainer_dep,
    nntrainer_ccapi_dep,
    gtest_dep,
    causallm_layer_dependencies,
    causallm_dep,
  ],
  install: false,
)

test('unittest_causallm',
  unittest_causallm,
  args: ['--gtest_output=xml:@0@/unittest_causallm.xml'.format(meson.build_root())],
  timeout: 300,
  suite: 'causallm_unittests',
)
