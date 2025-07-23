ndk_build_command = [ndk_build]
foreach key, val : ndk_base_args
  ndk_build_command += '@0@=@1@'.format(key, val)
endforeach

ndk_args = {}
foreach key, val : ndk_args
  ndk_build_command += '@0@=@1@'.format(key, val)
endforeach

ndk_build_command += ['-C', meson.current_build_dir()]

message('@0@'.format(ndk_build_command))

and_conf = configuration_data()

and_conf.set('MESON_CAUSALLM_INC', ' '.join(casuallm_inc_abs))
and_conf.set('MESON_CAUSALLM_SRCS', ' '.join(causallm_src_abs))
and_conf.set('MESON_EXECUTABLE_SRC', ' '.join(executable_src))

configure_file(input: 'Android.mk.in', output: 'Android.mk',
  configuration: and_conf
)
configure_file(input: 'Application.mk', output: 'Application.mk',
  configuration: and_conf
)

ndk_result = custom_target('ndk-build',
                          output: 'jni',
                          build_by_default: true,
                          depends: [causallm_layers_tgt],
                          command: [ndk_build_command])