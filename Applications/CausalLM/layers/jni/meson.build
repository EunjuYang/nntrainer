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

and_conf.set('MESON_RMS_SRCS', ' '.join(causallm_rms_norm_src_abs))
and_conf.set('MESON_SWIGLU_SRCS', ' '.join(causallm_swiglu_src_abs))
and_conf.set('MESON_TIEWORD_EMBEDDING_SRCS', ' '.join(causallm_tie_word_embedding_src_abs))
and_conf.set('MESON_MHA_CORE_SRCS', ' '.join(causallm_mha_core_abs))
and_conf.set('MESON_EMBEDDING_SRCS', ' '.join(causallm_embedding_src_abs))
and_conf.set('MESON_RESHAPED_RMS_NORM_SRCS', ' '.join(causallm_reshaped_rms_norm_src_abs))
and_conf.set('MESON_MOE_LAYER_SRCS', ' '.join(causallm_moe_layer_src_abs))

and_conf.set('MESON_CAUSALLM_LAYER_INC', ' '.join(causallm_layer_inc_abs))

configure_file(input: 'Android.mk.in', output: 'Android.mk',
  configuration: and_conf
)
configure_file(input: 'Application.mk', output: 'Application.mk',
  configuration: and_conf
)

causallm_layers_tgt = custom_target('ndk-build',
                          output: 'jni',
                          build_by_default: true,
                          command: [ndk_build_command])
