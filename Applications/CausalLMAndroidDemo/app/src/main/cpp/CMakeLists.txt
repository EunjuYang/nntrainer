cmake_minimum_required(VERSION 3.18.1)
project("causallmdemo")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add CausalLM source directory
set(CAUSALLM_DIR ${CMAKE_SOURCE_DIR}/../../../../../../CausalLM)
set(NNTRAINER_DIR ${CMAKE_SOURCE_DIR}/../../../../../../../)

# Include directories
include_directories(
    ${CAUSALLM_DIR}
    ${CAUSALLM_DIR}/layers
    ${NNTRAINER_DIR}/nntrainer/include
    ${NNTRAINER_DIR}/api/ccapi/include
    ${NNTRAINER_DIR}/api/capi/include/platform
)

# Find required libraries
find_library(log-lib log)
find_library(android-lib android)

# Add CausalLM sources
set(CAUSALLM_SOURCES
    ${CAUSALLM_DIR}/causal_lm.cpp
    ${CAUSALLM_DIR}/huggingface_tokenizer.cpp
    ${CAUSALLM_DIR}/llm_util.cpp
    ${CAUSALLM_DIR}/qwen3_causallm.cpp
    ${CAUSALLM_DIR}/qwen3_moe_causallm.cpp
    ${CAUSALLM_DIR}/qwen3_slim_moe_causallm.cpp
)

# Add layer sources
set(LAYER_SOURCES
    ${CAUSALLM_DIR}/layers/rms_norm.cpp
    ${CAUSALLM_DIR}/layers/tie_word_embedding.cpp
    ${CAUSALLM_DIR}/layers/swiglu.cpp
    ${CAUSALLM_DIR}/layers/mha_core.cpp
    ${CAUSALLM_DIR}/layers/embedding_layer.cpp
    ${CAUSALLM_DIR}/layers/reshaped_rms_norm.cpp
    ${CAUSALLM_DIR}/layers/moe_layer.cpp
    ${CAUSALLM_DIR}/layers/slim_moe_layer.cpp
)

# Create native library
add_library(causallm-native SHARED
    causallm_jni.cpp
    ${CAUSALLM_SOURCES}
    ${LAYER_SOURCES}
)

# Link libraries
target_link_libraries(causallm-native
    ${log-lib}
    ${android-lib}
    ${CAUSALLM_DIR}/lib/libtokenizers_c.a
    # Add nntrainer libraries - these need to be built for Android
    # ${NNTRAINER_DIR}/build/nntrainer/libnntrainer.so
    # ${NNTRAINER_DIR}/build/api/ccapi/libccapi-nntrainer.so
)

# Set compiler flags
target_compile_options(causallm-native PRIVATE
    -DPLUGGABLE
    -fexceptions
    -frtti
    -O3
)

# For OpenMP support (if needed)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(causallm-native OpenMP::OpenMP_CXX)
endif()