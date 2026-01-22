#!/bin/bash

# Build script for CausalLM API Test Application
# This script assumes that the API library has been built/provided (e.g. by build_android.sh api)
# It mimics an external developer's workflow who only has the shared library and headers.

set -e

# Check if NDK path is set
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set. Please set it to your Android NDK path."
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export NNTRAINER_ROOT

echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"

# Locations of prebuilt artifacts (simulating a distribution package)
PREBUILT_LIBS_DIR="$SCRIPT_DIR/jni/libs/arm64-v8a"
API_INCLUDE_DIR="$SCRIPT_DIR/api"

# Check if required libraries exist
if [ ! -f "$PREBUILT_LIBS_DIR/libnntrainer_causallm_api.so" ]; then
    echo "Error: libnntrainer_causallm_api.so not found in $PREBUILT_LIBS_DIR"
    echo "Please run './build_android.sh api' first to build the API library."
    exit 1
fi

echo "Building CausalLM API Test App using prebuilt API library..."

# Create a temporary build directory for the test app
TEST_BUILD_DIR="$SCRIPT_DIR/test_build_jni"
if [ -d "$TEST_BUILD_DIR" ]; then
    rm -rf "$TEST_BUILD_DIR"
fi
mkdir -p "$TEST_BUILD_DIR"

# Create Android.mk for the test app
cat <<EOF > "$TEST_BUILD_DIR/Android.mk"
LOCAL_PATH := \$(call my-dir)

include \$(CLEAR_VARS)

# Prebuilt API Library
include \$(CLEAR_VARS)
LOCAL_MODULE := nntrainer_causallm_api
LOCAL_SRC_FILES := $PREBUILT_LIBS_DIR/libnntrainer_causallm_api.so
include \$(PREBUILT_SHARED_LIBRARY)

# Prebuilt NNTrainer Libraries (Dependencies of API lib, needed for linking if not handled by rpath/linker automatically, usually needed)
include \$(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $PREBUILT_LIBS_DIR/libnntrainer.so
include \$(PREBUILT_SHARED_LIBRARY)

include \$(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $PREBUILT_LIBS_DIR/libccapi-nntrainer.so
include \$(PREBUILT_SHARED_LIBRARY)

# Test Application
include \$(CLEAR_VARS)
LOCAL_MODULE := nntrainer_causallm_api_test
LOCAL_SRC_FILES := $SCRIPT_DIR/api/test_api.cpp

LOCAL_C_INCLUDES := $API_INCLUDE_DIR

LOCAL_SHARED_LIBRARIES := nntrainer_causallm_api nntrainer ccapi-nntrainer
LOCAL_LDLIBS := -llog -landroid

# Ensure C++17
LOCAL_CPP_FEATURES := rtti exceptions
LOCAL_CPPFLAGS += -std=c++17

include \$(BUILD_EXECUTABLE)
EOF

# Create Application.mk
cat <<EOF > "$TEST_BUILD_DIR/Application.mk"
APP_ABI := arm64-v8a
APP_PLATFORM := android-29
APP_STL := c++_shared
EOF

# Build
cd "$TEST_BUILD_DIR"
ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)

echo "Build completed."
echo "Test App Executable: $TEST_BUILD_DIR/libs/arm64-v8a/nntrainer_causallm_api_test"

# Optionally copy the result back to main libs folder or leave it there
mkdir -p "$SCRIPT_DIR/jni/libs/arm64-v8a/"
cp "$TEST_BUILD_DIR/libs/arm64-v8a/nntrainer_causallm_api_test" "$SCRIPT_DIR/jni/libs/arm64-v8a/"
echo "Copied executable to $SCRIPT_DIR/jni/libs/arm64-v8a/"

# Cleanup
# rm -rf "$TEST_BUILD_DIR" 
