#!/bin/bash

# Build script for CausalLM API Android application
# This script builds libcausallm_api.so and test_api executable
set -e

# Check if NDK path is set
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set. Please set it to your Android NDK path."
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi

# Set NNTRAINER_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export NNTRAINER_ROOT

echo "========================================"
echo "Build CausalLM API Android Application"
echo "========================================"
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"
echo "Working directory: $(pwd)"
echo ""

# Check if CausalLM Core is built
echo "[Step 1/2] Check CausalLM Core"
echo "----------------------------------------"
if [ ! -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_core.so" ]; then
    echo "Error: libcausallm_core.so not found."
    echo "Please run build_android.sh first to build the core library."
    exit 1
fi
echo "[SUCCESS] CausalLM Core found"
echo ""

# Step 2: Build CausalLM API and test_api
echo "[Step 2/2] Build CausalLM API and Test App"
echo "----------------------------------------"
cd "$SCRIPT_DIR/jni"

echo "Building with ndk-build (builds libcausallm_api.so and test_api)..."
# We explicitly specify targets
if ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk libcausallm_api test_api -j $(nproc); then
    echo "[SUCCESS] Build completed successfully"
else
    echo "Error: Build failed"
    exit 1
fi

# Verify outputs
echo ""
echo "Build artifacts:"
if [ -f "libs/arm64-v8a/libcausallm_api.so" ]; then
    size=$(ls -lh "libs/arm64-v8a/libcausallm_api.so" | awk '{print $5}')
    echo "  [OK] libcausallm_api.so ($size)"
else
    echo "  [ERROR] libcausallm_api.so not found!"
    exit 1
fi

if [ -f "libs/arm64-v8a/test_api" ]; then
    size=$(ls -lh "libs/arm64-v8a/test_api" | awk '{print $5}')
    echo "  [OK] test_api ($size)"
else
    echo "  [ERROR] test_api not found!"
    exit 1
fi
echo ""

# Summary
echo "========================================"
echo "Build Summary"
echo "========================================"
echo "Build completed successfully!"
echo ""
echo "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
echo ""
echo "Executables:"
echo "  - test_api (API test application)"
echo ""
echo "Libraries:"
echo "  - libcausallm_api.so (CausalLM API library)"
echo ""
