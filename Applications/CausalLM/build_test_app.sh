#!/bin/bash

# Build script for CausalLM Test Application
# This script builds test_api executable only
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
echo "Build CausalLM Test Application"
echo "========================================"
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"
echo "Working directory: $(pwd)"
echo ""

# Check required libraries
echo "[Step 1/2] Check Dependencies"
echo "----------------------------------------"
MISSING_DEPS=false

# Check Core Lib
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_core.so" ]; then
    echo "  [OK] libcausallm_core.so found"
else
    echo "  [MISSING] libcausallm_core.so"
    echo "    -> Run ./build_android.sh first"
    MISSING_DEPS=true
fi

# Check API Lib
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_api.so" ]; then
    echo "  [OK] libcausallm_api.so found"
else
    echo "  [MISSING] libcausallm_api.so"
    echo "    -> Run ./build_api_lib.sh first"
    MISSING_DEPS=true
fi

if [ "$MISSING_DEPS" = true ]; then
    echo ""
    echo "Error: Missing dependencies. Please build required libraries first."
    exit 1
fi

echo "[SUCCESS] All dependencies found"
echo ""

# Step 2: Build Test App
echo "[Step 2/2] Build Test App"
echo "----------------------------------------"
cd "$SCRIPT_DIR/jni"

echo "Building with ndk-build (builds test_api)..."
if ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk test_api -j $(nproc); then
    echo "[SUCCESS] Build completed successfully"
else
    echo "Error: Build failed"
    exit 1
fi

# Verify output
echo ""
echo "Build artifacts:"
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
echo "To install and run:"
echo "  ./install_android.sh"
echo ""
