#!/bin/bash

# Build script for CausalLM API and Test App
# This script builds the API as a shared library and then builds the test app using that library.
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

echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"

prepare_deps() {
    # Step 1: Build nntrainer for Android if not already built
    if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
        echo "Building nntrainer for Android..."
        cd "$NNTRAINER_ROOT"
        if [ -d "$NNTRAINER_ROOT/builddir" ]; then
            rm -rf builddir
        fi
        ./tools/package_android.sh -Dmmap-read=false
    else
        echo "nntrainer for Android already built."
    fi

    # Check if build was successful
    if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
        echo "Error: nntrainer build failed. Please check the build logs."
        exit 1
    fi

    # Step 2: Build tokenizer library if not present
    echo "Check Tokenizer Library"
    cd "$SCRIPT_DIR"
    if [ ! -f "lib/libtokenizers_android_c.a" ]; then
        echo "Warning: libtokenizers_android_c.a not found in lib directory."
        echo "Attempting to build tokenizer library..."
        if [ -f "build_tokenizer_android.sh" ]; then
            ./build_tokenizer_android.sh
        else
            echo "Error: tokenizer library not found and build script is missing."
            exit 1
        fi
    fi

    # Step 3: Prepare json.hpp if not present
    if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
        echo "json.hpp not found. Downloading..."
        "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2"
        if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
            echo "Error: Failed to download json.hpp"
            exit 1
        fi
    fi
}

build_api() {
    echo "Building CausalLM API Shared Library and Test App..."
    cd "$SCRIPT_DIR/jni"
    
    # Clean previous builds if requested or to ensure clean state for API specific artifacts
    # rm -rf libs obj  <-- Optional, but maybe safer to keep incremental unless explicit clean

    ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android_api.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)
    
    echo "Build completed."
}

clean_build() {
    echo "Cleaning build artifacts..."
    cd "$SCRIPT_DIR/jni"
    rm -rf libs obj
}

CMD=${1:-all}

case "$CMD" in
    clean)
        clean_build
        ;;
    *)
        prepare_deps
        build_api
        ;;
esac

if [ "$CMD" != "clean" ]; then
    echo "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
    echo "Shared Library: libnntrainer_causallm_api.so"
    echo "Test Executable: nntrainer_causallm_api_test"
fi
