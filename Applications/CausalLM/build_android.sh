#!/bin/bash

# Build script for CausalLM Android application
# This script builds libcausallm_core.so and nntrainer_causallm executable
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN} $1 ${NC}"
    echo -e "${CYAN}========================================${NC}"
}

log_step() {
    echo -e "\n${YELLOW}[Step $1]${NC} $2"
    echo -e "${YELLOW}----------------------------------------${NC}"
}

# Check if NDK path is set
if [ -z "$ANDROID_NDK" ]; then
    log_error "ANDROID_NDK is not set. Please set it to your Android NDK path."
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi

# Set NNTRAINER_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export NNTRAINER_ROOT

log_header "Build CausalLM Android Application"
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"
echo "Working directory: $(pwd)"

# Step 1: Build nntrainer for Android if not already built
log_step "1/4" "Build nntrainer for Android"

if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    log_info "Building nntrainer for Android..."
    cd "$NNTRAINER_ROOT"
    if [ -d "$NNTRAINER_ROOT/builddir" ]; then
        rm -rf builddir
    fi
    ./tools/package_android.sh -Dmmap-read=false
else
    log_info "nntrainer for Android already built (skipping)"
fi

# Check if build was successful
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    log_error "nntrainer build failed. Please check the build logs."
    exit 1
fi
log_success "nntrainer ready"

# Step 2: Build tokenizer library if not present
log_step "2/4" "Build Tokenizer Library"

cd "$SCRIPT_DIR"
if [ ! -f "lib/libtokenizers_android_c.a" ]; then
    log_warning "libtokenizers_android_c.a not found in lib directory."
    log_info "Attempting to build tokenizer library..."
    if [ -f "build_tokenizer_android.sh" ]; then
        ./build_tokenizer_android.sh
    else
        log_error "tokenizer library not found and build script is missing."
        echo "Please build or download the tokenizer library for Android arm64-v8a"
        echo "and place it in: $SCRIPT_DIR/lib/libtokenizers_android_c.a"
        exit 1
    fi
else
    log_info "Tokenizer library already built (skipping)"
fi
log_success "Tokenizer library ready"

# Step 3: Prepare json.hpp if not present
log_step "3/4" "Prepare json.hpp"

if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
    log_info "json.hpp not found. Downloading..."
    # prepare_encoder.sh expects target directory as first argument and version as second
    # It copies json.hpp to ../Applications/CausalLM/ if version is 0.2
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2"
    
    if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
        log_error "Failed to download json.hpp"
        exit 1
    fi
else
    log_info "json.hpp already exists (skipping)"
fi
log_success "json.hpp ready"

# Step 4: Build CausalLM (libcausallm_core.so and nntrainer_causallm)
log_step "4/4" "Build CausalLM Core (library + executable)"

cd "$SCRIPT_DIR/jni"

# Clean previous builds
rm -rf obj libs

log_info "Building with ndk-build (builds nntrainer_causallm and dependencies)..."
if ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk nntrainer_causallm -j $(nproc); then
    log_success "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

# Verify outputs
echo ""
echo "Build artifacts:"
if [ -f "libs/arm64-v8a/libcausallm_core.so" ]; then
    size=$(ls -lh "libs/arm64-v8a/libcausallm_core.so" | awk '{print $5}')
    echo -e "  ${GREEN}[OK]${NC} libcausallm_core.so ($size)"
else
    echo -e "  ${RED}[ERROR]${NC} libcausallm_core.so not found!"
    echo "Directory contents of libs/arm64-v8a/:"
    ls -l libs/arm64-v8a/ || echo "Directory not found"
    exit 1
fi

if [ -f "libs/arm64-v8a/nntrainer_causallm" ]; then
    size=$(ls -lh "libs/arm64-v8a/nntrainer_causallm" | awk '{print $5}')
    echo -e "  ${GREEN}[OK]${NC} nntrainer_causallm ($size)"
else
    echo -e "  ${RED}[ERROR]${NC} nntrainer_causallm not found!"
    exit 1
fi

# Summary
log_header "Build Summary"
log_success "Build completed successfully!"
echo ""
echo "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
echo ""
echo "Executables:"
echo "  - nntrainer_causallm (main application)"
echo ""
echo "Libraries:"
echo "  - libcausallm_core.so (CausalLM Core library)"
echo "  - libnntrainer.so (nntrainer library)"
echo "  - libccapi-nntrainer.so (nntrainer C/C API)"
echo "  - libc++_shared.so (C++ runtime)"
echo ""
echo "To build API library, run:"
echo "  ./build_api_lib.sh"
echo ""
echo "To install and run:"
echo "  ./install_android.sh"
echo ""
