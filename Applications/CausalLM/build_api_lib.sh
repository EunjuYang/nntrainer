#!/bin/bash

# Build script for CausalLM API Library
# This script builds libcausallm_api.so only
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

log_header "Build CausalLM API Library"
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"
echo "Working directory: $(pwd)"

# Check if CausalLM Core is built
log_step "1/2" "Check CausalLM Core"

if [ ! -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_core.so" ]; then
    log_error "libcausallm_core.so not found."
    echo "Please run build_android.sh first to build the core library."
    exit 1
fi
log_success "CausalLM Core found"

# Step 2: Build CausalLM API
log_step "2/2" "Build CausalLM API Library"

cd "$SCRIPT_DIR/jni"

log_info "Building with ndk-build (builds libcausallm_api.so)..."
if ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk causallm_api -j $(nproc); then
    log_success "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

# Verify output
echo ""
echo "Build artifacts:"
if [ -f "libs/arm64-v8a/libcausallm_api.so" ]; then
    size=$(ls -lh "libs/arm64-v8a/libcausallm_api.so" | awk '{print $5}')
    echo -e "  ${GREEN}[OK]${NC} libcausallm_api.so ($size)"
else
    echo -e "  ${RED}[ERROR]${NC} libcausallm_api.so not found!"
    exit 1
fi

# Summary
log_header "Build Summary"
log_success "Build completed successfully!"
echo ""
echo "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
echo ""
echo "Libraries:"
echo "  - libcausallm_api.so (CausalLM API library)"
echo ""
echo "To build test app, run:"
echo "  ./build_test_app.sh"
echo ""
