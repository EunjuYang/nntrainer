#!/bin/bash

# Run CausalLM API unit tests
# Supports both x86 (local) and Android (adb) execution
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_header()  { echo -e "\n${YELLOW}========================================${NC}"; echo -e "${YELLOW} $1 ${NC}"; echo -e "${YELLOW}========================================${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default: auto-detect platform
PLATFORM="${1:-auto}"

detect_platform() {
    if [ "$PLATFORM" = "android" ]; then
        echo "android"
    elif [ "$PLATFORM" = "x86" ] || [ "$PLATFORM" = "local" ]; then
        echo "x86"
    else
        # Auto-detect: check if android binary exists
        if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/unittest_causallm_api" ]; then
            if command -v adb &> /dev/null; then
                DEVICES=$(adb devices 2>/dev/null | grep -c "device$") || true
                if [ "$DEVICES" -gt 0 ]; then
                    echo "android"
                    return
                fi
            fi
        fi

        # Check if x86 binary exists (meson build)
        local builddir="$NNTRAINER_ROOT/builddir"
        if [ -f "$builddir/Applications/CausalLM/test/unittest_causallm_api" ]; then
            echo "x86"
            return
        fi

        echo "unknown"
    fi
}

run_x86() {
    log_header "Run Unit Tests (x86/local)"

    local test_bin="$NNTRAINER_ROOT/builddir/Applications/CausalLM/test/unittest_causallm_api"

    if [ ! -f "$test_bin" ]; then
        log_error "Test binary not found: $test_bin"
        log_info "Build with: meson test -C builddir unittest_causallm_api"
        log_info "Or: ninja -C builddir unittest_causallm_api"
        exit 1
    fi

    log_info "Binary: $test_bin"
    log_info "Running tests..."
    echo ""

    "$test_bin" --gtest_output="xml:$NNTRAINER_ROOT/builddir/unittest_causallm_api.xml" "${@:2}"
    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "All tests passed!"
    else
        log_error "Tests failed with exit code: $exit_code"
    fi
    return $exit_code
}

run_android() {
    log_header "Run Unit Tests (Android)"

    local INSTALL_DIR="/data/local/tmp/nntrainer/causallm"
    local test_bin="$SCRIPT_DIR/jni/libs/arm64-v8a/unittest_causallm_api"

    # Check adb
    if ! command -v adb &> /dev/null; then
        log_error "adb command not found. Install Android SDK Platform Tools."
        exit 1
    fi

    # Check device
    DEVICES=$(adb devices | grep -c "device$") || true
    if [ "$DEVICES" -eq 0 ]; then
        log_error "No Android device connected."
        exit 1
    fi
    DEVICE_ID=$(adb devices | grep "device$" | head -1 | cut -f1)
    log_success "Device: $DEVICE_ID"

    # Check binary
    if [ ! -f "$test_bin" ]; then
        log_error "Test binary not found: $test_bin"
        log_info "Build with: cd jni && ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk unittest_causallm_api"
        exit 1
    fi

    # Push files
    log_info "Pushing test binary to device..."
    adb shell "mkdir -p $INSTALL_DIR" 2>/dev/null || true
    adb push "$test_bin" "$INSTALL_DIR/" > /dev/null

    # Push required libraries (if they exist)
    local libs_dir="$SCRIPT_DIR/jni/libs/arm64-v8a"
    for lib in libcausallm_core.so libcausallm_api.so libnntrainer.so libccapi-nntrainer.so libc++_shared.so libomp.so; do
        if [ -f "$libs_dir/$lib" ]; then
            adb push "$libs_dir/$lib" "$INSTALL_DIR/" > /dev/null
        fi
    done

    adb shell "chmod 755 $INSTALL_DIR/unittest_causallm_api"

    # Run tests
    log_info "Running tests on device..."
    echo ""

    adb shell "cd $INSTALL_DIR && export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH && ./unittest_causallm_api --gtest_output=xml:/data/local/tmp/unittest_causallm_api.xml ${*:2}" 2>&1
    local exit_code=$?

    # Pull result XML
    adb pull "/data/local/tmp/unittest_causallm_api.xml" "$SCRIPT_DIR/" 2>/dev/null || true

    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "All tests passed on device!"
    else
        log_error "Tests failed with exit code: $exit_code"
    fi
    return $exit_code
}

# Main
DETECTED=$(detect_platform)

case "$DETECTED" in
    x86)
        run_x86 "$@"
        ;;
    android)
        run_android "$@"
        ;;
    *)
        log_error "Could not detect platform. Specify explicitly:"
        log_info "  $0 x86      # Run locally (meson build)"
        log_info "  $0 android  # Run on Android device (ndk-build)"
        exit 1
        ;;
esac
