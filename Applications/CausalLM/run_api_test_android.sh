#!/bin/bash

# Script to deploy and run CausalLM API Test App on Android device
# Usage: ./run_api_test_android.sh [DEVICE_ID]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBS_DIR="$SCRIPT_DIR/jni/libs/arm64-v8a"
DEVICE_BIN_DIR="/data/local/tmp/causallm_test"

# Optional: Device ID argument
ADB_FLAGS=""
if [ ! -z "$1" ]; then
    ADB_FLAGS="-s $1"
fi

echo "Checking build artifacts..."
if [ ! -f "$LIBS_DIR/nntrainer_causallm_api_test" ]; then
    echo "Error: Test executable not found. Please run ./build_api_test_app.sh first."
    exit 1
fi
if [ ! -f "$LIBS_DIR/libnntrainer_causallm_api.so" ]; then
    echo "Error: API library not found. Please run ./build_android.sh api first."
    exit 1
fi

echo "Setting up device directory: $DEVICE_BIN_DIR"
adb $ADB_FLAGS shell "mkdir -p $DEVICE_BIN_DIR"

echo "Pushing libraries and executable..."
# Push shared libraries
adb $ADB_FLAGS push "$LIBS_DIR/libc++_shared.so" "$DEVICE_BIN_DIR/"
adb $ADB_FLAGS push "$LIBS_DIR/libnntrainer.so" "$DEVICE_BIN_DIR/"
adb $ADB_FLAGS push "$LIBS_DIR/libccapi-nntrainer.so" "$DEVICE_BIN_DIR/"
adb $ADB_FLAGS push "$LIBS_DIR/libnntrainer_causallm_api.so" "$DEVICE_BIN_DIR/"

# Push executable
adb $ADB_FLAGS push "$LIBS_DIR/nntrainer_causallm_api_test" "$DEVICE_BIN_DIR/"
adb $ADB_FLAGS shell "chmod +x $DEVICE_BIN_DIR/nntrainer_causallm_api_test"

# Optional: Push tokenizer resources if needed (adjust paths as necessary)
# If the test needs specific model files, they should ideally be pushed here or assumed to be on device.
# For now, we assume models are already at /data/local/tmp/causallm_test/models or similar, 
# or passed as arguments to the test binary.

echo "Running API Test App..."
echo "---------------------------------------------------"
# Set LD_LIBRARY_PATH to look in the current directory for our .so files
adb $ADB_FLAGS shell "cd $DEVICE_BIN_DIR && LD_LIBRARY_PATH=. ./nntrainer_causallm_api_test"
echo "---------------------------------------------------"
echo "Test finished."
