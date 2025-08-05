#!/bin/bash

# Quick run script for CausalLM Android Demo
set -e

echo "=== CausalLM Android Demo Quick Run ==="

# Check prerequisites
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set"
    echo "Please run: export ANDROID_NDK=/path/to/android-ndk"
    exit 1
fi

if [ -z "$ANDROID_HOME" ]; then
    echo "Error: ANDROID_HOME is not set"
    echo "Please run: export ANDROID_HOME=/path/to/android-sdk"
    exit 1
fi

# Step 1: Build nntrainer for Android (if needed)
echo "Step 1: Checking nntrainer build..."
if [ ! -f "/workspace/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    echo "Building nntrainer for Android..."
    cd /workspace
    ./tools/package_android.sh
else
    echo "nntrainer already built."
fi

# Step 2: Build tokenizer (if needed)
echo "Step 2: Checking tokenizer..."
cd /workspace/Applications/CausalLM
if [ ! -f "lib/libtokenizers_android_c.a" ]; then
    echo "Building tokenizer..."
    ./build_tokenizer_android.sh
else
    echo "Tokenizer already built."
fi

# Step 3: Build Android app
echo "Step 3: Building Android app..."
cd /workspace/Applications/CausalLMAndroidDemo
chmod +x gradlew
./gradlew assembleDebug

# Step 4: Install app
echo "Step 4: Installing app..."
APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
if [ -f "$APK_PATH" ]; then
    adb install -r "$APK_PATH"
    echo "App installed successfully!"
else
    echo "Error: APK not found at $APK_PATH"
    exit 1
fi

# Step 5: Create model directory
echo "Step 5: Creating model directory on device..."
adb shell mkdir -p /sdcard/Download/models

echo ""
echo "=== Setup Complete! ==="
echo "1. Copy your model files to device:"
echo "   adb push /path/to/model /sdcard/Download/models/model_name/"
echo "2. Launch 'CausalLM Demo' app on your device"
echo "3. Grant storage permissions when prompted"
echo "4. Select model and enter prompt to generate text"