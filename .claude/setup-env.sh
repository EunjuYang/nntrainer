#!/bin/bash
# nntrainer development environment setup script
# Called by Claude Code SessionStart hook to prepare the build environment.
set -e

NNTRAINER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARKER_DIR="/tmp/.nntrainer-env-setup"

# Skip if already set up in this session
if [ -f "$MARKER_DIR/done" ]; then
    echo "[setup-env] Environment already configured, skipping."
    exit 0
fi
mkdir -p "$MARKER_DIR"

echo "[setup-env] Setting up nntrainer development environment..."

##############################################################################
# 1. System packages for x86 meson build + gtest
##############################################################################
echo "[setup-env] Installing system dependencies..."
apt-get update -qq 2>/dev/null || true
apt-get install -y --no-install-recommends \
    build-essential gcc g++ pkg-config cmake \
    meson ninja-build \
    libgtest-dev \
    libopenblas-dev \
    libiniparser-dev \
    libjsoncpp-dev \
    libcurl4-gnutls-dev \
    libglib2.0-dev \
    flatbuffers-compiler \
    unzip wget curl \
    2>/dev/null || true

##############################################################################
# 2. Android NDK (r27c, arm64-v8a)
##############################################################################
NDK_VERSION="r27c"
NDK_DIR="/opt/android-ndk-${NDK_VERSION}"
if [ ! -d "$NDK_DIR" ]; then
    echo "[setup-env] Downloading Android NDK ${NDK_VERSION}..."
    NDK_ZIP="/tmp/android-ndk-${NDK_VERSION}-linux.zip"
    if [ ! -f "$NDK_ZIP" ]; then
        wget -q "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip" \
            -O "$NDK_ZIP" || {
            echo "[setup-env] WARNING: Failed to download NDK. Android builds will not work."
            touch "$MARKER_DIR/ndk-failed"
        }
    fi
    if [ -f "$NDK_ZIP" ] && [ ! -f "$MARKER_DIR/ndk-failed" ]; then
        echo "[setup-env] Extracting NDK..."
        unzip -q "$NDK_ZIP" -d /opt/ && rm -f "$NDK_ZIP"
    fi
else
    echo "[setup-env] Android NDK already present at $NDK_DIR"
fi

# Export ANDROID_NDK
if [ -d "$NDK_DIR" ]; then
    export ANDROID_NDK="$NDK_DIR"
    echo "export ANDROID_NDK=\"$NDK_DIR\"" >> "$MARKER_DIR/env.sh"
fi

##############################################################################
# 3. Rust + Android target (for tokenizer cross-compilation)
##############################################################################
if ! command -v rustc &>/dev/null; then
    echo "[setup-env] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>/dev/null || {
        echo "[setup-env] WARNING: Failed to install Rust. Tokenizer Android build will not work."
        touch "$MARKER_DIR/rust-failed"
    }
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
else
    echo "[setup-env] Rust already installed."
fi

if command -v rustup &>/dev/null && [ ! -f "$MARKER_DIR/rust-failed" ]; then
    echo "[setup-env] Adding Rust Android target..."
    rustup target add aarch64-linux-android 2>/dev/null || true
fi

##############################################################################
# 4. Initialize git submodules
##############################################################################
cd "$NNTRAINER_ROOT"
if [ ! -f "subprojects/googletest/CMakeLists.txt" ]; then
    echo "[setup-env] Initializing git submodules..."
    git submodule sync && git submodule update --init --recursive
else
    echo "[setup-env] Git submodules already initialized."
fi

##############################################################################
# 5. Meson x86 build (for running tests locally)
##############################################################################
if [ ! -d "$NNTRAINER_ROOT/build" ]; then
    echo "[setup-env] Configuring meson x86 build..."
    cd "$NNTRAINER_ROOT"
    meson setup \
        --buildtype=plain \
        --prefix=/usr \
        --sysconfdir=/etc \
        --libdir=lib/x86_64-linux-gnu \
        --bindir=lib/nntrainer/bin \
        --includedir=include \
        -Dinstall-app=true \
        -Dreduce-tolerance=false \
        -Denable-debug=true \
        -Denable-test=true \
        -Denable-transformer=true \
        -Dml-api-support=disabled \
        -Denable-nnstreamer-tensor-filter=disabled \
        -Denable-nnstreamer-tensor-trainer=disabled \
        -Denable-nnstreamer-backbone=false \
        -Denable-capi=disabled \
        -Denable-fp16=false \
        -Denable-fsu=false \
        -Denable-tflite-backbone=false \
        -Denable-tflite-interpreter=false \
        build 2>&1 || echo "[setup-env] WARNING: meson configure failed."
else
    echo "[setup-env] Meson build directory already exists."
fi

##############################################################################
# Done
##############################################################################
touch "$MARKER_DIR/done"
echo "[setup-env] Environment setup complete."
echo "[setup-env]   - x86 build: meson (ninja -C build)"
echo "[setup-env]   - Android build: cd Applications/CausalLM && ./build_android.sh"
if [ -d "$NDK_DIR" ]; then
    echo "[setup-env]   - ANDROID_NDK=$NDK_DIR"
fi
