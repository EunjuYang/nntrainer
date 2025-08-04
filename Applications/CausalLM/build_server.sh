#!/bin/bash

# Build script for CausalLM HTTP server

echo "Building CausalLM HTTP server..."

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    meson build
fi

cd build

# Build the server
echo "Compiling server..."
ninja nntr_causallm_server

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Server binary: build/nntr_causallm_server"
    echo ""
    echo "Usage:"
    echo "  ./build/nntr_causallm_server [port] [model_name model_path ...]"
    echo ""
    echo "Example:"
    echo "  ./build/nntr_causallm_server 8080 \"Qwen3ForCausalLM\" \"/path/to/model\""
else
    echo "Build failed!"
    exit 1
fi