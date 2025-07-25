#!/bin/bash

echo "Testing compilation of optimized qwen_moe_layer.cpp..."

# Create a simple test to check if the file compiles
cd /workspace/temp_nntrainer

# Check if the file exists
if [ ! -f "Applications/CausalLM/layers/qwen_moe_layer.cpp" ]; then
    echo "Error: qwen_moe_layer.cpp not found!"
    exit 1
fi

# Try to compile just the syntax (not full build)
echo "Checking C++ syntax..."
g++ -std=c++14 -c -fsyntax-only \
    -I./nntrainer \
    -I./nntrainer/layers \
    -I./nntrainer/tensor \
    -I./nntrainer/models \
    -I./nntrainer/utils \
    -I./Applications/CausalLM/layers \
    -fopenmp \
    Applications/CausalLM/layers/qwen_moe_layer.cpp 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Syntax check passed!"
else
    echo "✗ Syntax errors found"
fi

echo "Done."