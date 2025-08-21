#!/bin/bash

# Optimal OpenMP settings for Q4_0 model performance consistency
# These settings help reduce performance variance

echo "Setting optimal OpenMP environment variables for Q4_0 models..."

# 1. Thread affinity - bind threads to cores to prevent migration
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# 2. Fixed number of threads - prevent dynamic adjustment
export OMP_NUM_THREADS=6
export OMP_DYNAMIC=false

# 3. Scheduling policy - static for predictable performance
export OMP_SCHEDULE="static"

# 4. Thread idle behavior - keep threads spinning for low latency
export OMP_WAIT_POLICY=active

# 5. Stack size - ensure sufficient stack for quantized operations
export OMP_STACKSIZE=16M

# 6. Nested parallelism - disable to avoid oversubscription
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

# 7. CPU frequency governor (if available)
if command -v cpupower &> /dev/null; then
    echo "Setting CPU governor to performance mode..."
    sudo cpupower frequency-set -g performance 2>/dev/null || echo "Could not set CPU governor (may need sudo)"
fi

# 8. NUMA settings (if on NUMA system)
if command -v numactl &> /dev/null; then
    echo "Detected NUMA system"
    export OMP_PROC_BIND=spread
    # For single NUMA node, use local allocation
    export GOMP_CPU_AFFINITY="0-5"
fi

# 9. Disable CPU frequency scaling if possible
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo "Disabling Intel Turbo Boost for consistent performance..."
    echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "Could not disable Turbo Boost"
fi

# Print current settings
echo ""
echo "Current OpenMP Settings:"
echo "========================"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"
echo "OMP_PLACES=$OMP_PLACES"
echo "OMP_SCHEDULE=$OMP_SCHEDULE"
echo "OMP_DYNAMIC=$OMP_DYNAMIC"
echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
echo "OMP_STACKSIZE=$OMP_STACKSIZE"
echo "OMP_NESTED=$OMP_NESTED"
echo "OMP_MAX_ACTIVE_LEVELS=$OMP_MAX_ACTIVE_LEVELS"

# Additional recommendations
echo ""
echo "Additional Recommendations:"
echo "==========================="
echo "1. Build with: -DGGML_THREAD_BACKEND=omp"
echo "2. Use fixed batch sizes when possible"
echo "3. Warm up the model with a few iterations before benchmarking"
echo "4. Monitor CPU temperature to avoid thermal throttling"
echo "5. Disable other CPU-intensive processes during inference"

echo ""
echo "To make these settings permanent, add them to ~/.bashrc or /etc/environment"