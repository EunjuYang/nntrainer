#!/bin/bash

# Android Performance Optimization Script for CausalLM with MoE
# This script sets up environment variables and system settings for optimal performance

echo "=== Android CausalLM MoE Performance Optimizer ==="
echo ""

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then 
        echo "Warning: Some optimizations require root access"
        echo "Run with 'su' or 'adb shell' as root for full optimizations"
        HAVE_ROOT=0
    else
        HAVE_ROOT=1
    fi
}

# 1. Set MoE Cache Configuration
setup_moe_cache() {
    echo "1. Setting up MoE cache configuration..."
    
    # Increase cache size based on available memory
    TOTAL_MEM=$(cat /proc/meminfo | grep MemTotal | awk '{print $2}')
    TOTAL_MEM_MB=$((TOTAL_MEM / 1024))
    
    if [ $TOTAL_MEM_MB -gt 8192 ]; then
        # For devices with > 8GB RAM
        export NNTRAINER_MOE_CACHE_SIZE=32
        echo "   - Set cache size to 32 experts (high-memory device)"
    elif [ $TOTAL_MEM_MB -gt 4096 ]; then
        # For devices with 4-8GB RAM
        export NNTRAINER_MOE_CACHE_SIZE=24
        echo "   - Set cache size to 24 experts (medium-memory device)"
    else
        # For devices with < 4GB RAM
        export NNTRAINER_MOE_CACHE_SIZE=16
        echo "   - Set cache size to 16 experts (low-memory device)"
    fi
    
    # Enable prefetching for better performance
    export NNTRAINER_MOE_PREFETCH=1
    echo "   - Enabled expert prefetching"
    
    # Pin frequently used experts in memory (Android-specific)
    export NNTRAINER_PIN_EXPERT_MEMORY=1
    echo "   - Enabled memory pinning for frequently used experts"
}

# 2. Configure mmap optimizations
setup_mmap() {
    echo ""
    echo "2. Configuring mmap optimizations..."
    
    # Enable madvise for better memory access patterns
    export NNTRAINER_USE_MADVISE=1
    echo "   - Enabled madvise for sequential access"
    
    # Enable MADV_WILLNEED for prefetching (use carefully, may increase memory usage)
    export NNTRAINER_MADVISE_WILLNEED=1
    echo "   - Enabled MADV_WILLNEED for prefetching"
    
    # Enable MAP_POPULATE for pre-faulting pages (increases startup time but reduces runtime latency)
    # Only enable for high-memory devices
    if [ $TOTAL_MEM_MB -gt 6144 ]; then
        export NNTRAINER_MMAP_POPULATE=1
        echo "   - Enabled MAP_POPULATE for pre-faulting (high-memory device)"
    else
        echo "   - MAP_POPULATE disabled (conserve memory)"
    fi
}

# 3. System-level I/O optimizations (requires root)
setup_io_optimizations() {
    echo ""
    echo "3. Setting up I/O optimizations..."
    
    if [ $HAVE_ROOT -eq 1 ]; then
        # Increase read-ahead for better sequential performance
        for dev in /sys/block/*/queue/read_ahead_kb; do
            if [ -w "$dev" ]; then
                echo 4096 > "$dev" 2>/dev/null
            fi
        done
        echo "   - Increased block device read-ahead to 4096KB"
        
        # Set I/O scheduler to noop or deadline for SSDs
        for dev in /sys/block/*/queue/scheduler; do
            if [ -w "$dev" ]; then
                # Check if noop is available
                if grep -q "\[noop\]" "$dev" 2>/dev/null; then
                    echo noop > "$dev" 2>/dev/null
                elif grep -q "noop" "$dev" 2>/dev/null; then
                    echo noop > "$dev" 2>/dev/null
                elif grep -q "none" "$dev" 2>/dev/null; then
                    echo none > "$dev" 2>/dev/null
                fi
            fi
        done
        echo "   - Set I/O scheduler to noop/none for better SSD performance"
        
        # Disable unnecessary I/O stats
        for dev in /sys/block/*/queue/iostats; do
            if [ -w "$dev" ]; then
                echo 0 > "$dev" 2>/dev/null
            fi
        done
        echo "   - Disabled I/O statistics collection"
        
        # Increase nr_requests for better I/O queue depth
        for dev in /sys/block/*/queue/nr_requests; do
            if [ -w "$dev" ]; then
                echo 512 > "$dev" 2>/dev/null
            fi
        done
        echo "   - Increased I/O queue depth to 512"
    else
        echo "   - Skipping system I/O optimizations (requires root)"
    fi
}

# 4. CPU Performance Settings (requires root)
setup_cpu_performance() {
    echo ""
    echo "4. Configuring CPU performance..."
    
    if [ $HAVE_ROOT -eq 1 ]; then
        # Set CPU governor to performance
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            if [ -w "$cpu" ]; then
                echo performance > "$cpu" 2>/dev/null
            fi
        done
        echo "   - Set CPU governor to performance mode"
        
        # Disable CPU idle states for lower latency
        for idle in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
            if [ -w "$idle" ]; then
                echo 1 > "$idle" 2>/dev/null
            fi
        done
        echo "   - Disabled deep CPU idle states"
        
        # Set CPU affinity for big cores if available (ARM big.LITTLE)
        if [ -f /sys/devices/system/cpu/cpu0/topology/physical_package_id ]; then
            # Try to identify big cores (usually higher numbered CPUs)
            BIG_CORES=""
            for i in $(seq 4 7); do
                if [ -d /sys/devices/system/cpu/cpu$i ]; then
                    BIG_CORES="$BIG_CORES,$i"
                fi
            done
            if [ -n "$BIG_CORES" ]; then
                export OMP_PLACES="{${BIG_CORES:1}}"
                echo "   - Set OpenMP affinity to big cores: ${BIG_CORES:1}"
            fi
        fi
    else
        echo "   - Skipping CPU optimizations (requires root)"
    fi
}

# 5. Memory Management
setup_memory() {
    echo ""
    echo "5. Configuring memory management..."
    
    # Set OpenMP thread count based on CPU cores
    CPU_CORES=$(nproc)
    export OMP_NUM_THREADS=$CPU_CORES
    echo "   - Set OpenMP threads to $CPU_CORES"
    
    # Set thread affinity for better cache locality
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores
    echo "   - Enabled OpenMP thread binding to cores"
    
    # Set OpenMP scheduling for better load balancing
    export OMP_SCHEDULE="dynamic,1"
    echo "   - Set OpenMP scheduling to dynamic with chunk size 1"
    
    # Enable OpenMP wait policy for lower latency
    export OMP_WAIT_POLICY=active
    echo "   - Set OpenMP wait policy to active"
    
    if [ $HAVE_ROOT -eq 1 ]; then
        # Adjust swappiness for less swapping
        echo 10 > /proc/sys/vm/swappiness 2>/dev/null
        echo "   - Reduced swappiness to 10"
        
        # Increase VFS cache pressure
        echo 50 > /proc/sys/vm/vfs_cache_pressure 2>/dev/null
        echo "   - Set VFS cache pressure to 50"
        
        # Disable transparent huge pages (can cause latency spikes)
        echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null
        echo "   - Disabled transparent huge pages"
        
        # Increase dirty ratio for better write performance
        echo 40 > /proc/sys/vm/dirty_ratio 2>/dev/null
        echo 20 > /proc/sys/vm/dirty_background_ratio 2>/dev/null
        echo "   - Adjusted dirty page ratios for better write performance"
    else
        echo "   - Skipping memory optimizations (requires root)"
    fi
}

# 6. Print optimization summary
print_summary() {
    echo ""
    echo "=== Optimization Summary ==="
    echo ""
    echo "Environment variables set:"
    echo "  NNTRAINER_MOE_CACHE_SIZE=$NNTRAINER_MOE_CACHE_SIZE"
    echo "  NNTRAINER_MOE_PREFETCH=$NNTRAINER_MOE_PREFETCH"
    echo "  NNTRAINER_PIN_EXPERT_MEMORY=$NNTRAINER_PIN_EXPERT_MEMORY"
    echo "  NNTRAINER_USE_MADVISE=$NNTRAINER_USE_MADVISE"
    echo "  NNTRAINER_MADVISE_WILLNEED=$NNTRAINER_MADVISE_WILLNEED"
    if [ -n "$NNTRAINER_MMAP_POPULATE" ]; then
        echo "  NNTRAINER_MMAP_POPULATE=$NNTRAINER_MMAP_POPULATE"
    fi
    echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "  OMP_PROC_BIND=$OMP_PROC_BIND"
    echo "  OMP_PLACES=$OMP_PLACES"
    echo "  OMP_SCHEDULE=$OMP_SCHEDULE"
    echo "  OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
    echo ""
    echo "To make these settings permanent, add them to your shell profile"
    echo "or create a launcher script for your application."
    echo ""
    echo "Example launcher command:"
    echo "  ./nntrainer_causallm [your_model_path] [other_args]"
    echo ""
    echo "For testing different configurations:"
    echo "  - Increase cache: export NNTRAINER_MOE_CACHE_SIZE=48"
    echo "  - Disable prefaulting: unset NNTRAINER_MMAP_POPULATE"
    echo "  - Disable willneed: unset NNTRAINER_MADVISE_WILLNEED"
    echo ""
    echo "Performance monitoring:"
    echo "  - Watch I/O: iostat -x 1"
    echo "  - Watch memory: watch -n 1 'cat /proc/meminfo | grep -E \"MemFree|Cached\"'"
    echo "  - Watch CPU: top -H -p \$(pidof nntrainer_causallm)"
}

# Main execution
main() {
    check_root
    setup_moe_cache
    setup_mmap
    setup_io_optimizations
    setup_cpu_performance
    setup_memory
    print_summary
}

# Run main function
main

# Export all variables for child processes
export NNTRAINER_MOE_CACHE_SIZE
export NNTRAINER_MOE_PREFETCH
export NNTRAINER_PIN_EXPERT_MEMORY
export NNTRAINER_USE_MADVISE
export NNTRAINER_MADVISE_WILLNEED
export NNTRAINER_MMAP_POPULATE
export OMP_NUM_THREADS
export OMP_PROC_BIND
export OMP_PLACES
export OMP_SCHEDULE
export OMP_WAIT_POLICY