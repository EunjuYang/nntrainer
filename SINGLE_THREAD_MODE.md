# NNTrainer Single Thread Mode

## Overview

NNTrainer now supports an optional single thread mode that allows the entire framework to run in the main thread without creating any worker threads. This feature is essential for WebAssembly compilation and other environments where threading is not available or desired.

## Features

- **Optional Configuration**: Choose between threaded and non-threaded execution via a simple property
- **Backward Compatible**: Defaults to the original multi-threaded behavior
- **Runtime Configurable**: Can be set at runtime before training begins
- **Zero Performance Impact**: When disabled, uses the exact same code path as before
- **WebAssembly Ready**: Enables compilation to WebAssembly without pthread support

## How It Works

### Multi-threaded Mode (Default)
```
Main Thread                 Worker Thread
    |                            |
    ├─ startFetchWorker() ──────>├─ Continuously fill buffer
    |                            ├─ Load data in background
    ├─ fetch() <─────────────────├─ Serve filled iterations
    ├─ Process data              |
    └─ ...                       └─ ...
```

### Single Thread Mode
```
Main Thread Only
    |
    ├─ startFetchWorker() (no thread created)
    ├─ fetch() 
    │   ├─ Generate data on-demand
    │   └─ Return iteration
    ├─ Process data
    └─ ...
```

## Configuration

### Method 1: Property String
```cpp
DataBuffer buffer(std::move(producer));
buffer.setProperty({
    "buffer_size=4",
    "single_thread_mode=true"  // Enable single thread mode
});
```

### Method 2: Configuration File
```ini
[DataBuffer]
buffer_size=4
single_thread_mode=true
```

### Method 3: Compile-time Default
```cpp
#ifdef __EMSCRIPTEN__
    buffer.setProperty({"single_thread_mode=true"});
#endif
```

## Property Details

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `single_thread_mode` | bool | false | When true, runs entirely in main thread without creating worker threads |

## Usage Examples

### WebAssembly Build
```cpp
// For WebAssembly, enable single thread mode
buffer.setProperty({"single_thread_mode=true"});
```

### Native Build with Performance
```cpp
// For native builds, use default multi-threaded mode
buffer.setProperty({"single_thread_mode=false"});  // or omit for default
```

### Automatic Detection
```cpp
#ifdef __EMSCRIPTEN__
    buffer.setProperty({"single_thread_mode=true"});
#else
    buffer.setProperty({"single_thread_mode=false"});
#endif
```

## Implementation Details

When `single_thread_mode=true`:

1. `startFetchWorker()` does not create a thread via `std::async`
2. Initial data preparation happens immediately
3. Returns an already-fulfilled future
4. `fetch()` generates data on-demand when called
5. Uses non-blocking queue operations (`tryPop` instead of `waitAndPop`)
6. Maintains state between `fetch()` calls

When `single_thread_mode=false` (default):

1. Original behavior is preserved
2. Worker thread is created for background data loading
3. Uses blocking queue operations for thread synchronization
4. Better performance for native environments with thread support

## Performance Considerations

### Single Thread Mode
- ✅ No thread creation overhead
- ✅ Predictable execution in main thread
- ✅ WebAssembly compatible
- ⚠️ Data generation blocks main thread
- ⚠️ No background prefetching

### Multi Thread Mode (Default)
- ✅ Background data prefetching
- ✅ Better CPU utilization
- ✅ Main thread not blocked during I/O
- ⚠️ Thread creation overhead
- ⚠️ Not compatible with single-threaded environments

## Migration Guide

### For WebAssembly Projects

Before:
```cpp
// Would fail in WebAssembly due to thread creation
buffer.startFetchWorker(input_dims, label_dims);
```

After:
```cpp
// Works in WebAssembly
buffer.setProperty({"single_thread_mode=true"});
buffer.startFetchWorker(input_dims, label_dims);
```

### For Existing Native Projects

No changes required! The default behavior remains multi-threaded.

## Testing

Test both modes:
```cpp
// Test single thread mode
buffer.setProperty({"single_thread_mode=true"});
runTests();

// Test multi thread mode
buffer.setProperty({"single_thread_mode=false"});
runTests();
```

## Debugging

Enable logging to see which mode is active:
```cpp
// The implementation logs the mode being used
// "DataBuffer: Running in single thread mode (WebAssembly compatible)"
// or
// "DataBuffer: Running in multi-threaded mode"
```

## FAQ

**Q: Does this affect performance on native builds?**
A: No, when `single_thread_mode=false` (default), the exact same code path is used as before.

**Q: Can I switch modes at runtime?**
A: Yes, but only before calling `startFetchWorker()`. Once data fetching starts, the mode is fixed.

**Q: What happens if I don't set the property?**
A: The default is `false`, meaning multi-threaded mode is used (original behavior).

**Q: Is this compatible with all data producers?**
A: Yes, the change is transparent to data producers. They work the same in both modes.

**Q: Can I use this in production?**
A: Yes, the single thread mode is production-ready for WebAssembly deployments. For native deployments, we recommend using the default multi-threaded mode for better performance.

## Support

For issues or questions about single thread mode, please:
1. Check this documentation
2. Review the usage examples
3. Open an issue with the `single-thread-mode` label