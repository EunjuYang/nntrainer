/**
 * Example: Using NNTrainer with Optional Single Thread Mode
 * This demonstrates how to configure NNTrainer for different execution modes
 */

#include <nntrainer/dataset/databuffer.h>
#include <nntrainer/models/neuralnet.h>
#include <iostream>

// Example 1: WebAssembly build with single thread mode
void exampleWebAssemblyUsage() {
    std::cout << "=== WebAssembly Configuration ===" << std::endl;
    
    // Create your data producer
    auto producer = createYourDataProducer(); // Your implementation
    
    // Create DataBuffer with single thread mode enabled
    nntrainer::DataBuffer buffer(std::move(producer));
    
    // Enable single thread mode for WebAssembly
    buffer.setProperty({
        "buffer_size=4",           // Buffer size for iterations
        "single_thread_mode=true"  // Enable single thread mode
    });
    
    // Now use it normally - it will run entirely in the main thread
    auto future_iq = buffer.startFetchWorker(input_dims, label_dims, shuffle);
    
    // The future is immediately ready (no thread created)
    auto iq = future_iq.get();
    
    // Fetch data - will generate on-demand in main thread
    while (true) {
        auto iter_view = buffer.fetch();
        if (iter_view.isEmpty()) {
            break;
        }
        // Process iteration...
    }
    
    std::cout << "All processing done in main thread!" << std::endl;
}

// Example 2: Native build with multi-threading (default)
void exampleNativeUsage() {
    std::cout << "=== Native Multi-threaded Configuration ===" << std::endl;
    
    // Create your data producer
    auto producer = createYourDataProducer();
    
    // Create DataBuffer - defaults to multi-threaded mode
    nntrainer::DataBuffer buffer(std::move(producer));
    
    // Configure without specifying single_thread_mode (defaults to false)
    buffer.setProperty({
        "buffer_size=4"  // Only set buffer size, threading is default
    });
    
    // Or explicitly set to false for clarity
    buffer.setProperty({
        "buffer_size=4",
        "single_thread_mode=false"  // Explicitly use threads
    });
    
    // Use normally - will create worker thread for data loading
    auto future_iq = buffer.startFetchWorker(input_dims, label_dims, shuffle);
    
    // Worker thread is loading data in background
    // ... do other work ...
    
    auto iq = future_iq.get();
    
    // Fetch data - worker thread fills queue in background
    while (true) {
        auto iter_view = buffer.fetch();
        if (iter_view.isEmpty()) {
            break;
        }
        // Process iteration...
    }
}

// Example 3: Runtime detection and configuration
void exampleRuntimeConfiguration() {
    std::cout << "=== Runtime Configuration ===" << std::endl;
    
    auto producer = createYourDataProducer();
    nntrainer::DataBuffer buffer(std::move(producer));
    
    // Detect if running in WebAssembly environment
    bool is_wasm = false;
    #ifdef __EMSCRIPTEN__
        is_wasm = true;
        std::cout << "Detected WebAssembly environment" << std::endl;
    #else
        std::cout << "Detected native environment" << std::endl;
    #endif
    
    // Configure based on environment
    if (is_wasm) {
        buffer.setProperty({
            "buffer_size=2",           // Smaller buffer for WASM
            "single_thread_mode=true"  // Single thread for WASM
        });
    } else {
        buffer.setProperty({
            "buffer_size=8",            // Larger buffer for native
            "single_thread_mode=false"  // Multi-thread for native
        });
    }
    
    // Use the same way regardless of configuration
    auto future_iq = buffer.startFetchWorker(input_dims, label_dims, shuffle);
    // ... rest of the code ...
}

// Example 4: Configuration file based setup
void exampleConfigFileUsage() {
    std::cout << "=== Configuration File Based Setup ===" << std::endl;
    
    // config_wasm.ini:
    // [DataBuffer]
    // buffer_size=4
    // single_thread_mode=true
    
    // config_native.ini:
    // [DataBuffer]
    // buffer_size=8
    // single_thread_mode=false
    
    // Load configuration
    std::string config_file;
    #ifdef __EMSCRIPTEN__
        config_file = "config_wasm.ini";
    #else
        config_file = "config_native.ini";
    #endif
    
    // Create neural network with configuration
    nntrainer::NeuralNetwork nn;
    nn.loadFromConfig(config_file);
    
    // The DataBuffer will automatically use the configured mode
    nn.train();
}

// Example 5: Environment variable based configuration
void exampleEnvironmentVariableUsage() {
    std::cout << "=== Environment Variable Configuration ===" << std::endl;
    
    auto producer = createYourDataProducer();
    nntrainer::DataBuffer buffer(std::move(producer));
    
    // Check environment variable
    const char* single_thread_env = std::getenv("NNTRAINER_SINGLE_THREAD");
    bool use_single_thread = (single_thread_env && std::string(single_thread_env) == "1");
    
    if (use_single_thread) {
        std::cout << "Single thread mode enabled via environment variable" << std::endl;
        buffer.setProperty({
            "buffer_size=4",
            "single_thread_mode=true"
        });
    } else {
        std::cout << "Multi-thread mode (default)" << std::endl;
        buffer.setProperty({
            "buffer_size=4",
            "single_thread_mode=false"
        });
    }
    
    // Usage:
    // export NNTRAINER_SINGLE_THREAD=1  # Enable single thread mode
    // ./your_application
}

// Example 6: Testing both modes
void testBothModes() {
    std::cout << "=== Testing Both Modes ===" << std::endl;
    
    // Test single thread mode
    {
        std::cout << "\nTesting single thread mode:" << std::endl;
        auto producer = createYourDataProducer();
        nntrainer::DataBuffer buffer(std::move(producer));
        buffer.setProperty({"buffer_size=4", "single_thread_mode=true"});
        
        auto start = std::chrono::steady_clock::now();
        runTraining(buffer);
        auto end = std::chrono::steady_clock::now();
        
        std::cout << "Single thread time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << "ms" << std::endl;
    }
    
    // Test multi thread mode
    {
        std::cout << "\nTesting multi thread mode:" << std::endl;
        auto producer = createYourDataProducer();
        nntrainer::DataBuffer buffer(std::move(producer));
        buffer.setProperty({"buffer_size=4", "single_thread_mode=false"});
        
        auto start = std::chrono::steady_clock::now();
        runTraining(buffer);
        auto end = std::chrono::steady_clock::now();
        
        std::cout << "Multi thread time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << "ms" << std::endl;
    }
}

int main() {
    std::cout << "NNTrainer Optional Single Thread Mode Examples\n" << std::endl;
    
    // Choose which example to run based on your needs
    #ifdef __EMSCRIPTEN__
        exampleWebAssemblyUsage();
    #else
        exampleNativeUsage();
    #endif
    
    // Or test runtime configuration
    exampleRuntimeConfiguration();
    
    return 0;
}