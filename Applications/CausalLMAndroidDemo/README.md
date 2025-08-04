# CausalLM Android Demo App

This is an Android demo application that runs CausalLM models directly on the device using JNI (Java Native Interface). The app allows users to select models, input prompts, and generate text using various language models.

## Features

- **On-Device Execution**: Runs models directly on Android device without server
- **Model Selection**: Choose from available models via dropdown menu
- **Text Input**: Enter prompts for the model to process
- **Real-time Generation**: Generate text locally on device
- **Modern UI**: Material Design 3 interface with cards and smooth interactions

## Architecture

- **Native Layer**: C++ code using JNI to integrate CausalLM
- **Android Layer**: Kotlin-based Android app
- **Model Loading**: Dynamically loads models from device storage
- **Memory Management**: Efficient model loading and unloading

## Prerequisites

1. **Android NDK**: Required for building native code
2. **CMake**: Version 3.18.1 or higher
3. **Android Studio**: Latest version recommended
4. **Model Files**: CausalLM model files in proper format

## Build Instructions

### 1. Prepare the Environment

```bash
# Ensure Android NDK is installed
# In Android Studio: Tools -> SDK Manager -> SDK Tools -> NDK

# Clone the repository
cd /workspace/Applications/CausalLMAndroidDemo
```

### 2. Build nntrainer for Android

Before building the app, you need to build nntrainer libraries for Android:

```bash
cd /workspace
# Build nntrainer for Android using appropriate toolchain
# This step requires Android NDK cross-compilation setup
```

### 3. Build the Android App

```bash
cd /workspace/Applications/CausalLMAndroidDemo

# Build debug APK
./gradlew assembleDebug

# Or build in Android Studio
```

## Model Setup

### Model Directory Structure

Place your models in the following structure on your Android device:

```
/sdcard/Download/models/
├── model_name_1/
│   ├── config.json
│   ├── generation_config.json
│   ├── nntr_config.json
│   └── model_weights.bin
└── model_name_2/
    ├── config.json
    ├── generation_config.json
    ├── nntr_config.json
    └── model_weights.bin
```

### Supported Model Types

- LlamaForCausalLM
- Qwen3ForCausalLM
- Qwen3MoeForCausalLM
- Qwen3SlimMoeForCausalLM

## Usage

1. **Grant Permissions**: App will request storage permissions on first launch
2. **Model Scanning**: App automatically scans for models in:
   - `/sdcard/Download/models/`
   - App's private storage
3. **Select Model**: Choose a model from the dropdown
4. **Enter Prompt**: Type your prompt in the text field
5. **Generate**: Tap "Generate" to run the model

## Technical Details

### JNI Interface

The app uses JNI to bridge between Kotlin/Java and C++ code:

```kotlin
object CausalLMNative {
    external fun loadModel(modelName: String, modelPath: String): Boolean
    external fun generateText(modelName: String, prompt: String, doSample: Boolean): String
    external fun getLoadedModels(): Array<String>
    external fun unloadModel(modelName: String)
}
```

### Memory Management

- Models are loaded on-demand
- Only one model is kept in memory at a time
- Models are automatically unloaded when app is destroyed

### Build Configuration

The app uses CMake for native code compilation:
- C++17 standard
- Supports ARM64 and ARMv7 architectures
- Links against nntrainer and tokenizer libraries

## Troubleshooting

### Build Issues
- Ensure NDK is properly installed
- Check CMakeLists.txt paths are correct
- Verify nntrainer libraries are built for Android

### Runtime Issues
- Check model files are in correct location
- Verify storage permissions are granted
- Monitor logcat for native layer errors

### Model Loading Issues
- Ensure all required config files are present
- Check model compatibility with app
- Verify sufficient device memory

## Performance Considerations

- Model loading can take significant time
- Text generation speed depends on device capabilities
- Consider model size vs device memory limitations

## Future Improvements

- Model download functionality
- Quantization support for smaller models
- Streaming text generation
- Model caching and optimization
- Support for more model architectures