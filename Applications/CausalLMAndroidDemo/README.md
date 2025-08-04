# CausalLM Android Demo App

This is an Android demo application that interfaces with the CausalLM backend server to provide a user-friendly interface for text generation using various language models.

## Features

- **Model Selection**: Choose from available language models via dropdown menu
- **Text Input**: Enter prompts for the model to process
- **Real-time Generation**: Send requests to the backend and display generated text
- **Modern UI**: Material Design 3 interface with cards and smooth interactions

## Architecture

### Backend (C++ HTTP Server)
- Located in `/workspace/Applications/CausalLM/server.cpp`
- Provides REST API endpoints:
  - `GET /models` - List available models
  - `POST /models/load` - Load a new model
  - `POST /generate` - Generate text from prompt
  - `GET /health` - Health check

### Android App
- **Language**: Kotlin
- **UI**: Material Design Components
- **Networking**: Retrofit2 with Coroutines
- **Architecture**: Simple MVVM pattern

## Setup Instructions

### 1. Build and Run the Backend Server

```bash
cd /workspace/Applications/CausalLM

# Build the server (assuming meson build system is configured)
meson build
cd build
ninja

# Run the server
./nntr_causallm_server [port] [model_name model_path ...]

# Example:
./nntr_causallm_server 8080 "Qwen3ForCausalLM" "/path/to/qwen3/model"
```

### 2. Configure Android App

1. **Update Server URL** (if not using emulator):
   - Edit `/app/src/main/java/com/samsung/causallmdemo/network/ApiClient.kt`
   - Change `BASE_URL` from `http://10.0.2.2:8080/` to your server's IP address

2. **Build the Android App**:
   ```bash
   cd /workspace/Applications/CausalLMAndroidDemo
   ./gradlew assembleDebug
   ```

3. **Install on Device/Emulator**:
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```

## Usage

1. Start the backend server with desired models loaded
2. Launch the Android app
3. Select a model from the dropdown
4. Enter your prompt in the text field
5. Tap "Generate" to get the model's response

## Network Configuration

### For Android Emulator
- Default configuration uses `10.0.2.2:8080` (host machine's localhost)
- No changes needed if running server on the same machine

### For Physical Device
1. Ensure device and server are on the same network
2. Update `BASE_URL` in `ApiClient.kt` to server's IP address
3. Server must be accessible from the device's network

## API Endpoints

### Get Available Models
```
GET /models
Response: {"models": ["LlamaForCausalLM", "Qwen3ForCausalLM", ...]}
```

### Generate Text
```
POST /generate
Body: {
  "model": "Qwen3ForCausalLM",
  "prompt": "Hello, world!",
  "do_sample": false
}
Response: {
  "status": "success",
  "generated_text": "Generated response...",
  "model": "Qwen3ForCausalLM"
}
```

## Troubleshooting

### Connection Issues
- Check if server is running and accessible
- Verify firewall settings allow connections on the server port
- For physical devices, ensure correct IP address is configured

### Model Loading Issues
- Verify model files exist at specified paths
- Check server logs for detailed error messages
- Ensure sufficient memory for model loading

### Build Issues
- Ensure Android SDK is properly installed
- Check Gradle version compatibility
- Verify all dependencies are available

## Future Improvements

- Add model configuration options
- Implement streaming responses
- Add conversation history
- Support for multiple concurrent requests
- Model performance metrics display