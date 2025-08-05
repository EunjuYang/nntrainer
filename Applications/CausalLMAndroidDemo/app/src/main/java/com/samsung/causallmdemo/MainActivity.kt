package com.samsung.causallmdemo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Environment
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.samsung.causallmdemo.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var selectedModel: String? = null
    private var currentLoadedModel: String? = null
    private val availableModels = mutableListOf<String>()
    private val modelPaths = mutableMapOf<String, String>()
    
    companion object {
        private const val PERMISSION_REQUEST_CODE = 100
        private val MODEL_TYPES = listOf(
            "LlamaForCausalLM",
            "Qwen3ForCausalLM", 
            "Qwen3MoeForCausalLM",
            "Qwen3SlimMoeForCausalLM"
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        checkPermissions()
        setupUI()
        scanForModels()
    }

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                PERMISSION_REQUEST_CODE
            )
        }
    }

    private fun setupUI() {
        // Setup spinner
        binding.modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position > 0) {
                    val newModel = availableModels[position - 1]
                    if (selectedModel != newModel) {
                        selectedModel = newModel
                        // Preload model when selected
                        preloadModel(newModel)
                    }
                } else {
                    selectedModel = null
                    updateModelStatus("No model selected")
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                selectedModel = null
                updateModelStatus("No model selected")
            }
        }

        // Setup generate button
        binding.generateButton.setOnClickListener {
            generateText()
        }
        
        // Initially disable generate button
        binding.generateButton.isEnabled = false
    }

    private fun scanForModels() {
        lifecycleScope.launch {
            try {
                showLoading(true, "Scanning for models...")
                
                withContext(Dispatchers.IO) {
                    // Scan for models in the Downloads directory
                    val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                    val modelDir = File(downloadsDir, "models")
                    
                    if (modelDir.exists() && modelDir.isDirectory) {
                        modelDir.listFiles()?.forEach { dir ->
                            if (dir.isDirectory) {
                                // Check if it's a valid model directory
                                val configFile = File(dir, "config.json")
                                val generationConfigFile = File(dir, "generation_config.json")
                                val nntrConfigFile = File(dir, "nntr_config.json")
                                
                                if (configFile.exists() && generationConfigFile.exists() && nntrConfigFile.exists()) {
                                    val modelName = dir.name
                                    availableModels.add(modelName)
                                    modelPaths[modelName] = dir.absolutePath
                                }
                            }
                        }
                    }
                    
                    // Also check app's private storage
                    val appModelDir = File(filesDir, "models")
                    if (appModelDir.exists() && appModelDir.isDirectory) {
                        appModelDir.listFiles()?.forEach { dir ->
                            if (dir.isDirectory) {
                                val configFile = File(dir, "config.json")
                                if (configFile.exists()) {
                                    val modelName = dir.name
                                    if (!availableModels.contains(modelName)) {
                                        availableModels.add(modelName)
                                        modelPaths[modelName] = dir.absolutePath
                                    }
                                }
                            }
                        }
                    }
                }
                
                updateModelSpinner()
                
                if (availableModels.isEmpty()) {
                    showError("No models found. Please place model files in Downloads/models/")
                    updateModelStatus("No models found")
                } else {
                    updateModelStatus("${availableModels.size} models found")
                }
            } catch (e: Exception) {
                showError("Error scanning for models: ${e.message}")
                updateModelStatus("Error scanning models")
            } finally {
                showLoading(false)
            }
        }
    }

    private fun updateModelSpinner() {
        val spinnerItems = mutableListOf("Select a model to load...")
        spinnerItems.addAll(availableModels)
        
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, spinnerItems)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.modelSpinner.adapter = adapter
    }

    private fun preloadModel(modelName: String) {
        lifecycleScope.launch {
            try {
                showLoading(true, "Loading model: $modelName")
                binding.generateButton.isEnabled = false
                
                val success = withContext(Dispatchers.IO) {
                    // Check if we need to unload previous model
                    if (currentLoadedModel != null && currentLoadedModel != modelName) {
                        val modelType = detectModelType(currentLoadedModel!!)
                        CausalLMNative.unloadModel(modelType)
                        currentLoadedModel = null
                    }
                    
                    // Check if model is already loaded
                    val loadedModels = CausalLMNative.getLoadedModels()
                    val modelType = detectModelType(modelName)
                    
                    if (!loadedModels.contains(modelType)) {
                        // Load the model
                        val modelPath = modelPaths[modelName] ?: throw Exception("Model path not found")
                        
                        if (!CausalLMNative.loadModel(modelType, modelPath)) {
                            throw Exception("Failed to load model")
                        }
                    }
                    
                    currentLoadedModel = modelName
                    true
                }
                
                if (success) {
                    updateModelStatus("Model loaded: $modelName")
                    binding.generateButton.isEnabled = true
                    showToast("Model loaded successfully!")
                }
            } catch (e: Exception) {
                showError("Failed to load model: ${e.message}")
                updateModelStatus("Failed to load model")
                binding.generateButton.isEnabled = false
                selectedModel = null
                binding.modelSpinner.setSelection(0)
            } finally {
                showLoading(false)
            }
        }
    }

    private fun generateText() {
        val prompt = binding.promptInput.text?.toString()?.trim()
        
        if (prompt.isNullOrEmpty()) {
            showError("Please enter a prompt")
            return
        }
        
        if (selectedModel == null || currentLoadedModel == null) {
            showError("Please select and load a model first")
            return
        }

        lifecycleScope.launch {
            try {
                showLoading(true, "Generating text...")
                binding.outputCard.visibility = View.GONE
                
                val result = withContext(Dispatchers.IO) {
                    // Generate text using the preloaded model
                    val modelType = detectModelType(currentLoadedModel!!)
                    CausalLMNative.generateText(modelType, prompt, false)
                }
                
                displayOutput(result)
            } catch (e: Exception) {
                showError("Error generating text: ${e.message}")
            } finally {
                showLoading(false)
            }
        }
    }

    private fun detectModelType(modelName: String): String {
        // Try to detect model type from the model name
        return when {
            modelName.contains("qwen3", ignoreCase = true) && modelName.contains("slim", ignoreCase = true) && modelName.contains("moe", ignoreCase = true) -> "Qwen3SlimMoeForCausalLM"
            modelName.contains("qwen3", ignoreCase = true) && modelName.contains("moe", ignoreCase = true) -> "Qwen3MoeForCausalLM"
            modelName.contains("qwen3", ignoreCase = true) -> "Qwen3ForCausalLM"
            modelName.contains("llama", ignoreCase = true) -> "LlamaForCausalLM"
            else -> "Qwen3ForCausalLM" // Default
        }
    }

    private fun displayOutput(text: String) {
        binding.outputCard.visibility = View.VISIBLE
        binding.outputText.text = text
    }

    private fun showLoading(show: Boolean, message: String = "Loading...") {
        binding.progressBar.visibility = if (show) View.VISIBLE else View.GONE
        binding.loadingText.visibility = if (show) View.VISIBLE else View.GONE
        binding.loadingText.text = message
        binding.generateButton.isEnabled = !show && currentLoadedModel != null
        binding.modelSpinner.isEnabled = !show
    }

    private fun updateModelStatus(status: String) {
        binding.modelStatusText.text = status
    }

    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }
    
    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        // Unload all models when app is destroyed
        lifecycleScope.launch(Dispatchers.IO) {
            CausalLMNative.getLoadedModels().forEach { model ->
                CausalLMNative.unloadModel(model)
            }
        }
    }
}