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
                selectedModel = if (position > 0) availableModels[position - 1] else null
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                selectedModel = null
            }
        }

        // Setup generate button
        binding.generateButton.setOnClickListener {
            generateText()
        }
    }

    private fun scanForModels() {
        lifecycleScope.launch {
            try {
                showLoading(true)
                
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
                }
            } catch (e: Exception) {
                showError("Error scanning for models: ${e.message}")
            } finally {
                showLoading(false)
            }
        }
    }

    private fun updateModelSpinner() {
        val spinnerItems = mutableListOf("Select a model...")
        spinnerItems.addAll(availableModels)
        
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, spinnerItems)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.modelSpinner.adapter = adapter
    }

    private fun generateText() {
        val prompt = binding.promptInput.text?.toString()?.trim()
        
        if (prompt.isNullOrEmpty()) {
            showError("Please enter a prompt")
            return
        }
        
        if (selectedModel == null) {
            showError("Please select a model")
            return
        }

        lifecycleScope.launch {
            try {
                showLoading(true)
                binding.outputCard.visibility = View.GONE
                
                val result = withContext(Dispatchers.IO) {
                    // Check if model is already loaded
                    val loadedModels = CausalLMNative.getLoadedModels()
                    if (!loadedModels.contains(selectedModel)) {
                        // Load the model
                        val modelPath = modelPaths[selectedModel!!] ?: throw Exception("Model path not found")
                        val modelType = detectModelType(selectedModel!!)
                        
                        if (!CausalLMNative.loadModel(modelType, modelPath)) {
                            throw Exception("Failed to load model")
                        }
                    }
                    
                    // Generate text
                    CausalLMNative.generateText(detectModelType(selectedModel!!), prompt, false)
                }
                
                displayOutput(result)
            } catch (e: Exception) {
                showError("Error: ${e.message}")
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

    private fun showLoading(show: Boolean) {
        binding.progressBar.visibility = if (show) View.VISIBLE else View.GONE
        binding.generateButton.isEnabled = !show
    }

    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
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