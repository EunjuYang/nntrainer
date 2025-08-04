package com.samsung.causallmdemo

import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.samsung.causallmdemo.databinding.ActivityMainBinding
import com.samsung.causallmdemo.network.ApiClient
import com.samsung.causallmdemo.network.GenerateRequest
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var selectedModel: String? = null
    private val availableModels = mutableListOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
        loadAvailableModels()
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

    private fun loadAvailableModels() {
        lifecycleScope.launch {
            try {
                showLoading(true)
                val response = ApiClient.apiService.getModels()
                
                if (response.isSuccessful) {
                    response.body()?.let { modelsResponse ->
                        availableModels.clear()
                        availableModels.addAll(modelsResponse.models)
                        updateModelSpinner()
                    }
                } else {
                    showError("Failed to load models: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Network error: ${e.message}")
                // Add some default models for testing
                availableModels.clear()
                availableModels.addAll(listOf("LlamaForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "Qwen3SlimMoeForCausalLM"))
                updateModelSpinner()
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
                
                val request = GenerateRequest(
                    model = selectedModel!!,
                    prompt = prompt,
                    do_sample = false
                )
                
                val response = ApiClient.apiService.generateText(request)
                
                if (response.isSuccessful) {
                    response.body()?.let { generateResponse ->
                        if (generateResponse.status == "success") {
                            displayOutput(generateResponse.generated_text ?: "No output generated")
                        } else {
                            showError(generateResponse.message ?: "Generation failed")
                        }
                    }
                } else {
                    showError("Failed to generate text: ${response.message()}")
                }
            } catch (e: Exception) {
                showError("Network error: ${e.message}")
            } finally {
                showLoading(false)
            }
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
}