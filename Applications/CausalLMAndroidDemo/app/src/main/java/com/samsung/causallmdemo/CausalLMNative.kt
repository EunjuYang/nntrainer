package com.samsung.causallmdemo

object CausalLMNative {
    init {
        System.loadLibrary("causallm-native")
    }

    external fun loadModel(modelName: String, modelPath: String): Boolean
    external fun generateText(modelName: String, prompt: String, doSample: Boolean): String
    external fun getLoadedModels(): Array<String>
    external fun unloadModel(modelName: String)
}