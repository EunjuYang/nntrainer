/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <jni.h>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <android/log.h>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"
#include "qwen3_causallm.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"

#define LOG_TAG "CausalLM-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using json = nlohmann::json;

class CausalLMManager {
private:
    std::map<std::string, std::unique_ptr<causallm::CausalLM>> models;
    std::mutex model_mutex;
    bool is_initialized = false;

    void registerModels() {
        if (is_initialized) return;
        
        causallm::Factory::Instance().registerModel(
            "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
                return std::make_unique<causallm::CausalLM>(cfg, generation_cfg, nntr_cfg);
            });
        causallm::Factory::Instance().registerModel(
            "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
                return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg, nntr_cfg);
            });
        causallm::Factory::Instance().registerModel(
            "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
                return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg, nntr_cfg);
            });
        causallm::Factory::Instance().registerModel(
            "Qwen3SlimMoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
                return std::make_unique<causallm::Qwen3SlimMoECausalLM>(cfg, generation_cfg, nntr_cfg);
            });
        
        is_initialized = true;
    }

public:
    static CausalLMManager& getInstance() {
        static CausalLMManager instance;
        return instance;
    }

    CausalLMManager() {
        registerModels();
    }

    bool loadModel(const std::string& model_name, const std::string& model_path) {
        try {
            LOGI("Loading model %s from %s", model_name.c_str(), model_path.c_str());
            
            json cfg = causallm::LoadJsonFile(model_path + "/config.json");
            json generation_cfg = causallm::LoadJsonFile(model_path + "/generation_config.json");
            json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

            const std::string weight_file = model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
            
            LOGI("Creating model instance...");
            auto model = causallm::Factory::Instance().create(
                cfg["architectures"].get<std::vector<std::string>>()[0], 
                cfg, generation_cfg, nntr_cfg);
            
            LOGI("Initializing model...");
            model->initialize();
            
            LOGI("Loading weights from %s", weight_file.c_str());
            model->load_weight(weight_file);

            std::lock_guard<std::mutex> lock(model_mutex);
            models[model_name] = std::move(model);
            
            LOGI("Model %s loaded successfully", model_name.c_str());
            return true;
        } catch (const std::exception& e) {
            LOGE("Failed to load model %s: %s", model_name.c_str(), e.what());
            return false;
        }
    }

    std::string generateText(const std::string& model_name, const std::string& prompt, bool do_sample) {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        auto it = models.find(model_name);
        if (it == models.end()) {
            LOGE("Model %s not found", model_name.c_str());
            return "Error: Model not found";
        }

        try {
            LOGI("Running model %s with prompt: %s", model_name.c_str(), prompt.c_str());
            
            // Run the model
            it->second->run(prompt, do_sample);
            
            // Get the generated text from output_list
            std::string generated_text;
            auto& output_list = it->second->output_list;
            if (!output_list.empty()) {
                for (const auto& output : output_list) {
                    generated_text += output;
                }
            } else {
                generated_text = "No output generated";
            }
            
            LOGI("Generation completed");
            return generated_text;
        } catch (const std::exception& e) {
            LOGE("Error during generation: %s", e.what());
            return std::string("Error: ") + e.what();
        }
    }

    std::vector<std::string> getLoadedModels() {
        std::lock_guard<std::mutex> lock(model_mutex);
        std::vector<std::string> model_names;
        for (const auto& [name, _] : models) {
            model_names.push_back(name);
        }
        return model_names;
    }

    void unloadModel(const std::string& model_name) {
        std::lock_guard<std::mutex> lock(model_mutex);
        models.erase(model_name);
        LOGI("Model %s unloaded", model_name.c_str());
    }
};

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_samsung_causallmdemo_CausalLMNative_loadModel(
    JNIEnv* env,
    jobject /* this */,
    jstring modelName,
    jstring modelPath) {
    
    const char* model_name = env->GetStringUTFChars(modelName, nullptr);
    const char* model_path = env->GetStringUTFChars(modelPath, nullptr);
    
    bool result = CausalLMManager::getInstance().loadModel(model_name, model_path);
    
    env->ReleaseStringUTFChars(modelName, model_name);
    env->ReleaseStringUTFChars(modelPath, model_path);
    
    return result ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_com_samsung_causallmdemo_CausalLMNative_generateText(
    JNIEnv* env,
    jobject /* this */,
    jstring modelName,
    jstring prompt,
    jboolean doSample) {
    
    const char* model_name = env->GetStringUTFChars(modelName, nullptr);
    const char* prompt_text = env->GetStringUTFChars(prompt, nullptr);
    
    std::string result = CausalLMManager::getInstance().generateText(
        model_name, prompt_text, doSample == JNI_TRUE);
    
    env->ReleaseStringUTFChars(modelName, model_name);
    env->ReleaseStringUTFChars(prompt, prompt_text);
    
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT jobjectArray JNICALL
Java_com_samsung_causallmdemo_CausalLMNative_getLoadedModels(
    JNIEnv* env,
    jobject /* this */) {
    
    std::vector<std::string> models = CausalLMManager::getInstance().getLoadedModels();
    
    jobjectArray result = env->NewObjectArray(models.size(), 
        env->FindClass("java/lang/String"), nullptr);
    
    for (size_t i = 0; i < models.size(); i++) {
        env->SetObjectArrayElement(result, i, env->NewStringUTF(models[i].c_str()));
    }
    
    return result;
}

JNIEXPORT void JNICALL
Java_com_samsung_causallmdemo_CausalLMNative_unloadModel(
    JNIEnv* env,
    jobject /* this */,
    jstring modelName) {
    
    const char* model_name = env->GetStringUTFChars(modelName, nullptr);
    CausalLMManager::getInstance().unloadModel(model_name);
    env->ReleaseStringUTFChars(modelName, model_name);
}

} // extern "C"