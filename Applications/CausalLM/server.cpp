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
 *
 * @file	server.cpp
 * @date	23 December 2025
 * @brief	HTTP server wrapper for CausalLM application
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>

#include "httplib.h"
#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"
#include "qwen3_causallm.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"

using json = nlohmann::json;

class CausalLMServer {
private:
    std::map<std::string, std::unique_ptr<causallm::CausalLM>> models;
    std::map<std::string, std::string> model_paths;
    std::mutex model_mutex;
    httplib::Server server;

    void registerModels() {
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
    }

    bool loadModel(const std::string& model_name, const std::string& model_path) {
        try {
            json cfg = causallm::LoadJsonFile(model_path + "/config.json");
            json generation_cfg = causallm::LoadJsonFile(model_path + "/generation_config.json");
            json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

            const std::string weight_file = model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();

            auto model = causallm::Factory::Instance().create(
                cfg["architectures"].get<std::vector<std::string>>()[0], 
                cfg, generation_cfg, nntr_cfg);
            
            model->initialize();
            model->load_weight(weight_file);

            std::lock_guard<std::mutex> lock(model_mutex);
            models[model_name] = std::move(model);
            model_paths[model_name] = model_path;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load model " << model_name << ": " << e.what() << std::endl;
            return false;
        }
    }

public:
    CausalLMServer() {
        registerModels();
    }

    void setupRoutes() {
        // CORS headers for Android app
        server.set_post_routing_handler([](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
        });

        // Handle OPTIONS requests for CORS
        server.Options(".*", [](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            res.status = 200;
        });

        // Get available models
        server.Get("/models", [this](const httplib::Request& req, httplib::Response& res) {
            json response;
            response["models"] = json::array();
            
            std::lock_guard<std::mutex> lock(model_mutex);
            for (const auto& [name, _] : models) {
                response["models"].push_back(name);
            }
            
            res.set_content(response.dump(), "application/json");
        });

        // Load a new model
        server.Post("/models/load", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                json request = json::parse(req.body);
                std::string model_name = request["model_name"];
                std::string model_path = request["model_path"];

                if (loadModel(model_name, model_path)) {
                    json response;
                    response["status"] = "success";
                    response["message"] = "Model loaded successfully";
                    res.set_content(response.dump(), "application/json");
                } else {
                    json response;
                    response["status"] = "error";
                    response["message"] = "Failed to load model";
                    res.status = 500;
                    res.set_content(response.dump(), "application/json");
                }
            } catch (const std::exception& e) {
                json response;
                response["status"] = "error";
                response["message"] = e.what();
                res.status = 400;
                res.set_content(response.dump(), "application/json");
            }
        });

        // Generate text
        server.Post("/generate", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                json request = json::parse(req.body);
                std::string model_name = request["model"];
                std::string prompt = request["prompt"];
                bool do_sample = request.value("do_sample", false);

                std::lock_guard<std::mutex> lock(model_mutex);
                
                if (models.find(model_name) == models.end()) {
                    json response;
                    response["status"] = "error";
                    response["message"] = "Model not found";
                    res.status = 404;
                    res.set_content(response.dump(), "application/json");
                    return;
                }

                // Run the model
                models[model_name]->run(prompt, do_sample);

                // Get the generated text from output_list
                std::string generated_text;
                auto& output_list = models[model_name]->output_list;
                if (!output_list.empty()) {
                    // Concatenate all outputs
                    for (const auto& output : output_list) {
                        generated_text += output;
                    }
                } else {
                    generated_text = "No output generated";
                }

                json response;
                response["status"] = "success";
                response["generated_text"] = generated_text;
                response["model"] = model_name;
                
                res.set_content(response.dump(), "application/json");
            } catch (const std::exception& e) {
                json response;
                response["status"] = "error";
                response["message"] = e.what();
                res.status = 500;
                res.set_content(response.dump(), "application/json");
            }
        });

        // Health check
        server.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
            json response;
            response["status"] = "healthy";
            res.set_content(response.dump(), "application/json");
        });
    }

    void loadDefaultModels(const std::vector<std::pair<std::string, std::string>>& default_models) {
        for (const auto& [name, path] : default_models) {
            std::cout << "Loading model: " << name << " from " << path << std::endl;
            if (loadModel(name, path)) {
                std::cout << "Successfully loaded " << name << std::endl;
            }
        }
    }

    void start(int port = 8080) {
        std::cout << "Starting CausalLM server on port " << port << std::endl;
        server.listen("0.0.0.0", port);
    }
};

int main(int argc, char* argv[]) {
    int port = 8080;
    
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }

    CausalLMServer server;
    server.setupRoutes();

    // Load default models if paths are provided
    if (argc > 2) {
        std::vector<std::pair<std::string, std::string>> default_models;
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 < argc) {
                default_models.emplace_back(argv[i], argv[i + 1]);
            }
        }
        server.loadDefaultModels(default_models);
    }

    server.start(port);
    
    return 0;
}