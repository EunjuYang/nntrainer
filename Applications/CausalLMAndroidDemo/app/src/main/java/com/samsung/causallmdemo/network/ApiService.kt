package com.samsung.causallmdemo.network

import retrofit2.Response
import retrofit2.http.*

interface ApiService {
    @GET("models")
    suspend fun getModels(): Response<ModelsResponse>

    @POST("models/load")
    suspend fun loadModel(@Body request: LoadModelRequest): Response<ApiResponse>

    @POST("generate")
    suspend fun generateText(@Body request: GenerateRequest): Response<GenerateResponse>

    @GET("health")
    suspend fun checkHealth(): Response<HealthResponse>
}

// Data classes for API communication
data class ModelsResponse(
    val models: List<String>
)

data class LoadModelRequest(
    val model_name: String,
    val model_path: String
)

data class GenerateRequest(
    val model: String,
    val prompt: String,
    val do_sample: Boolean = false
)

data class GenerateResponse(
    val status: String,
    val generated_text: String?,
    val model: String?,
    val message: String?
)

data class ApiResponse(
    val status: String,
    val message: String
)

data class HealthResponse(
    val status: String
)