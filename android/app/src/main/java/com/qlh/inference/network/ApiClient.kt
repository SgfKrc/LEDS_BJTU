package com.qlh.inference.network

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.qlh.inference.BuildConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.logging.HttpLoggingInterceptor
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.TimeoutException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

// ================================================================
// DTO — 匹配 PC 端 api_server.py 的 ChatRequest / ChatResponse
// ================================================================

data class ChatRequest(
    val message: String,
    /** PC /api/chat 的多模态输入；空列表保持原有纯文本请求语义。 */
    @SerializedName("image_data_urls")
    val imageDataUrls: List<String> = emptyList(),
    @SerializedName("max_new_tokens")
    val maxNewTokens: Int = 1024,
    val temperature: Float = 0.7f,
    @SerializedName("top_p")
    val topP: Float = 0.9f,
    @SerializedName("show_thinking")
    val showThinking: Boolean = false,
    @SerializedName("session_id")
    val sessionId: String? = null,
    @SerializedName("streaming_mode")
    val streamingMode: String = "full",    // "full" | "fast"（对应 PC 端 streaming_mode）
    @SerializedName("client_node_id")
    val clientNodeId: String? = null,
    @SerializedName("client_node_type")
    val clientNodeType: String? = null,
    @SerializedName("client_mode")
    val clientMode: String? = null,
    @SerializedName("client_app_variant")
    val clientAppVariant: String? = null,
    /** Only populated for an image request; PC requires explicit multimodal routing consent. */
    @SerializedName("allow_external")
    val allowExternal: Boolean? = null,
    @SerializedName("prefer_external")
    val preferExternal: Boolean? = null,
)

data class ChatResponse(
    val content: String = "",
    val metrics: Map<String, Any>? = null,
    val followups: List<String>? = null,
    val error: String? = null
)

data class SessionInfo(
    val id: String = "",
    val title: String = "新对话",
    @SerializedName("message_count")
    val messageCount: Int = 0,
    @SerializedName("created_at")
    val createdAt: String? = null,
    @SerializedName("updated_at")
    val updatedAt: String? = null
)

data class ClusterStatus(
    val master: ClusterNode? = null,
    val nodes: List<ClusterNode>? = null,
    @SerializedName("online_count")
    val onlineCount: Int = 0,
    @SerializedName("total_count")
    val totalCount: Int = 0,
    @SerializedName("distributed_enabled")
    val distributedEnabled: Boolean = false
)

data class ClusterNode(
    @SerializedName("node_id")
    val nodeId: String = "",
    val role: String = "",
    @SerializedName("node_type")
    val nodeType: String = "pc",
    val state: String = "",
    val hostname: String = "",
    val address: String = "",
    @SerializedName("network_type")
    val networkType: String = "",
    @SerializedName("task_count")
    val taskCount: Int = 0,
    @SerializedName("avg_rtt_ms")
    val avgRttMs: Float = 0f
)

data class RegisterNodeRequest(
    @SerializedName("node_id")
    val nodeId: String,
    val hostname: String,
    val address: String = "",
    @SerializedName("network_type")
    val networkType: String = "unknown",
    @SerializedName("node_type")
    val nodeType: String = "android",
    @SerializedName("device_info")
    val deviceInfo: Map<String, Any?> = emptyMap(),
    @SerializedName("client_mode")
    val clientMode: String = "thin",
    @SerializedName("app_variant")
    val appVariant: String = if (BuildConfig.IS_LITE) "lite" else "full",
    @SerializedName("app_version")
    val appVersion: String = BuildConfig.VERSION_NAME
)

data class RegisterNodeResponse(
    val status: String = "",
    @SerializedName("node_id")
    val nodeId: String? = null,
    val message: String? = null,
    val state: String? = null
)

data class BootstrapRequest(
    @SerializedName("node_id")
    val nodeId: String,
    @SerializedName("node_type")
    val nodeType: String = "android",
    val hostname: String = "",
    val platform: String = "android",
    @SerializedName("app_variant")
    val appVariant: String = if (BuildConfig.IS_LITE) "lite" else "full",
    @SerializedName("app_version")
    val appVersion: String = BuildConfig.VERSION_NAME,
    val capabilities: Map<String, Any?> = emptyMap()
)

data class BootstrapCluster(
    @SerializedName("cluster_id")
    val clusterId: String = "",
    @SerializedName("master_api_host")
    val masterApiHost: String = "",
    @SerializedName("master_api_port")
    val masterApiPort: Int = 8000,
    @SerializedName("master_tcp_host")
    val masterTcpHost: String = "",
    @SerializedName("master_tcp_port")
    val masterTcpPort: Int = 8888,
    @SerializedName("cluster_secret")
    val clusterSecret: String = ""
)

data class BootstrapNode(
    @SerializedName("node_id")
    val nodeId: String = "",
    val role: String = "client",
    @SerializedName("node_type")
    val nodeType: String = "android",
    @SerializedName("pipeline_worker")
    val pipelineWorker: Boolean = false
)

data class BootstrapAndroid(
    @SerializedName("presence_interval_seconds")
    val presenceIntervalSeconds: Int = 45,
    @SerializedName("pipeline_worker")
    val pipelineWorker: Boolean = false,
    @SerializedName("model_manifest_url")
    val modelManifestUrl: String = ""
)

data class BootstrapResponse(
    val status: String = "",
    val cluster: BootstrapCluster = BootstrapCluster(),
    val node: BootstrapNode = BootstrapNode(),
    val android: BootstrapAndroid = BootstrapAndroid()
)

// ================================================================
// Remote SD DTOs (PC /api/diffusion/*)
// ================================================================

const val DIFFUSION_MAX_UPLOAD_BYTES = 16 * 1024 * 1024
const val DIFFUSION_MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024

data class DiffusionGenerateRequest(
    @SerializedName("preset_id") val presetId: String? = null,
    val prompt: String? = null,
    @SerializedName("negative_prompt") val negativePrompt: String? = null,
    val seed: Int? = null,
    val width: Int? = null,
    val height: Int? = null,
    val steps: Int? = null,
    @SerializedName("guidance_scale") val guidanceScale: Float? = null,
    val scheduler: String? = null,
)

data class DiffusionEditRequest(
    val mode: String,
    @SerializedName("preset_id") val presetId: String? = null,
    @SerializedName("source_blob_id") val sourceBlobId: String,
    @SerializedName("mask_blob_id") val maskBlobId: String? = null,
    val prompt: String? = null,
    @SerializedName("negative_prompt") val negativePrompt: String? = null,
    val seed: Int? = null,
    val width: Int? = null,
    val height: Int? = null,
    val steps: Int? = null,
    @SerializedName("guidance_scale") val guidanceScale: Float? = null,
    val scheduler: String? = null,
    val strength: Float = 0.75f,
    val instruction: String? = null,
    @SerializedName("edit_adapter_id") val editAdapterId: String? = null,
    @SerializedName("conditioning_scale") val conditioningScale: Float? = null,
    @SerializedName("image_guidance_scale") val imageGuidanceScale: Float? = null,
    @SerializedName("ip_adapter_scale") val ipAdapterScale: Float? = null,
)

data class DiffusionBlobUpload(
    val data: ByteArray,
    val fileName: String = "input.png",
    val contentType: String = "image/png",
    val purpose: String = "input_image",
)

data class DiffusionProgress(
    val step: Int = 0,
    val total: Int = 0,
)

data class DiffusionBlobDescriptor(
    @SerializedName("blob_id") val blobId: String = "",
    @SerializedName("content_type") val contentType: String = "",
    @SerializedName("size_bytes") val sizeBytes: Long = 0L,
    val sha256: String = "",
    val width: Int? = null,
    val height: Int? = null,
    val purpose: String? = null,
    @SerializedName("created_at") val createdAt: Double? = null,
    @SerializedName("expires_at") val expiresAt: Double? = null,
)

data class DiffusionJob(
    @SerializedName("job_id") val jobId: String = "",
    @SerializedName("artifact_id") val artifactId: String = "",
    val kind: String = "generate",
    @SerializedName("owner_scope") val ownerScope: String = "local",
    @SerializedName("input_blob_ids") val inputBlobIds: List<String> = emptyList(),
    val state: String = "queued",
    @SerializedName("created_at") val createdAt: Double? = null,
    @SerializedName("started_at") val startedAt: Double? = null,
    @SerializedName("completed_at") val completedAt: Double? = null,
    val progress: DiffusionProgress = DiffusionProgress(),
    @SerializedName("cancel_requested") val cancelRequested: Boolean = false,
    val parameters: Map<String, Any?> = emptyMap(),
    val blob: DiffusionBlobDescriptor? = null,
    @SerializedName("output_blob_id") val outputBlobId: String? = null,
    val metrics: Map<String, Any?> = emptyMap(),
    val error: String? = null,
    @SerializedName("error_code") val errorCode: String? = null,
) {
    val isTerminal: Boolean
        get() = state == "completed" || state == "failed" || state == "cancelled"
}

data class DiffusionCancelResponse(
    val accepted: Boolean = false,
    val job: DiffusionJob = DiffusionJob(),
)

data class DiffusionBlobDownload(
    val data: ByteArray,
    val contentType: String = "application/octet-stream",
    val eTag: String? = null,
)

data class GgufModelInfo(
    val filename: String = "",
    @SerializedName("size_bytes") val sizeBytes: Long = 0L,
    @SerializedName("size_mb") val sizeMb: Double = 0.0,
    val sha256: String = "",
    @SerializedName("download_url") val downloadUrl: String = "",
)

data class GgufModelsResponse(
    val models: List<GgufModelInfo> = emptyList(),
    val exists: Boolean = false,
    val count: Int = 0,
)

// ================================================================
// API 客户端
// ================================================================

class ApiClient(
    private val baseUrl: String,
    private val gson: Gson = Gson()
) {
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)   // 推理可能较慢
        .writeTimeout(30, TimeUnit.SECONDS)
        .addInterceptor(HttpLoggingInterceptor().apply {
            // BODY 级别会打印完整聊天内容，仅 debug 构建时使用
            level = if (BuildConfig.DEBUG) {
                HttpLoggingInterceptor.Level.BODY
            } else {
                HttpLoggingInterceptor.Level.NONE
            }
        })
        .build()

    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    private val diffusionTerminalStates = setOf("completed", "failed", "cancelled")

    // ==================== 聊天 ====================

    /** 发送消息并等待完整回复（非流式） */
    suspend fun chat(request: ChatRequest): Result<ChatResponse> = withContext(Dispatchers.IO) {
        try {
            val body = gson.toJson(request).toRequestBody(jsonMediaType)
            val httpRequest = Request.Builder()
                .url("$baseUrl/api/chat")
                .post(body)
                .header("Content-Type", "application/json")
                .build()

            val response = executeAsync(httpRequest)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(
                    IOException("HTTP ${response.code}: $responseBody")
                )
            }
            val chatResponse = gson.fromJson(responseBody, ChatResponse::class.java)
            Result.success(chatResponse)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ==================== 会话管理 ====================

    /** 获取服务端会话列表 */
    suspend fun getSessions(): Result<List<SessionInfo>> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/sessions")
                .get()
                .build()

            val response = executeAsync(request)
            val body = response.body?.string() ?: "[]"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }

            // 响应可能是 { "sessions": [...] } 或直接是数组
            val sessions: List<SessionInfo> = try {
                val array = gson.fromJson(body, Array<SessionInfo>::class.java)
                array.toList()
            } catch (e: Exception) {
                // 尝试解析为包装对象
                val wrapper = gson.fromJson(body, SessionsWrapper::class.java)
                wrapper.sessions ?: emptyList()
            }
            Result.success(sessions)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 创建服务端会话 */
    suspend fun createSession(title: String = "新对话"): Result<SessionInfo> = withContext(Dispatchers.IO) {
        try {
            val body = gson.toJson(mapOf("title" to title)).toRequestBody(jsonMediaType)
            val request = Request.Builder()
                .url("$baseUrl/api/sessions")
                .post(body)
                .header("Content-Type", "application/json")
                .build()

            val response = executeAsync(request)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            Result.success(gson.fromJson(responseBody, SessionInfo::class.java))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 删除服务端会话 */
    suspend fun deleteSession(sessionId: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/sessions/$sessionId")
                .delete()
                .build()

            val response = executeAsync(request)
            if (!response.isSuccessful && response.code != 404) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 获取服务端某会话的历史消息 */
    suspend fun getSessionMessages(sessionId: String): Result<List<MessageDto>> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/sessions/$sessionId/messages")
                .get()
                .build()

            val response = executeAsync(request)
            val body = response.body?.string() ?: "[]"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            val messages = gson.fromJson(body, Array<MessageDto>::class.java)
            Result.success(messages.toList())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ==================== 集群状态 ====================

    /** 获取集群状态 */
    suspend fun getClusterStatus(): Result<ClusterStatus> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/cluster/status")
                .get()
                .build()

            val response = executeAsync(request)
            val body = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}"))
            }
            Result.success(gson.fromJson(body, ClusterStatus::class.java))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 测试与主节点的连接 */
    suspend fun testConnection(): Result<Boolean> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/cluster/status")
                .get()
                .build()

            val response = executeAsync(request)
            Result.success(response.isSuccessful)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Android 薄客户端通过 HTTP 向主节点登记自身存在（非 TCP worker 注册）。 */
    suspend fun registerAndroidNode(request: RegisterNodeRequest): Result<RegisterNodeResponse> = withContext(Dispatchers.IO) {
        try {
            val body = gson.toJson(request).toRequestBody(jsonMediaType)
            val httpRequest = Request.Builder()
                .url("$baseUrl/api/cluster/android/register")
                .post(body)
                .header("Content-Type", "application/json")
                .build()

            val response = executeAsync(httpRequest)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}: $responseBody"))
            }
            Result.success(gson.fromJson(responseBody, RegisterNodeResponse::class.java))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** 首次连接自动部署：从主节点获取 Android 节点配置。 */
    suspend fun firstConnectBootstrap(request: BootstrapRequest): Result<BootstrapResponse> = withContext(Dispatchers.IO) {
        try {
            val body = gson.toJson(request).toRequestBody(jsonMediaType)
            val httpRequest = Request.Builder()
                .url("$baseUrl/api/bootstrap/first-connect")
                .post(body)
                .header("Content-Type", "application/json")
                .build()

            val response = executeAsync(httpRequest)
            val responseBody = response.body?.string() ?: "{}"
            if (!response.isSuccessful) {
                return@withContext Result.failure(IOException("HTTP ${response.code}: $responseBody"))
            }
            Result.success(gson.fromJson(responseBody, BootstrapResponse::class.java))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ==================== Remote SD ====================

    /** Submit a text-to-image job to the PC diffusion workspace. */
    suspend fun submitDiffusionGeneration(
        request: DiffusionGenerateRequest,
    ): Result<DiffusionJob> = withContext(Dispatchers.IO) {
        try {
            postDiffusionJson("/api/diffusion/generate", request, DiffusionJob::class.java)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Upload a PNG/JPEG/WebP source image used by img2img/reference/inpaint. */
    suspend fun uploadDiffusionBlob(
        upload: DiffusionBlobUpload,
    ): Result<DiffusionBlobDescriptor> = withContext(Dispatchers.IO) {
        try {
            require(upload.data.isNotEmpty()) { "image upload is empty" }
            require(upload.data.size <= DIFFUSION_MAX_UPLOAD_BYTES) {
                "image upload exceeds the 16 MiB limit"
            }
            require(upload.purpose == "input_image" || upload.purpose == "mask") {
                "purpose must be input_image or mask"
            }
            val multipart = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("purpose", upload.purpose)
                .addFormDataPart(
                    "file",
                    upload.fileName.ifBlank { "input.png" },
                    upload.data.toRequestBody(upload.contentType.toMediaType()),
                )
                .build()
            val request = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/diffusion/blobs")
                .post(multipart)
                .build()
            executeJson(request, DiffusionBlobDescriptor::class.java)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Submit an image editing job after its source blob has been uploaded. */
    suspend fun submitDiffusionEdit(
        request: DiffusionEditRequest,
    ): Result<DiffusionJob> = withContext(Dispatchers.IO) {
        try {
            postDiffusionJson("/api/diffusion/edit", request, DiffusionJob::class.java)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Read one remote SD job snapshot. */
    suspend fun getDiffusionJob(jobId: String): Result<DiffusionJob> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url(diffusionJobUrl(jobId))
                .get()
                .build()
            executeJson(request, DiffusionJob::class.java)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Ask the PC service to cancel an active SD job. */
    suspend fun cancelDiffusionJob(jobId: String): Result<DiffusionCancelResponse> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url(diffusionJobUrl(jobId, "/cancel"))
                .post(ByteArray(0).toRequestBody(null))
                .build()
            executeJson(request, DiffusionCancelResponse::class.java)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Download one completed result blob with an Android-side memory bound. */
    suspend fun downloadDiffusionBlob(blobId: String): Result<DiffusionBlobDownload> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url(diffusionBlobUrl(blobId))
                .get()
                .build()
            val response = executeAsync(request)
            if (!response.isSuccessful) {
                val body = response.body?.string() ?: ""
                return@withContext Result.failure(IOException("HTTP ${response.code}: $body"))
            }
            val body = response.body
                ?: return@withContext Result.failure(IOException("diffusion blob response has no body"))
            val contentLength = body.contentLength()
            if (contentLength > DIFFUSION_MAX_DOWNLOAD_BYTES) {
                body.close()
                return@withContext Result.failure(
                    IOException("diffusion blob exceeds the 32 MiB Android download limit"),
                )
            }
            val data = body.byteStream().use { input ->
                val output = ByteArrayOutputStream()
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    require(output.size() + count <= DIFFUSION_MAX_DOWNLOAD_BYTES) {
                        "diffusion blob exceeds the 32 MiB Android download limit"
                    }
                    output.write(buffer, 0, count)
                }
                output.toByteArray()
            }
            if (data.isEmpty()) {
                return@withContext Result.failure(IOException("diffusion blob response is empty"))
            }
            Result.success(
                DiffusionBlobDownload(
                    data = data,
                    contentType = body.contentType()?.toString() ?: "application/octet-stream",
                    eTag = response.header("ETag"),
                ),
            )
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Poll until a terminal snapshot; coroutine cancellation stops polling and its HTTP call. */
    suspend fun pollDiffusionJob(
        jobId: String,
        intervalMillis: Long = 1_000L,
        maxPolls: Int = 600,
    ): Result<DiffusionJob> = withContext(Dispatchers.IO) {
        if (intervalMillis < 0L) {
            return@withContext Result.failure(IllegalArgumentException("intervalMillis must be non-negative"))
        }
        if (maxPolls < 1) {
            return@withContext Result.failure(IllegalArgumentException("maxPolls must be positive"))
        }
        var latest: DiffusionJob? = null
        repeat(maxPolls) { index ->
            currentCoroutineContext().ensureActive()
            val result = getDiffusionJob(jobId)
            if (result.isFailure) return@withContext Result.failure(result.exceptionOrNull()!!)
            val job = result.getOrThrow()
            latest = job
            if (job.state in diffusionTerminalStates) return@withContext Result.success(job)
            if (index + 1 < maxPolls) delay(intervalMillis)
        }
        Result.failure(TimeoutException("diffusion job did not reach a terminal state: ${latest?.jobId ?: jobId}"))
    }

    // ==================== Remote GGUF catalog ====================

    suspend fun getGgufModels(): Result<List<GgufModelInfo>> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl.trimEnd('/')}/api/models/gguf")
                .get()
                .build()
            executeJson(request, GgufModelsResponse::class.java).map { response ->
                response.models.filter { model ->
                    model.filename.endsWith(".gguf", ignoreCase = true) &&
                        model.sizeBytes > 0L &&
                        model.sha256.matches(Regex("[a-fA-F0-9]{64}"))
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ==================== 内部方法 ====================

    private suspend fun executeAsync(request: Request): Response =
        suspendCancellableCoroutine { continuation ->
            val call = client.newCall(request)
            call.enqueue(object : Callback {
                override fun onResponse(call: Call, response: Response) {
                    continuation.resume(response)
                }

                override fun onFailure(call: Call, e: IOException) {
                    if (continuation.isCancelled) return
                    continuation.resumeWithException(e)
                }
            })
            continuation.invokeOnCancellation {
                call.cancel()
            }
        }

    private suspend fun <T> executeJson(request: Request, responseType: Class<T>): Result<T> {
        val response = executeAsync(request)
        val body = response.body?.string() ?: "{}"
        if (!response.isSuccessful) {
            return Result.failure(IOException("HTTP ${response.code}: $body"))
        }
        return Result.success(gson.fromJson(body, responseType))
    }

    private suspend fun <T> postDiffusionJson(
        path: String,
        payload: Any,
        responseType: Class<T>,
    ): Result<T> {
        val body = gson.toJson(payload).toRequestBody(jsonMediaType)
        val request = Request.Builder()
            .url("${baseUrl.trimEnd('/')}$path")
            .post(body)
            .header("Content-Type", "application/json")
            .build()
        return executeJson(request, responseType)
    }

    private fun diffusionJobUrl(jobId: String, suffix: String = ""): String {
        requireDiffusionResourceId(jobId, "job")
        return "${baseUrl.trimEnd('/')}/api/diffusion/jobs/$jobId$suffix"
    }

    private fun diffusionBlobUrl(blobId: String): String {
        requireDiffusionResourceId(blobId, "blob")
        return "${baseUrl.trimEnd('/')}/api/diffusion/blobs/$blobId"
    }

    private fun requireDiffusionResourceId(value: String, kind: String) {
        require(value.matches(Regex("[A-Za-z0-9._~-]{1,128}"))) {
            "invalid diffusion $kind id"
        }
    }

    // ---- DTO 包装 ----

    private data class SessionsWrapper(
        val sessions: List<SessionInfo>? = null
    )
}

/** 服务端消息 DTO */
data class MessageDto(
    val id: Long = 0,
    val role: String = "",
    val content: String = "",
    val timestamp: String? = null,
    val metrics: Map<String, Any>? = null
)
