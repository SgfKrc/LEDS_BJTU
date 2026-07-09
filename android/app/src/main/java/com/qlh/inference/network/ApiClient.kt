package com.qlh.inference.network

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.qlh.inference.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.logging.HttpLoggingInterceptor
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

// ================================================================
// DTO — 匹配 PC 端 api_server.py 的 ChatRequest / ChatResponse
// ================================================================

data class ChatRequest(
    val message: String,
    @SerializedName("max_new_tokens")
    val maxNewTokens: Int = 512,
    val temperature: Float = 0.7f,
    @SerializedName("top_p")
    val topP: Float = 0.9f,
    @SerializedName("show_thinking")
    val showThinking: Boolean = false,
    @SerializedName("session_id")
    val sessionId: String? = null,
    @SerializedName("streaming_mode")
    val streamingMode: String = "full"    // "full" | "fast"（对应 PC 端 streaming_mode）
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
    val nodeType: String = "android"
)

data class RegisterNodeResponse(
    val status: String = "",
    @SerializedName("node_id")
    val nodeId: String? = null,
    val message: String? = null,
    val state: String? = null
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
                .url("$baseUrl/api/cluster/nodes/register")
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
