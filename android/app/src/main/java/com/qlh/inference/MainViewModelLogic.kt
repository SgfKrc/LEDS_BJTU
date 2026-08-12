package com.qlh.inference

import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.status.GpuStatus
import com.qlh.inference.status.MemoryStatus
import com.qlh.inference.status.SystemStatus

/**
 * MainViewModel 的纯逻辑（不访问 Android API，JVM 可单测）。
 */

/** 由 ConnectivityManager transport 标志判定网络类型（顺序敏感：wifi 优先）。 */
fun networkTypeFromTransports(
    hasWifi: Boolean,
    hasCellular: Boolean,
    hasEthernet: Boolean,
    hasVpn: Boolean,
): String = when {
    hasWifi -> "wifi"
    hasCellular -> "mobile"
    hasEthernet -> "ethernet"
    hasVpn -> "vpn"
    else -> "other"
}

// ---- 聊天与会话 UI 状态机（不访问 Room、DataStore 或网络，JVM 可单测） ----

fun selectUiTab(state: MainUiState, tab: String): MainUiState =
    state.copy(currentTab = tab)

fun selectUiSession(
    state: MainUiState,
    sessionId: Long,
    sessionTitle: String,
): MainUiState = state.copy(
    currentSessionId = sessionId,
    currentSessionTitle = sessionTitle,
    currentTab = "chat",
)

fun startMessageSubmission(state: MainUiState, message: String): MainUiState =
    state.copy(isLoading = true, error = null, lastSentMessage = message)

fun completeMessageSubmission(state: MainUiState): MainUiState =
    state.copy(isLoading = false, error = null)

fun failMessageSubmission(state: MainUiState, error: String): MainUiState =
    state.copy(isLoading = false, error = error)

fun startMessageRetry(state: MainUiState): MainUiState =
    if (state.lastSentMessage == null) state else state.copy(isLoading = true, error = null)

fun clearMessageError(state: MainUiState): MainUiState =
    state.copy(error = null)

fun formatMessageSendError(error: Throwable): String = when (error) {
    is java.net.ConnectException -> "无法连接主节点，请检查地址和网络"
    is java.net.SocketTimeoutException -> "连接超时，请检查主节点是否运行"
    is UnsupportedOperationException -> error.message ?: "当前模式暂不支持"
    else -> "发送失败: ${error.message ?: error.javaClass.simpleName}"
}

/** Android 节点 presence 设备信息 payload（纯组装；不包含模型绝对路径/密钥）。 */
fun buildAndroidPresencePayload(
    inferenceMode: String,
    appVariant: String,
    appVersion: String,
    system: SystemStatus,
    memory: MemoryStatus,
    gpu: GpuStatus,
    runtime: AndroidRuntimeStatus?,
): Map<String, Any?> = mapOf(
    "connection_type" to "http_thin",
    "pipeline_worker" to false,
    "client_mode" to inferenceMode,
    "app_variant" to appVariant,
    "app_version" to appVersion,
    "android" to mapOf(
        "manufacturer" to system.manufacturer,
        "brand" to system.brand,
        "model" to system.model,
        "device" to system.device,
        "hardware" to system.hardware,
        "soc_manufacturer" to system.socManufacturer,
        "soc_model" to system.socModel,
        "sdk_int" to system.sdkInt,
        "android_release" to system.androidRelease,
        "abis" to system.abis,
        "cpu_cores" to system.cpuCores,
        "thermal_status" to system.thermalStatus,
    ),
    "memory" to mapOf(
        "available_bytes" to memory.availableBytes,
        "total_bytes" to memory.totalBytes,
        "low_memory" to memory.lowMemory,
        "low_ram_device" to memory.lowRamDevice,
    ),
    "gpu" to mapOf(
        "vendor" to gpu.vendor,
        "renderer" to gpu.renderer,
        "version" to gpu.version,
        "probe_error" to gpu.probeError,
        "supports_gpu_offload" to (runtime?.gpu?.supportsGpuOffload ?: false),
        "backend_devices" to (runtime?.gpu?.backendDevices ?: ""),
        "note" to gpu.note,
    ),
    "backend" to mapOf(
        "engine" to (runtime?.backend?.engine ?: ""),
        "supports_gpu_offload" to (runtime?.backend?.supportsGpuOffload ?: false),
    ),
)
