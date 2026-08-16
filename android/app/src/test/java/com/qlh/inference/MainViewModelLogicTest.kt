package com.qlh.inference

import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.status.BackendStatus
import com.qlh.inference.status.GpuStatus
import com.qlh.inference.status.MemoryStatus
import com.qlh.inference.status.SystemStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * MainViewModel 纯逻辑与 UI 状态默认值测试（JVM，无需设备）。
 */
class MainViewModelLogicTest {

    // ---- networkTypeFromTransports ----

    @Test
    fun `network type wifi wins over others`() {
        assertEquals("wifi", networkTypeFromTransports(true, true, true, true))
        assertEquals("wifi", networkTypeFromTransports(true, false, false, false))
    }

    @Test
    fun `network type falls through cellular ethernet vpn`() {
        assertEquals("mobile", networkTypeFromTransports(false, true, false, false))
        assertEquals("ethernet", networkTypeFromTransports(false, false, true, false))
        assertEquals("vpn", networkTypeFromTransports(false, false, false, true))
    }

    @Test
    fun `network type other when no transport`() {
        assertEquals("other", networkTypeFromTransports(false, false, false, false))
    }

    // ---- buildAndroidPresencePayload ----

    private fun sampleSystem() = SystemStatus(
        manufacturer = "TestMaker",
        brand = "TestBrand",
        model = "TestModel",
        device = "test-device",
        hardware = "test-hw",
        socManufacturer = "SoC",
        socModel = "SoC-1",
        sdkInt = 34,
        androidRelease = "14",
        abis = listOf("arm64-v8a", "x86_64"),
        cpuCores = 8,
        thermalStatus = "normal",
    )

    private fun sampleMemory() = MemoryStatus(
        availableBytes = 1024L * 1024 * 1024,
        totalBytes = 8L * 1024 * 1024 * 1024,
        lowMemory = false,
        lowRamDevice = false,
    )

    private fun sampleGpu() = GpuStatus(
        vendor = "Qualcomm",
        renderer = "Adreno",
        version = "1.0",
        probeError = null,
        supportsGpuOffload = false,
        backendDevices = "",
        note = "probe note",
    )

    @Test
    fun `payload has fixed envelope fields`() {
        val payload = buildAndroidPresencePayload(
            inferenceMode = "thin",
            appVariant = "full",
            appVersion = "0.1.0",
            system = sampleSystem(),
            memory = sampleMemory(),
            gpu = sampleGpu(),
            runtime = null,
        )
        assertEquals("http_thin", payload["connection_type"])
        assertEquals(false, payload["pipeline_worker"])
        assertEquals("thin", payload["client_mode"])
        assertEquals("full", payload["app_variant"])
        assertEquals("0.1.0", payload["app_version"])
    }

    @Test
    fun `payload nests android memory gpu backend sections`() {
        val payload = buildAndroidPresencePayload(
            inferenceMode = "full",
            appVariant = "lite",
            appVersion = "0.2.0",
            system = sampleSystem(),
            memory = sampleMemory(),
            gpu = sampleGpu(),
            runtime = null,
        )
        @Suppress("UNCHECKED_CAST")
        val android = payload["android"] as Map<String, Any?>
        assertEquals("TestMaker", android["manufacturer"])
        assertEquals("TestModel", android["model"])
        assertEquals(listOf("arm64-v8a", "x86_64"), android["abis"])
        assertEquals(8, android["cpu_cores"])
        assertEquals("normal", android["thermal_status"])

        @Suppress("UNCHECKED_CAST")
        val memory = payload["memory"] as Map<String, Any?>
        assertEquals(1024L * 1024 * 1024, memory["available_bytes"])
        assertEquals(false, memory["low_ram_device"])

        @Suppress("UNCHECKED_CAST")
        val gpu = payload["gpu"] as Map<String, Any?>
        assertEquals("Qualcomm", gpu["vendor"])
        assertEquals(false, gpu["supports_gpu_offload"])
        assertEquals("probe note", gpu["note"])

        @Suppress("UNCHECKED_CAST")
        val backend = payload["backend"] as Map<String, Any?>
        assertEquals("", backend["engine"])
        assertEquals(false, backend["supports_gpu_offload"])

        @Suppress("UNCHECKED_CAST")
        val multimodal = payload["multimodal"] as Map<String, Any?>
        assertEquals(false, multimodal["vision_supported"])
        assertEquals(false, multimodal["assets_present"])
        assertEquals(false, multimodal["ready"])
    }

    @Test
    fun `payload reflects runtime gpu and backend status when present`() {
        val runtime = AndroidRuntimeStatus(
            gpu = GpuStatus(supportsGpuOffload = true, backendDevices = "gpu:0"),
            backend = BackendStatus(engine = "llama.cpp", supportsGpuOffload = true),
        )
        val payload = buildAndroidPresencePayload(
            inferenceMode = "full",
            appVariant = "full",
            appVersion = "0.1.0",
            system = sampleSystem(),
            memory = sampleMemory(),
            gpu = sampleGpu(),
            runtime = runtime,
        )
        @Suppress("UNCHECKED_CAST")
        val gpu = payload["gpu"] as Map<String, Any?>
        assertEquals(true, gpu["supports_gpu_offload"])
        assertEquals("gpu:0", gpu["backend_devices"])

        @Suppress("UNCHECKED_CAST")
        val backend = payload["backend"] as Map<String, Any?>
        assertEquals("llama.cpp", backend["engine"])
        assertEquals(true, backend["supports_gpu_offload"])
    }

    @Test
    fun `payload never contains absolute paths or credentials`() {
        val payload = buildAndroidPresencePayload(
            inferenceMode = "full",
            appVariant = "full",
            appVersion = "0.1.0",
            system = sampleSystem(),
            memory = sampleMemory(),
            gpu = sampleGpu(),
            runtime = AndroidRuntimeStatus(),
        )
        val json = payload.toString()
        assertFalse(json.contains("/storage/"))
        assertFalse(json.contains("token"))
        assertFalse(json.contains("password"))
        assertFalse(json.contains("secret"))
    }

    // ---- MainUiState 默认值 ----

    @Test
    fun `MainUiState defaults match settings defaults`() {
        val state = MainUiState()
        assertEquals("chat", state.currentTab)
        assertEquals(0L, state.currentSessionId)
        assertEquals("新对话", state.currentSessionTitle)
        assertTrue(state.messages.isEmpty())
        assertFalse(state.isLoading)
        assertEquals(SettingsDataStore.DEFAULT_HOST, state.serverHost)
        assertEquals(SettingsDataStore.DEFAULT_PORT, state.serverPort)
        assertEquals(SettingsDataStore.DEFAULT_MODE, state.inferenceMode)
        assertEquals(SettingsDataStore.DEFAULT_MAX_TOKENS, state.maxTokens)
        assertEquals(SettingsDataStore.DEFAULT_TEMPERATURE, state.temperature, 0.001f)
        assertEquals(SettingsDataStore.DEFAULT_TOP_P, state.topP, 0.001f)
        assertEquals(SettingsDataStore.DEFAULT_CONTEXT_SIZE, state.contextSize)
        assertEquals("", state.modelTreeUri)
        assertEquals(SettingsDataStore.DEFAULT_MODEL_STORAGE_MODE, state.modelStorageMode)
        assertEquals(SettingsDataStore.DEFAULT_THEME_MODE, state.themeMode)
        assertTrue(state.availableModels.isEmpty())
        assertTrue(state.sessions.isEmpty())
    }

    // ---- 聊天与会话状态机 ----

    @Test
    fun `selecting a session returns to chat and preserves unrelated state`() {
        val state = MainUiState(
            currentTab = "sessions",
            serverHost = "100.64.0.2",
            error = "旧错误",
        )

        val selected = selectUiSession(state, sessionId = 42L, sessionTitle = "部署讨论")

        assertEquals("chat", selected.currentTab)
        assertEquals(42L, selected.currentSessionId)
        assertEquals("部署讨论", selected.currentSessionTitle)
        assertEquals("100.64.0.2", selected.serverHost)
        assertEquals("旧错误", selected.error)
    }

    @Test
    fun `message submission lifecycle records retry payload and clears transient error`() {
        val initial = MainUiState(currentSessionId = 9L, error = "上次失败")
        val sending = startMessageSubmission(initial, "重新发送这条消息")
        val failed = failMessageSubmission(sending, "连接超时")
        val retrying = startMessageRetry(failed)
        val completed = completeMessageSubmission(retrying)

        assertTrue(sending.isLoading)
        assertEquals("重新发送这条消息", sending.lastSentMessage)
        assertEquals(null, sending.error)
        assertFalse(failed.isLoading)
        assertEquals("连接超时", failed.error)
        assertTrue(retrying.isLoading)
        assertEquals(null, retrying.error)
        assertFalse(completed.isLoading)
        assertEquals(null, completed.error)
        assertEquals("重新发送这条消息", completed.lastSentMessage)
    }

    @Test
    fun `retry without a prior message is a no-op and clear only removes error`() {
        val state = MainUiState(currentTab = "settings", error = "发送失败")

        assertSame(state, startMessageRetry(state))
        val cleared = clearMessageError(state)
        assertEquals("settings", cleared.currentTab)
        assertEquals(null, cleared.error)
        assertFalse(cleared.isLoading)
    }

    @Test
    fun `send error formatting keeps actionable network and fallback messages`() {
        assertEquals(
            "无法连接主节点，请检查地址和网络",
            formatMessageSendError(java.net.ConnectException()),
        )
        assertEquals(
            "连接超时，请检查主节点是否运行",
            formatMessageSendError(java.net.SocketTimeoutException()),
        )
        assertEquals(
            "当前模式暂不支持",
            formatMessageSendError(UnsupportedOperationException()),
        )
        assertEquals(
            "发送失败: bad gateway",
            formatMessageSendError(IllegalStateException("bad gateway")),
        )
    }

    // ---- 聊天/会话 UI 状态机（MainViewModelLogic 纯函数） ----

    @Test
    fun `select tab switches tab without touching other fields`() {
        val base = MainUiState(currentTab = "chat", serverHost = "1.2.3.4")
        val next = selectUiTab(base, "settings")
        assertEquals("settings", next.currentTab)
        assertEquals("1.2.3.4", next.serverHost)
    }

    @Test
    fun `select session sets id title and jumps to chat tab`() {
        val base = MainUiState(currentTab = "sessions", currentSessionId = 0)
        val next = selectUiSession(base, sessionId = 7L, sessionTitle = "网络排查")
        assertEquals(7L, next.currentSessionId)
        assertEquals("网络排查", next.currentSessionTitle)
        assertEquals("chat", next.currentTab)
    }

    @Test
    fun `start submission marks loading clears error and records last message`() {
        val base = MainUiState(isLoading = false, error = "旧错误")
        val next = startMessageSubmission(base, "你好")
        assertTrue(next.isLoading)
        assertNull(next.error)
        assertEquals("你好", next.lastSentMessage)
    }

    @Test
    fun `complete submission stops loading and clears error`() {
        val base = MainUiState(isLoading = true, error = "旧错误")
        val next = completeMessageSubmission(base)
        assertFalse(next.isLoading)
        assertNull(next.error)
        assertEquals("旧错误", base.error) // 纯函数不改原状态
    }

    @Test
    fun `fail submission stops loading and records error`() {
        val next = failMessageSubmission(MainUiState(isLoading = true), "后端 500")
        assertFalse(next.isLoading)
        assertEquals("后端 500", next.error)
    }

    @Test
    fun `retry only starts when a last message exists`() {
        // 无 lastSentMessage 时保持原状态（不进入加载）
        val idle = MainUiState(isLoading = false, lastSentMessage = null)
        assertSame(idle, startMessageRetry(idle))
        // 有 lastSentMessage：进入加载并清错误（保留原文）
        val next = startMessageRetry(MainUiState(isLoading = false, error = "旧错", lastSentMessage = "原文"))
        assertTrue(next.isLoading)
        assertNull(next.error)
        assertEquals("原文", next.lastSentMessage)
    }

    @Test
    fun `clear error nulls the error only`() {
        val next = clearMessageError(MainUiState(error = "x", isLoading = true))
        assertNull(next.error)
        assertTrue(next.isLoading)
    }

    @Test
    fun `image submission keeps a bounded normalized payload`() {
        val image = "data:image/png;base64,iVBORw0KGgo="
        val urls = normalizeChatImageDataUrls(listOf("  $image  ", "", image))

        assertEquals(listOf(image, image), urls)
        assertNull(validateChatImageSubmission("thin", urls))
    }

    @Test
    fun `image submission rejects invalid format and local mode`() {
        assertEquals(
            "图像仅支持 PNG/JPEG/WebP base64 data URL",
            validateChatImageSubmission("thin", listOf("https://example/image.png")),
        )
        assertEquals(
            "本地模式暂不支持图像理解，请切换远程模式",
            validateChatImageSubmission("full", listOf("data:image/png;base64,iVBORw0KGgo=")),
        )
    }

    @Test
    fun `image submission rejects more than four non-empty urls`() {
        val images = (1..5).map { "data:image/png;base64,$it" }
        assertEquals(
            "一次最多附加 4 张图片",
            validateChatImageSubmission("thin", images),
        )
    }

    @Test
    fun `retry state retains image payload`() {
        val image = "data:image/png;base64,iVBORw0KGgo="
        val state = startMessageSubmission(MainUiState(), "看图", listOf(image))
        val retry = startMessageRetry(failMessageSubmission(state, "超时"))

        assertEquals(listOf(image), retry.lastSentImageDataUrls)
        assertEquals("看图", retry.lastSentMessage)
    }
}
