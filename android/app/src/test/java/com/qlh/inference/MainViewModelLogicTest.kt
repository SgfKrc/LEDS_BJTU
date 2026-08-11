package com.qlh.inference

import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.status.BackendStatus
import com.qlh.inference.status.GpuStatus
import com.qlh.inference.status.MemoryStatus
import com.qlh.inference.status.SystemStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
}
