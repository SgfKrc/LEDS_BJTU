package com.qlh.inference

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.qlh.inference.BuildConfig
import com.qlh.inference.data.MessageEntity
import com.qlh.inference.data.SessionEntity
import com.qlh.inference.data.SettingsDataStore
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.network.ApiClient
import com.qlh.inference.network.ChatRepository
import com.qlh.inference.service.InferenceService
import com.qlh.inference.service.ModelManager
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.system.AndroidDeviceInfoProvider
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

// ================================================================
// UI 状态
// ================================================================

data class MainUiState(
    // 导航
    val currentTab: String = "chat",

    // 当前会话
    val currentSessionId: Long = 0,
    val currentSessionTitle: String = "新对话",

    // 消息
    val messages: List<MessageEntity> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null,

    // 会话列表
    val sessions: List<SessionEntity> = emptyList(),

    // 设置
    val serverHost: String = SettingsDataStore.DEFAULT_HOST,
    val serverPort: Int = SettingsDataStore.DEFAULT_PORT,
    val inferenceMode: String = SettingsDataStore.DEFAULT_MODE,
    val maxTokens: Int = SettingsDataStore.DEFAULT_MAX_TOKENS,
    val temperature: Float = SettingsDataStore.DEFAULT_TEMPERATURE,
    val topP: Float = SettingsDataStore.DEFAULT_TOP_P,
    val contextSize: Int = SettingsDataStore.DEFAULT_CONTEXT_SIZE,
    val showThinking: Boolean = false,

    // 模型管理
    val modelTreeUri: String = "",
    val selectedModelUri: String = "",
    val modelStorageMode: String = SettingsDataStore.DEFAULT_MODEL_STORAGE_MODE,
    val availableModels: List<ModelManager.ModelDocument> = emptyList(),
    val selectedModelName: String = "",
    val selectedModelSizeBytes: Long = 0L,
    val isScanningModels: Boolean = false,
    val modelMessage: String? = null,

    // 本地运行时状态
    val runtimeStatus: AndroidRuntimeStatus? = null,
    val runtimeStatusLoading: Boolean = false,
    val runtimeStatusError: String? = null,

    // 上次发送的消息（用于重试）
    val lastSentMessage: String? = null
)

// ================================================================
// ViewModel
// ================================================================

@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val database = QlhApplication.instance.database
    private val settings = SettingsDataStore(application)
    private val modelManager = ModelManager(application)

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    // ---- 仓库（根据模式动态创建 ApiClient 或使用本地引擎） ----
    private val repository = ChatRepository(
        sessionDao = database.sessionDao(),
        messageDao = database.messageDao(),
        apiClient = {
            val state = _uiState.value
            if (state.inferenceMode == "thin") {
                ApiClient("http://${state.serverHost}:${state.serverPort}")
            } else {
                null // 全有模式 — 使用本地推理引擎
            }
        },
        inferenceService = {
            // 全有模式：返回 InferenceService 实例（由 Application 管理）
            QlhApplication.instance.inferenceService
        }
    )

    init {
        // 加载设置
        viewModelScope.launch {
            val host = settings.getServerHost()
            val port = settings.getServerPort()
            val mode = if (BuildConfig.IS_LITE) "thin" else settings.getInferenceMode()
            val maxTokens = settings.getMaxTokens()
            val temp = settings.getTemperature()
            val topP = settings.getTopP()
            val contextSize = settings.getContextSize()
            val modelTreeUri = settings.getModelTreeUri()
            val selectedModelUri = settings.getSelectedModelUri()
            val storageMode = settings.getModelStorageMode()
            val selectedModel = modelManager.getSelectedModel()
            val sessions = database.sessionDao().getAllSessions().first()
            val initialSessionId = sessions.firstOrNull()?.id ?: repository.createSession("新对话")
            val initialSession = database.sessionDao().getById(initialSessionId)

            _uiState.value = _uiState.value.copy(
                currentSessionId = initialSessionId,
                currentSessionTitle = initialSession?.title ?: "新对话",
                serverHost = host,
                serverPort = port,
                inferenceMode = mode,
                maxTokens = maxTokens,
                temperature = temp,
                topP = topP,
                contextSize = contextSize,
                modelTreeUri = modelTreeUri,
                selectedModelUri = selectedModelUri,
                modelStorageMode = storageMode,
                selectedModelName = selectedModel?.name.orEmpty(),
                selectedModelSizeBytes = selectedModel?.sizeBytes ?: 0L
            )
            QlhApplication.instance.inferenceService?.modelContextSize = contextSize
            refreshModels(showMessage = false)
            refreshRuntimeStatus()
        }

        // 监听会话列表
        viewModelScope.launch {
            database.sessionDao().getAllSessions().collect { sessions ->
                _uiState.value = _uiState.value.copy(sessions = sessions)

                // 如果没有会话，自动创建默认会话
                if (sessions.isEmpty()) {
                    createSessionInternal("新对话")
                } else if (_uiState.value.currentSessionId == 0L) {
                    // 选择最近更新的会话
                    selectSession(sessions.first().id)
                }
            }
        }

        // 监听当前会话消息 — flatMapLatest 自动取消旧 collector，避免泄漏
        viewModelScope.launch {
            _uiState
                .map { it.currentSessionId }
                .distinctUntilChanged()
                .filter { it > 0L }
                .flatMapLatest { sessionId ->
                    database.messageDao().getMessagesBySession(sessionId)
                }
                .collect { messages ->
                    _uiState.value = _uiState.value.copy(messages = messages)
                }
        }

        // 监听设置变化
        viewModelScope.launch {
            settings.serverHost.collect { host ->
                _uiState.value = _uiState.value.copy(serverHost = host)
            }
        }
        viewModelScope.launch {
            settings.serverPort.collect { port ->
                _uiState.value = _uiState.value.copy(serverPort = port)
            }
        }
        viewModelScope.launch {
            settings.inferenceMode.collect { mode ->
                _uiState.value = _uiState.value.copy(
                    inferenceMode = if (BuildConfig.IS_LITE) "thin" else mode
                )
            }
        }
        viewModelScope.launch {
            settings.modelTreeUri.collect { uri ->
                _uiState.value = _uiState.value.copy(modelTreeUri = uri)
            }
        }
        viewModelScope.launch {
            settings.selectedModelUri.collect { uri ->
                val selected = modelManager.getSelectedModel()
                _uiState.value = _uiState.value.copy(
                    selectedModelUri = uri,
                    selectedModelName = selected?.name.orEmpty(),
                    selectedModelSizeBytes = selected?.sizeBytes ?: 0L
                )
            }
        }
        viewModelScope.launch {
            settings.modelStorageMode.collect { mode ->
                _uiState.value = _uiState.value.copy(modelStorageMode = mode)
            }
        }
        viewModelScope.launch {
            settings.contextSize.collect { size ->
                _uiState.value = _uiState.value.copy(contextSize = size)
            }
        }
    }

    // ==================== 导航 ====================

    fun selectTab(tab: String) {
        _uiState.value = _uiState.value.copy(currentTab = tab)
    }

    // ==================== 会话管理 ====================

    fun createSession() {
        viewModelScope.launch {
            createSessionInternal("新对话")
            _uiState.value = _uiState.value.copy(currentTab = "chat")
        }
    }

    private suspend fun createSessionInternal(title: String) {
        val id = repository.createSession(title)
        selectSessionInternal(id)
    }

    fun selectSession(sessionId: Long) {
        viewModelScope.launch {
            selectSessionInternal(sessionId)
        }
    }

    /** 同步版本 — 供内部 suspend 函数直接调用，避免 race condition */
    private suspend fun selectSessionInternal(sessionId: Long) {
        val session = repository.getSession(sessionId)
        _uiState.value = _uiState.value.copy(
            currentSessionId = sessionId,
            currentSessionTitle = session?.title ?: "新对话",
            currentTab = "chat"
        )
    }

    fun deleteSession(sessionId: Long) {
        viewModelScope.launch {
            repository.deleteSession(sessionId)
            // 如果删除的是当前会话，切换到其他会话
            if (_uiState.value.currentSessionId == sessionId) {
                val sessions = database.sessionDao().getAllSessions().first()
                if (sessions.isNotEmpty()) {
                    selectSessionInternal(sessions.first().id)
                } else {
                    createSessionInternal("新对话")
                }
            }
        }
    }

    /** 确保当前会话 ID 一定存在，避免首次启动竞态导致外键崩溃。 */
    private suspend fun ensureActiveSession(): MainUiState {
        val current = _uiState.value
        if (current.currentSessionId > 0L && repository.getSession(current.currentSessionId) != null) {
            return current
        }
        val sessions = database.sessionDao().getAllSessions().first()
        val sessionId = sessions.firstOrNull()?.id ?: repository.createSession("新对话")
        val session = repository.getSession(sessionId)
        return current.copy(
            currentSessionId = sessionId,
            currentSessionTitle = session?.title ?: "新对话",
            currentTab = "chat"
        ).also { _uiState.value = it }
    }

    // ==================== 消息 ====================

    fun sendMessage(message: String) {
        QlhLogger.i("MainViewModel", "sendMessage start: ${message.length} chars")
        viewModelScope.launch {
            try {
                val state = ensureActiveSession()
                _uiState.value = state.copy(
                    isLoading = true,
                    error = null,
                    lastSentMessage = message
                )

                val result = repository.sendMessage(
                    sessionId = state.currentSessionId,
                    message = message,
                    maxTokens = state.maxTokens,
                    temperature = state.temperature,
                    topP = state.topP,
                    showThinking = state.showThinking
                )

                result.onSuccess {
                    _uiState.value = _uiState.value.copy(isLoading = false, error = null)
                    refreshRuntimeStatus()
                }.onFailure { e ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        error = formatSendError(e)
                    )
                }
            } catch (e: Exception) {
                QlhLogger.e("MainViewModel", "sendMessage crashed", e)
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = formatSendError(e)
                )
            }
        }
    }

    fun retryLastMessage() {
        val lastMsg = _uiState.value.lastSentMessage ?: return
        viewModelScope.launch {
            try {
                val state = ensureActiveSession()
                _uiState.value = state.copy(isLoading = true, error = null)

                // 跳过用户消息保存（上次失败的尝试已保存），只重新调用 API
                val result = repository.sendMessage(
                    sessionId = state.currentSessionId,
                    message = lastMsg,
                    maxTokens = state.maxTokens,
                    temperature = state.temperature,
                    topP = state.topP,
                    showThinking = state.showThinking,
                    skipUserSave = true  // ★ 避免重复用户消息
                )

                result.onSuccess {
                    _uiState.value = _uiState.value.copy(isLoading = false, error = null)
                    refreshRuntimeStatus()
                }.onFailure { e ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        error = formatSendError(e)
                    )
                }
            } catch (e: Exception) {
                QlhLogger.e("MainViewModel", "retryLastMessage crashed", e)
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = formatSendError(e)
                )
            }
        }
    }

    fun clearError() {
        _uiState.value = _uiState.value.copy(error = null)
    }

    fun refreshRuntimeStatus() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                runtimeStatusLoading = true,
                runtimeStatusError = null
            )
            try {
                val state = _uiState.value
                val service = QlhApplication.instance.inferenceService
                val status = service?.getRuntimeStatus(state.inferenceMode, BuildConfig.IS_LITE)
                    ?: createPassiveRuntimeStatus(state.inferenceMode)
                _uiState.value = _uiState.value.copy(
                    runtimeStatus = status,
                    runtimeStatusLoading = false,
                    runtimeStatusError = null
                )
            } catch (e: Exception) {
                QlhLogger.e("MainViewModel", "refreshRuntimeStatus failed", e)
                _uiState.value = _uiState.value.copy(
                    runtimeStatusLoading = false,
                    runtimeStatusError = e.message ?: e.javaClass.simpleName
                )
            }
        }
    }

    private fun createPassiveRuntimeStatus(inferenceMode: String): AndroidRuntimeStatus {
        val provider = AndroidDeviceInfoProvider(getApplication())
        val nativeResult = runCatching { System.loadLibrary("qlh_llama_jni") }
        return AndroidRuntimeStatus(
            nativeRuntimeAvailable = nativeResult.isSuccess,
            nativeRuntimeError = nativeResult.exceptionOrNull()?.message,
            serviceRunning = false,
            inferenceMode = inferenceMode,
            isLite = BuildConfig.IS_LITE,
            system = provider.getSystemStatus(),
            memory = provider.getMemoryStatus(),
            storage = provider.getStorageStatus(),
            gpu = provider.getGpuStatus(),
        )
    }

    private fun formatSendError(e: Throwable): String {
        return when (e) {
            is java.net.ConnectException -> "无法连接主节点，请检查地址和网络"
            is java.net.SocketTimeoutException -> "连接超时，请检查主节点是否运行"
            is UnsupportedOperationException -> e.message ?: "当前模式暂不支持"
            else -> "发送失败: ${e.message ?: e.javaClass.simpleName}"
        }
    }

    // ==================== 设置 ====================

    fun setServerHost(host: String) {
        viewModelScope.launch {
            settings.setServerHost(host)
            _uiState.value = _uiState.value.copy(serverHost = host)
        }
    }

    fun setServerPort(port: Int) {
        viewModelScope.launch {
            settings.setServerPort(port)
            _uiState.value = _uiState.value.copy(serverPort = port)
        }
    }

    fun setInferenceMode(mode: String) {
        if (BuildConfig.IS_LITE && mode != "thin") return
        viewModelScope.launch {
            settings.setInferenceMode(mode)
            _uiState.value = _uiState.value.copy(inferenceMode = mode)
            refreshRuntimeStatus()
        }
    }

    fun setMaxTokens(tokens: Int) {
        viewModelScope.launch {
            settings.setMaxTokens(tokens)
            _uiState.value = _uiState.value.copy(maxTokens = tokens)
        }
    }

    fun setTemperature(temp: Float) {
        viewModelScope.launch {
            settings.setTemperature(temp)
            _uiState.value = _uiState.value.copy(temperature = temp)
        }
    }

    fun setTopP(topP: Float) {
        viewModelScope.launch {
            settings.setTopP(topP)
            _uiState.value = _uiState.value.copy(topP = topP)
        }
    }

    fun setContextSize(size: Int) {
        viewModelScope.launch {
            settings.setContextSize(size)
            _uiState.value = _uiState.value.copy(contextSize = size)
            QlhApplication.instance.inferenceService?.modelContextSize = size
            refreshRuntimeStatus()
        }
    }

    // ==================== 模型管理 ====================

    fun selectModelDirectory(treeUri: Uri) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isScanningModels = true, modelMessage = null)
            unloadRunningModel()
            val result = modelManager.selectModelDirectory(treeUri)
            result.onSuccess { models ->
                val selected = modelManager.getSelectedModel()
                _uiState.value = _uiState.value.copy(
                    availableModels = models,
                    selectedModelUri = selected?.uri?.toString().orEmpty(),
                    selectedModelName = selected?.name.orEmpty(),
                    selectedModelSizeBytes = selected?.sizeBytes ?: 0L,
                    isScanningModels = false,
                    modelMessage = if (models.isEmpty()) {
                        "目录已授权，但未发现 .gguf 模型文件"
                    } else {
                        "已发现 ${models.size} 个 GGUF 模型"
                    }
                )
                refreshRuntimeStatus()
            }.onFailure { e ->
                _uiState.value = _uiState.value.copy(
                    isScanningModels = false,
                    modelMessage = "目录授权失败: ${e.message}"
                )
            }
        }
    }

    fun refreshModels(showMessage: Boolean = true) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isScanningModels = true, modelMessage = null)
            val result = modelManager.listModels()
            result.onSuccess { models ->
                val selected = modelManager.getSelectedModel()
                _uiState.value = _uiState.value.copy(
                    availableModels = models,
                    selectedModelName = selected?.name.orEmpty(),
                    selectedModelSizeBytes = selected?.sizeBytes ?: 0L,
                    selectedModelUri = selected?.uri?.toString().orEmpty(),
                    isScanningModels = false,
                    modelMessage = if (showMessage) {
                        if (models.isEmpty()) "未发现 .gguf 模型文件" else "已扫描到 ${models.size} 个模型"
                    } else {
                        null
                    }
                )
            }.onFailure { e ->
                _uiState.value = _uiState.value.copy(
                    availableModels = emptyList(),
                    isScanningModels = false,
                    modelMessage = "扫描失败: ${e.message}"
                )
            }
        }
    }

    fun selectModel(model: ModelManager.ModelDocument) {
        viewModelScope.launch {
            unloadRunningModel()
            val result = modelManager.selectModel(model.uri)
            result.onSuccess {
                _uiState.value = _uiState.value.copy(
                    selectedModelUri = model.uri.toString(),
                    selectedModelName = model.name,
                    selectedModelSizeBytes = model.sizeBytes,
                    modelMessage = "已选择模型: ${model.name}"
                )
                refreshRuntimeStatus()
            }.onFailure { e ->
                _uiState.value = _uiState.value.copy(modelMessage = "选择模型失败: ${e.message}")
            }
        }
    }

    fun deleteSelectedModel() {
        viewModelScope.launch {
            val name = _uiState.value.selectedModelName
            unloadRunningModel()
            val result = modelManager.deleteSelectedModel()
            result.onSuccess {
                _uiState.value = _uiState.value.copy(
                    selectedModelUri = "",
                    selectedModelName = "",
                    selectedModelSizeBytes = 0L,
                    modelMessage = if (name.isBlank()) "没有已选择的模型" else "已删除模型: $name"
                )
                refreshModels(showMessage = false)
                refreshRuntimeStatus()
            }.onFailure { e ->
                _uiState.value = _uiState.value.copy(modelMessage = "删除模型失败: ${e.message}")
            }
        }
    }

    private suspend fun unloadRunningModel() {
        QlhApplication.instance.inferenceService?.unloadModel()
    }
}
