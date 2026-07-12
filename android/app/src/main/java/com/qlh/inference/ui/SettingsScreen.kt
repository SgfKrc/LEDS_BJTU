package com.qlh.inference.ui

import android.app.ActivityManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Build
import com.qlh.inference.BuildConfig
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.clickable
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.NetworkCheck
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.qlh.inference.logging.QlhLogger
import com.qlh.inference.network.ApiClient
import com.qlh.inference.service.ModelManager
import com.qlh.inference.status.AndroidRuntimeStatus
import kotlinx.coroutines.launch

// ================================================================
// 设置界面
// ================================================================

@Composable
fun SettingsScreen(
    serverHost: String,
    serverPort: Int,
    inferenceMode: String,
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    contextSize: Int,
    showThinking: Boolean,
    modelTreeUri: String,
    selectedModelUri: String,
    modelStorageMode: String,
    availableModels: List<ModelManager.ModelDocument>,
    selectedModelName: String,
    selectedModelSizeBytes: Long,
    isScanningModels: Boolean,
    modelMessage: String?,
    onServerHostChange: (String) -> Unit,
    onServerPortChange: (Int) -> Unit,
    onInferenceModeChange: (String) -> Unit,
    onMaxTokensChange: (Int) -> Unit,
    onTemperatureChange: (Float) -> Unit,
    onTopPChange: (Float) -> Unit,
    onContextSizeChange: (Int) -> Unit,
    onShowThinkingChange: (Boolean) -> Unit,
    onChooseModelDirectory: () -> Unit,
    onRefreshModels: () -> Unit,
    onModelSelected: (ModelManager.ModelDocument) -> Unit,
    onDeleteSelectedModel: () -> Unit,
    runtimeStatus: AndroidRuntimeStatus?,
    runtimeStatusLoading: Boolean,
    runtimeStatusError: String?,
    onRefreshRuntimeStatus: () -> Unit,
    onConnectionTestSuccess: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val scope = rememberCoroutineScope()
    val isLite = BuildConfig.IS_LITE
    var isTesting by remember { mutableStateOf(false) }
    var connectionResult by remember { mutableStateOf<Boolean?>(null) }
    var showModeDialog by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        onRefreshRuntimeStatus()
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // ---- 主节点连接 ----
        SectionHeader(title = "主节点连接", icon = Icons.Default.NetworkCheck)

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                OutlinedTextField(
                    value = serverHost,
                    onValueChange = onServerHostChange,
                    label = { Text("主节点地址") },
                    placeholder = { Text("例如: 192.168.1.100") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    enabled = !isTesting
                )
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedTextField(
                    value = serverPort.toString(),
                    onValueChange = { it.toIntOrNull()?.let(onServerPortChange) },
                    label = { Text("端口") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    enabled = !isTesting
                )
                Spacer(modifier = Modifier.height(12.dp))

                // 测试连接按钮
                Button(
                    onClick = {
                        isTesting = true
                        connectionResult = null
                        scope.launch {
                            val url = "http://$serverHost:$serverPort"
                            QlhLogger.i("Settings", "测试连接: $url")
                            val client = ApiClient(url)
                            val raw = client.testConnection()
                            connectionResult = raw.getOrDefault(false)
                            isTesting = false
                            raw.onSuccess {
                                QlhLogger.i("Settings", "连接测试成功: $url")
                                onConnectionTestSuccess()
                            }.onFailure { e ->
                                QlhLogger.w("Settings", "连接测试失败: $url — ${e.message ?: e.javaClass.simpleName}")
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isTesting && serverHost.isNotBlank()
                ) {
                    if (isTesting) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("测试中…")
                    } else {
                        Text("测试连接")
                    }
                }

                // 连接状态
                connectionResult?.let { success ->
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = if (success) Icons.Default.CheckCircle else Icons.Default.Error,
                            contentDescription = null,
                            tint = if (success) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.error
                            },
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (success) "✓ 已连接" else "✗ 无法连接",
                            style = MaterialTheme.typography.bodyMedium,
                            color = if (success) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.error
                            }
                        )
                    }
                }
            }
        }

        // ---- 推理模式 ----
        SectionHeader(title = "推理模式", icon = Icons.Default.Info)

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = if (inferenceMode == "thin") "全无 (远程推理)" else "全有 (本地推理)",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = if (inferenceMode == "thin") {
                                "请求发送给 PC 主节点，本机不计算"
                            } else {
                                "本机加载 GGUF 模型，离线完整推理"
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Button(
                        onClick = { showModeDialog = true },
                        enabled = !isLite,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.secondaryContainer,
                            contentColor = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                    ) {
                        Text(if (isLite) "极简版" else "切换")
                    }
                }

                if (!isLite && inferenceMode == "full") {
                    Spacer(modifier = Modifier.height(12.dp))
                    HorizontalDivider()
                    Spacer(modifier = Modifier.height(12.dp))
                    ModelManagementPanel(
                        modelTreeUri = modelTreeUri,
                        selectedModelUri = selectedModelUri,
                        modelStorageMode = modelStorageMode,
                        availableModels = availableModels,
                        selectedModelName = selectedModelName,
                        selectedModelSizeBytes = selectedModelSizeBytes,
                        isScanningModels = isScanningModels,
                        modelMessage = modelMessage,
                        onChooseModelDirectory = onChooseModelDirectory,
                        onRefreshModels = onRefreshModels,
                        onModelSelected = onModelSelected,
                        onDeleteSelectedModel = onDeleteSelectedModel
                    )
                }
            }
        }

        // ---- 推理参数 ----
        SectionHeader(title = "推理参数")

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                // Max Tokens
                var maxTokensText by remember(maxTokens) {
                    mutableStateOf(maxTokens.toString())
                }
                OutlinedTextField(
                    value = maxTokensText,
                    onValueChange = {
                        maxTokensText = it
                        it.toIntOrNull()?.let { v -> if (v in 1..4096) onMaxTokensChange(v) }
                    },
                    label = { Text("最大 Token 数") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Temperature
                Text(
                    text = "温度: ${"%.1f".format(temperature)}",
                    style = MaterialTheme.typography.bodyMedium
                )
                Slider(
                    value = temperature,
                    onValueChange = onTemperatureChange,
                    valueRange = 0f..2f,
                    steps = 19,
                    colors = SliderDefaults.colors(
                        thumbColor = MaterialTheme.colorScheme.primary,
                        activeTrackColor = MaterialTheme.colorScheme.primary
                    )
                )

                Spacer(modifier = Modifier.height(12.dp))

                // Top-P
                Text(
                    text = "Top-P: ${"%.2f".format(topP)}",
                    style = MaterialTheme.typography.bodyMedium
                )
                Slider(
                    value = topP,
                    onValueChange = onTopPChange,
                    valueRange = 0f..1f,
                    steps = 9,
                    colors = SliderDefaults.colors(
                        thumbColor = MaterialTheme.colorScheme.primary,
                        activeTrackColor = MaterialTheme.colorScheme.primary
                    )
                )

                Spacer(modifier = Modifier.height(16.dp))

                // 上下文长度
                Text(
                    text = "上下文长度: $contextSize",
                    style = MaterialTheme.typography.bodyMedium
                )
                Slider(
                    value = contextSize.toFloat(),
                    onValueChange = { onContextSizeChange(it.toInt()) },
                    valueRange = 512f..4096f,
                    steps = 6,  // 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096
                    colors = SliderDefaults.colors(
                        thumbColor = MaterialTheme.colorScheme.primary,
                        activeTrackColor = MaterialTheme.colorScheme.primary
                    )
                )
                Text(
                    text = "更大的上下文可以处理更长的对话历史，但会增加内存占用和首次加载时间",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        // ---- 设备状态 ----
        SectionHeader(title = "设备状态", icon = Icons.Default.Memory)

        // 刷新按钮
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End
        ) {
            OutlinedButton(
                onClick = onRefreshRuntimeStatus,
                enabled = !runtimeStatusLoading
            ) {
                if (runtimeStatusLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("刷新中…")
                } else {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("刷新状态")
                }
            }
        }

        runtimeStatusError?.let { err ->
            Text(
                text = "状态获取失败: $err",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error
            )
        }

        val status = runtimeStatus
        if (status != null) {
            RuntimeStatusCard(status = status)
        } else {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Text(
                    text = "状态尚未加载，请点击上方「刷新状态」",
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        // ---- 日志管理 ----
        SectionHeader(title = "日志管理", icon = Icons.Default.Description)

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            val ctx = LocalContext.current
            var logRefreshKey by remember { mutableStateOf(0) }
            val logFiles = remember(logRefreshKey) { QlhLogger.getLogFiles() }
            var showLogViewer by remember { mutableStateOf(false) }
            var showClearConfirm by remember { mutableStateOf(false) }
            // L4: 日志查看器搜索状态
            var logSearchQuery by remember { mutableStateOf("") }

            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "日志文件: ${logFiles.size} 个",
                    style = MaterialTheme.typography.bodyMedium
                )
                if (logFiles.isNotEmpty()) {
                    val totalSize = logFiles.sumOf { it.size }
                    Text(
                        text = "总大小: ${formatBytes(totalSize)} · 最新: ${logFiles.first().name} (${formatBytes(logFiles.first().size)})",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Spacer(modifier = Modifier.height(12.dp))

                // L4: 极简版仅保留复制和分享，完整版提供查看+清理
                if (isLite) {
                    // ---- 极简版日志操作 ----
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedButton(
                            onClick = {
                                val allLogText = buildString {
                                    QlhLogger.getLogFiles().forEach { fi ->
                                        appendLine("===== ${fi.name} (${formatBytes(fi.size)}) =====")
                                        appendLine(QlhLogger.readLogFile(fi.name) ?: "(读取失败)")
                                        appendLine()
                                    }
                                }
                                val clipboard = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                clipboard.setPrimaryClip(ClipData.newPlainText("QLH Logs", allLogText))
                            },
                            modifier = Modifier.weight(1f),
                            enabled = logFiles.isNotEmpty()
                        ) {
                            Text("复制日志")
                        }
                        OutlinedButton(
                            onClick = {
                                val file = QlhLogger.getLogFiles().firstOrNull() ?: return@OutlinedButton
                                val content = QlhLogger.readLogFile(file.name) ?: return@OutlinedButton
                                val sendIntent = Intent().apply {
                                    action = Intent.ACTION_SEND
                                    putExtra(Intent.EXTRA_TEXT, content)
                                    putExtra(Intent.EXTRA_SUBJECT, "QLH 日志: ${file.name} (${formatBytes(file.size)})")
                                    type = "text/plain"
                                }
                                ctx.startActivity(Intent.createChooser(sendIntent, "分享日志 — ${file.name} (${formatBytes(file.size)})"))
                            },
                            modifier = Modifier.weight(1f),
                            enabled = logFiles.isNotEmpty()
                        ) {
                            Text("分享日志")
                        }
                    }
                } else {
                    // ---- 完整版日志操作 ----
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Button(
                            onClick = { showLogViewer = true },
                            modifier = Modifier.weight(1f),
                            enabled = logFiles.isNotEmpty()
                        ) {
                            Text("查看日志")
                        }
                        OutlinedButton(
                            onClick = {
                                val allLogText = buildString {
                                    QlhLogger.getLogFiles().forEach { fi ->
                                        appendLine("===== ${fi.name} (${formatBytes(fi.size)}) =====")
                                        val result = QlhLogger.readLogFileWithInfo(fi.name)
                                        if (result.content != null) {
                                            if (result.truncated) {
                                                appendLine("[⚠ 文件过大 (${formatBytes(result.fileSize)})，仅显示末尾 ${formatBytes(QlhLogger.READ_MAX_BYTES)}]")
                                            }
                                            appendLine(result.content)
                                        } else {
                                            appendLine("(读取失败)")
                                        }
                                        appendLine()
                                    }
                                }
                                val clipboard = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                clipboard.setPrimaryClip(ClipData.newPlainText("QLH Logs", allLogText))
                            },
                            modifier = Modifier.weight(1f),
                            enabled = logFiles.isNotEmpty()
                        ) {
                            Text("复制日志")
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedButton(
                            onClick = {
                                val file = QlhLogger.getLogFiles().firstOrNull() ?: return@OutlinedButton
                                val result = QlhLogger.readLogFileWithInfo(file.name)
                                val content = result.content ?: return@OutlinedButton
                                val prefix = if (result.truncated) {
                                    "[⚠ 文件截断: ${formatBytes(result.fileSize)} 仅显示末尾 ${formatBytes(QlhLogger.READ_MAX_BYTES)}]\n\n"
                                } else ""
                                val sendIntent = Intent().apply {
                                    action = Intent.ACTION_SEND
                                    putExtra(Intent.EXTRA_TEXT, prefix + content)
                                    putExtra(Intent.EXTRA_SUBJECT, "QLH 日志: ${file.name} (${formatBytes(file.size)})")
                                    type = "text/plain"
                                }
                                ctx.startActivity(Intent.createChooser(sendIntent, "分享日志 — ${file.name} (${formatBytes(file.size)})"))
                            },
                            modifier = Modifier.weight(1f),
                            enabled = logFiles.isNotEmpty()
                        ) {
                            Text("分享日志")
                        }
                        OutlinedButton(
                            onClick = { showClearConfirm = true },
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.error
                            ),
                            enabled = logFiles.isNotEmpty()
                        ) {
                            Text("清理日志", color = MaterialTheme.colorScheme.error)
                        }
                    }
                }
            }

            // ---- L4 增强日志查看对话框（仅完整版） ----
            if (showLogViewer && !isLite) {
                // 构建每个文件的内容（含截断信息）
                val fileContents = remember(showLogViewer, logRefreshKey) {
                    QlhLogger.getLogFiles().map { fi ->
                        val result = QlhLogger.readLogFileWithInfo(fi.name)
                        Triple(fi, result.content, result.truncated)
                    }
                }

                // 根据搜索词过滤
                val filteredContents = remember(fileContents, logSearchQuery) {
                    if (logSearchQuery.isBlank()) {
                        fileContents
                    } else {
                        fileContents.mapNotNull { (fi, content, truncated) ->
                            if (content != null && content.contains(logSearchQuery, ignoreCase = true)) {
                                Triple(fi, content, truncated)
                            } else if (fi.name.contains(logSearchQuery, ignoreCase = true)) {
                                // 文件名匹配也保留
                                Triple(fi, content, truncated)
                            } else {
                                null
                            }
                        }
                    }
                }

                val displayText = remember(filteredContents, logSearchQuery) {
                    if (filteredContents.isEmpty()) {
                        if (logSearchQuery.isNotBlank()) {
                            "没有匹配「${logSearchQuery}」的日志内容。\n"
                        } else {
                            "(无日志文件)\n"
                        }
                    } else {
                        buildString {
                            filteredContents.forEach { (fi, content, truncated) ->
                                appendLine("=" .repeat(60))
                                appendLine("  📄 ${fi.name}")
                                appendLine("  大小: ${formatBytes(fi.size)}")
                                if (truncated) {
                                    appendLine("  ⚠️ 日志文件过大 (${formatBytes(fi.size)})，仅显示末尾 ${formatBytes(QlhLogger.READ_MAX_BYTES)} 内容")
                                }
                                appendLine("=" .repeat(60))
                                appendLine(content ?: "(读取失败)")
                                appendLine()
                            }
                        }
                    }
                }

                AlertDialog(
                    onDismissRequest = {
                        showLogViewer = false
                        logSearchQuery = ""
                    },
                    title = {
                        Column {
                            Text("日志文件 (${logFiles.size} 个)")
                            // L4: 关键词搜索
                            Spacer(modifier = Modifier.height(8.dp))
                            OutlinedTextField(
                                value = logSearchQuery,
                                onValueChange = { logSearchQuery = it },
                                label = { Text("🔍 搜索日志内容...") },
                                modifier = Modifier.fillMaxWidth(),
                                singleLine = true,
                                textStyle = MaterialTheme.typography.bodySmall
                            )
                            if (logSearchQuery.isNotBlank()) {
                                Text(
                                    text = "匹配 ${filteredContents.size}/${fileContents.size} 个文件",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    },
                    text = {
                        SelectionContainer {
                            Text(
                                text = displayText,
                                fontFamily = FontFamily.Monospace,
                                fontSize = MaterialTheme.typography.bodySmall.fontSize,
                                modifier = Modifier
                                    .horizontalScroll(rememberScrollState())
                                    .heightIn(max = 380.dp)
                            )
                        }
                    },
                    confirmButton = {
                        TextButton(onClick = {
                            showLogViewer = false
                            logSearchQuery = ""
                        }) {
                            Text("关闭")
                        }
                    }
                )
            }

            // ---- 极简版日志查看对话框（保持简洁，无搜索） ----
            if (showLogViewer && isLite) {
                val logContent = remember(showLogViewer, logRefreshKey) {
                    buildString {
                        QlhLogger.getLogFiles().forEach { fi ->
                            val result = QlhLogger.readLogFileWithInfo(fi.name)
                            appendLine("===== ${fi.name} (${formatBytes(fi.size)}) =====")
                            if (result.truncated) {
                                appendLine("[⚠ 文件过大，仅显示末尾 ${formatBytes(QlhLogger.READ_MAX_BYTES)}]")
                            }
                            appendLine(result.content ?: "(读取失败)")
                            appendLine()
                        }
                    }
                }

                AlertDialog(
                    onDismissRequest = { showLogViewer = false },
                    title = { Text("日志文件") },
                    text = {
                        SelectionContainer {
                            Text(
                                text = logContent,
                                fontFamily = FontFamily.Monospace,
                                fontSize = MaterialTheme.typography.bodySmall.fontSize,
                                modifier = Modifier
                                    .horizontalScroll(rememberScrollState())
                                    .heightIn(max = 400.dp)
                            )
                        }
                    },
                    confirmButton = {
                        TextButton(onClick = { showLogViewer = false }) {
                            Text("关闭")
                        }
                    }
                )
            }

            // ---- 清理确认对话框 ----
            if (showClearConfirm) {
                AlertDialog(
                    onDismissRequest = { showClearConfirm = false },
                    title = { Text("清理日志") },
                    text = { Text("确定删除所有日志文件？此操作不可撤销。\n删除后当前日志会自动重新生成。") },
                    confirmButton = {
                        TextButton(
                            onClick = {
                                QlhLogger.clearLogs()
                                logRefreshKey++
                                showClearConfirm = false
                            }
                        ) {
                            Text("删除", color = MaterialTheme.colorScheme.error)
                        }
                    },
                    dismissButton = {
                        TextButton(onClick = { showClearConfirm = false }) {
                            Text("取消")
                        }
                    }
                )
            }
        }

        // ---- 关于 ----
        SectionHeader(title = "关于")

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "轻量化大模型分布式边缘推理系统",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "版本 0.1.0 (Android)",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "北京交通大学 · 大学生创新创业训练计划",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.outline
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "© 2026 北京交通大学 · 项目团队",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.outline
            )
            }
        }

        Spacer(modifier = Modifier.height(32.dp))
    }

    // ---- 模式切换对话框 ----
    if (showModeDialog) {
        AlertDialog(
            onDismissRequest = { showModeDialog = false },
            title = { Text("选择推理模式") },
            text = {
                Column {
                    TextButton(
                        onClick = {
                            onInferenceModeChange("thin")
                            showModeDialog = false
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.fillMaxWidth()) {
                            Text(
                                "📡 全无模式 (远程推理)",
                                style = MaterialTheme.typography.titleMedium,
                                color = if (inferenceMode == "thin") {
                                    MaterialTheme.colorScheme.primary
                                } else {
                                    MaterialTheme.colorScheme.onSurface
                                }
                            )
                            Text(
                                "请求发送给 PC 主节点，本机不计算",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                    if (!isLite) {
                        HorizontalDivider()
                        TextButton(
                            onClick = {
                                onInferenceModeChange("full")
                                showModeDialog = false
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(modifier = Modifier.fillMaxWidth()) {
                                Text(
                                    "🧠 全有模式 (本地推理)",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = if (inferenceMode == "full") {
                                        MaterialTheme.colorScheme.primary
                                    } else {
                                        MaterialTheme.colorScheme.onSurface
                                    }
                                )
                                Text(
                                    "本机加载 GGUF 模型，离线完整推理",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showModeDialog = false }) {
                    Text("取消")
                }
            }
        )
    }
}

// ================================================================
// 模型管理
// ================================================================

@Composable
private fun ModelManagementPanel(
    modelTreeUri: String,
    selectedModelUri: String,
    modelStorageMode: String,
    availableModels: List<ModelManager.ModelDocument>,
    selectedModelName: String,
    selectedModelSizeBytes: Long,
    isScanningModels: Boolean,
    modelMessage: String?,
    onChooseModelDirectory: () -> Unit,
    onRefreshModels: () -> Unit,
    onModelSelected: (ModelManager.ModelDocument) -> Unit,
    onDeleteSelectedModel: () -> Unit
) {
    var showDeleteDialog by remember { mutableStateOf(false) }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(
            text = "模型管理",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold
        )

        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = "存储位置：${if (modelTreeUri.isBlank()) "未选择" else "SAF 外部目录"}",
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = "当前模型：${selectedModelName.ifBlank { "未选择" }}",
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            if (selectedModelSizeBytes > 0L) {
                Text(
                    text = "模型大小：${formatBytes(selectedModelSizeBytes)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Text(
                text = "加载策略：${storageModeLabel(modelStorageMode)}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = onChooseModelDirectory,
                modifier = Modifier.weight(1f),
                enabled = !isScanningModels
            ) {
                Icon(Icons.Default.FolderOpen, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("选择目录")
            }
            OutlinedButton(
                onClick = onRefreshModels,
                modifier = Modifier.weight(1f),
                enabled = !isScanningModels
            ) {
                if (isScanningModels) {
                    CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                } else {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                }
                Spacer(modifier = Modifier.width(6.dp))
                Text("扫描")
            }
        }

        if (availableModels.isNotEmpty()) {
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                availableModels.forEach { model ->
                    ModelRow(
                        model = model,
                        selected = selectedModelUri == model.uri.toString(),
                        onClick = { onModelSelected(model) }
                    )
                }
            }
        }

        modelMessage?.let { message ->
            Text(
                text = message,
                style = MaterialTheme.typography.bodySmall,
                color = if (message.contains("失败")) {
                    MaterialTheme.colorScheme.error
                } else {
                    MaterialTheme.colorScheme.primary
                }
            )
        }

        Text(
            text = "建议选择直接包含 .gguf 的目录；也支持向下扫描 2 层子目录。模型保存在用户授权的外部目录中，卸载 APK 默认保留。",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        OutlinedButton(
            onClick = { showDeleteDialog = true },
            enabled = selectedModelUri.isNotBlank(),
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = MaterialTheme.colorScheme.error
            ),
            modifier = Modifier.fillMaxWidth()
        ) {
            Icon(Icons.Default.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(modifier = Modifier.width(6.dp))
            Text(if (modelStorageMode == ModelManager.STORAGE_MODE_INTERNAL_TEST) "删除内部测试模型" else "删除外部模型文件")
        }
    }

    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text(if (modelStorageMode == ModelManager.STORAGE_MODE_INTERNAL_TEST) "删除内部测试模型" else "删除外部模型文件") },
            text = {
                Text(
                    text = if (modelStorageMode == ModelManager.STORAGE_MODE_INTERNAL_TEST) {
                        "将删除应用内部测试目录中的 GGUF 文件：${selectedModelName.ifBlank { "未选择" }}\n\n此操作不可撤销。"
                    } else {
                        "将删除你授权的外部模型目录中的原始 GGUF 文件：${selectedModelName.ifBlank { "未选择" }}\n\n这不是只删除缓存，而是删除外部目录中的模型原件。删除后无法从应用内恢复，重新使用需要重新下载或导入。是否继续？"
                    }
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showDeleteDialog = false
                        onDeleteSelectedModel()
                    }
                ) {
                    Text("删除", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) {
                    Text("取消")
                }
            }
        )
    }
}

@Composable
private fun ModelRow(
    model: ModelManager.ModelDocument,
    selected: Boolean,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        RadioButton(selected = selected, onClick = onClick)
        Spacer(modifier = Modifier.width(6.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = model.name,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = "${formatBytes(model.sizeBytes)} · ${if (model.source == ModelManager.ModelSource.SAF) "SAF" else "内部测试"}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

private fun storageModeLabel(mode: String): String = when (mode) {
    ModelManager.STORAGE_MODE_SAF_FD -> "SAF fd"
    ModelManager.STORAGE_MODE_SAF_CACHE -> "SAF 缓存副本"
    ModelManager.STORAGE_MODE_INTERNAL_TEST -> "内部测试目录"
    else -> mode.ifBlank { "未设置" }
}

private fun formatBytes(bytes: Long): String {
    if (bytes < 0L) return "未知"
    val units = arrayOf("B", "KB", "MB", "GB", "TB")
    var value = bytes.toDouble()
    var unitIndex = 0
    while (value >= 1024.0 && unitIndex < units.lastIndex) {
        value /= 1024.0
        unitIndex++
    }
    return if (unitIndex == 0) {
        "${bytes} ${units[unitIndex]}"
    } else {
        "%.2f %s".format(value, units[unitIndex])
    }
}

// ================================================================
// 运行时状态卡片
// ================================================================

@Composable
private fun RuntimeStatusCard(status: AndroidRuntimeStatus) {
    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // 原生运行时
        StatusRow(
            label = "llama.cpp 本地运行时",
            value = if (status.nativeRuntimeAvailable) "可用" else "不可用",
            valueColor = if (status.nativeRuntimeAvailable) {
                MaterialTheme.colorScheme.primary
            } else {
                MaterialTheme.colorScheme.error
            }
        )
        status.nativeRuntimeError?.let { err ->
            Text(
                text = "错误: $err",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error
            )
        }

        if (status.isLite) {
            StatusRow("模式", "极简版 (仅远程推理)")
        } else {
            StatusRow("模式", if (status.inferenceMode == "thin") "全无 (远程推理)" else "全有 (本地推理)")
        }
        if (status.serviceRunning) {
            StatusRow("推理服务", "运行中")
        }

        HorizontalDivider()

        // ---- 系统信息 ----
        val sys = status.system
        StatusRow("设备", "${sys.manufacturer} ${sys.brand} ${sys.model}".trim())
        StatusRow("ABI", sys.abis.joinToString(", ").ifBlank { "未知" })
        StatusRow("CPU 核心", "${sys.cpuCores}")
        StatusRow("Android", "${sys.androidRelease} (SDK ${sys.sdkInt})")
        if (sys.socModel.isNotBlank()) {
            StatusRow("SoC", "${sys.socManufacturer} ${sys.socModel}".trim())
        }
        StatusRow("省电模式", if (sys.powerSaveMode) "开启" else "关闭")
        StatusRow("热状态", sys.thermalStatus)

        // ---- 内存 ----
        val mem = status.memory
        StatusRow("系统内存", "${formatBytes(mem.availableBytes)} / ${formatBytes(mem.totalBytes)}")
        if (mem.lowMemory) {
            Text("系统内存不足", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
        }
        StatusRow("JVM Heap", "${formatBytes(mem.heapFreeBytes)} / ${formatBytes(mem.heapTotalBytes)} (max ${formatBytes(mem.heapMaxBytes)})")
        if (mem.lowRamDevice) {
            StatusRow("低 RAM 设备", "是")
        }

        // ---- 存储 ----
        val st = status.storage
        StatusRow("存储 (文件)", "${formatBytes(st.filesAvailableBytes)} / ${formatBytes(st.filesTotalBytes)}")
        StatusRow("存储 (缓存)", "${formatBytes(st.cacheAvailableBytes)} / ${formatBytes(st.cacheTotalBytes)}")

        // ---- GPU ----
        SectionHeader(title = "GPU", icon = Icons.Default.Memory)
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                val gpu = status.gpu
                if (gpu.probeError != null) {
                    StatusRow("GPU 探测", "失败: ${gpu.probeError}")
                } else {
                    StatusRow("Renderer", gpu.renderer.ifBlank { "未知" })
                    StatusRow("Vendor", gpu.vendor.ifBlank { "未知" })
                    StatusRow("GL Version", gpu.version.ifBlank { "未知" })
                }
                StatusRow("GPU Offload 支持", if (gpu.supportsGpuOffload) "是" else "否")
                StatusRow("当前推理后端", if (gpu.supportsGpuOffload) "GPU offload 可用" else "CPU llama.cpp")
                StatusRow("Android GPU 版", "计划单独构建")
                if (gpu.backendDevices.isNotBlank()) {
                    StatusRow("GGML 后端设备", gpu.backendDevices)
                }
                Text(
                    text = gpu.note.ifBlank { "GPU 仅用于设备画像展示；当前 Android Full/Lite 版本不启用 GPU 推理，GPU 版将作为单独版本规划。" },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.outline
                )
            }
        }

        // ---- 后端能力 ----
        val be = status.backend
        if (be.systemInfo.isNotBlank()) {
            SectionHeader(title = "llama.cpp 后端", icon = Icons.Default.Info)
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    StatusRow("引擎", be.engine)
                    StatusRow("mmap", if (be.supportsMmap) "支持" else "不支持")
                    StatusRow("mlock", if (be.supportsMlock) "支持" else "不支持")
                    StatusRow("GPU Offload", if (be.supportsGpuOffload) "支持" else "不支持")
                    StatusRow("RPC", if (be.supportsRpc) "支持" else "不支持")
                    if (be.systemInfo.isNotBlank()) {
                        Text(
                            text = be.systemInfo,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.outline,
                            maxLines = 6,
                            overflow = TextOverflow.Ellipsis,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }
        }

        // ---- 模型 ----
        val mdl = status.model
        SectionHeader(title = "模型", icon = Icons.Default.Info)
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                StatusRow("已加载", if (mdl.loaded) "是" else "否")
                StatusRow("已选择模型", mdl.selectedName.ifBlank { "无" })
                if (mdl.selectedSizeBytes > 0L) {
                    StatusRow("选择文件大小", formatBytes(mdl.selectedSizeBytes))
                }
                StatusRow("来源", mdl.selectedSource.ifBlank { "-" })
                if (mdl.loaded) {
                    StatusRow("模型名", mdl.name.ifBlank { "-" })
                    StatusRow("后端", mdl.backend.ifBlank { "-" })
                    StatusRow("参数量", mdl.params.ifBlank { "-" })
                    StatusRow("层数", mdl.layers.ifBlank { "-" })
                    StatusRow("嵌入维度", mdl.embedding.ifBlank { "-" })
                    StatusRow("Heads", mdl.heads.ifBlank { "-" })
                    StatusRow("词汇表", mdl.vocabTokens.ifBlank { "-" })
                    if (mdl.sizeBytes > 0L) {
                        StatusRow("模型大小", formatBytes(mdl.sizeBytes))
                    }
                }
            }
        }

        // ---- 上下文 / KV ----
        if (mdl.loaded) {
            val ctx = status.context
            SectionHeader(title = "上下文 / KV (估算)", icon = Icons.Default.Memory)
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    StatusRow("配置上下文", "${ctx.configuredContextSize}")
                    StatusRow("模型上下文 (n_ctx)", ctx.modelContextSize.ifBlank { "-" })
                    StatusRow("训练上下文", ctx.trainContextSize.ifBlank { "-" })
                    StatusRow("Batch", ctx.batchSize.ifBlank { "-" })
                    StatusRow("Micro-Batch", ctx.microBatchSize.ifBlank { "-" })
                    if (ctx.lastTotalTokens > 0) {
                        StatusRow("Last Prompt Tokens", "${ctx.lastPromptTokens}")
                        StatusRow("Last Generated Tokens", "${ctx.lastGeneratedTokens}")
                        StatusRow("Last Total Tokens", "${ctx.lastTotalTokens}")
                        if (ctx.lastElapsedSeconds > 0) {
                            StatusRow("Last 耗时", "%.2f s".format(ctx.lastElapsedSeconds))
                        }
                        if (ctx.lastTokensPerSecond > 0) {
                            StatusRow("Last tok/s", "%.2f".format(ctx.lastTokensPerSecond))
                        }
                        StatusRow("停止原因", ctx.stopReason.ifBlank { "-" })
                    }
                    if (ctx.estimatedKvMemoryMb > 0) {
                        StatusRow("估算 KV 内存", "%.1f MB".format(ctx.estimatedKvMemoryMb))
                    }
                    StatusRow("持久 KV 复用", if (ctx.persistentKvReuseEnabled) "开启" else "关闭")
                    if (ctx.note.isNotBlank()) {
                        Text(
                            text = ctx.note,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.outline
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusRow(
    label: String,
    value: String,
    valueColor: androidx.compose.ui.graphics.Color = MaterialTheme.colorScheme.onSurfaceVariant
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(0.45f)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            color = valueColor,
            modifier = Modifier.weight(0.55f),
            maxLines = 2,
            overflow = TextOverflow.Ellipsis
        )
    }
}

// ================================================================
// 分区标题
// ================================================================

@Composable
private fun SectionHeader(
    title: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector? = null
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        icon?.let {
            Icon(
                imageVector = it,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.width(6.dp))
        }
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.primary
        )
    }
}
