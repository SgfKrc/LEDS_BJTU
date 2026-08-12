package com.qlh.inference.ui

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import com.qlh.inference.data.MessageEntity
import com.qlh.inference.ui.components.EmptyState
import com.qlh.inference.ui.components.QlhTopBar
import com.qlh.inference.ui.components.StatusChip
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// ================================================================
// 聊天主界面
// ================================================================

@Composable
fun ChatScreen(
    sessionId: Long,
    sessionTitle: String,
    messages: List<MessageEntity>,
    isLoading: Boolean,
    error: String?,
    onSendMessage: (String) -> Unit,
    onRetry: () -> Unit,
    onClearError: () -> Unit,
    modifier: Modifier = Modifier,
    inferenceMode: String = ""
) {
    val listState = rememberLazyListState()
    val focusManager = LocalFocusManager.current
    val clipboardManager = LocalClipboardManager.current
    val context = LocalContext.current
    var inputText by rememberSaveable { mutableStateOf("") }

    val copyAssistantMessage: (String) -> Unit = { text ->
        clipboardManager.setText(AnnotatedString(text))
        Toast.makeText(context, "回答已复制", Toast.LENGTH_SHORT).show()
    }

    // 新消息到达时自动滚动到底部
    LaunchedEffect(messages.size, isLoading, error) {
        val messageItemCount = if (messages.isEmpty() && !isLoading) 1 else messages.size
        val loadingItemCount = if (isLoading) 1 else 0
        val errorItemCount = if (error != null) 1 else 0
        val spacerItemCount = 1
        val itemCount = messageItemCount + loadingItemCount + errorItemCount + spacerItemCount
        if (itemCount > 0) {
            listState.animateScrollToItem(itemCount - 1)
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .imePadding()
            .testTag("chat_screen")
    ) {
        // ---- 顶栏：会话标题 + 模式角标 ----
        QlhTopBar(
            title = sessionTitle.ifBlank { "对话" },
            actions = {
                if (inferenceMode.isNotBlank()) {
                    if (inferenceMode == "thin") {
                        StatusChip(
                            text = "远程推理",
                            containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                            contentColor = MaterialTheme.colorScheme.onTertiaryContainer
                        )
                    } else {
                        StatusChip(
                            text = "本地推理",
                            containerColor = MaterialTheme.colorScheme.primaryContainer,
                            contentColor = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    }
                }
            }
        )
        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))

        // ---- 消息列表 ----
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (messages.isEmpty() && !isLoading) {
                item(key = "empty") {
                    EmptyChatHint()
                }
            }

            items(messages, key = { it.id }) { msg ->
                ChatBubble(
                    message = msg,
                    onCopyAssistantMessage = copyAssistantMessage
                )
            }

            // 加载 / 错误指示器
            if (isLoading) {
                item(key = "loading") {
                    LoadingBubble()
                }
            }
            if (error != null) {
                item(key = "error") {
                    ErrorBubble(error = error, onRetry = {
                        onRetry()
                        onClearError()
                    })
                }
            }

            // 底部留白，避免被输入框遮挡
            item(key = "spacer") {
                Spacer(modifier = Modifier.height(4.dp))
            }
        }

        // ---- 输入框 ----
        ChatInputBar(
            text = inputText,
            onTextChange = { inputText = it },
            onSend = {
                if (inputText.isNotBlank()) {
                    onSendMessage(inputText.trim())
                    inputText = ""
                    focusManager.clearFocus()
                }
            },
            enabled = !isLoading
        )
    }
}

// ================================================================
// 聊天气泡
// ================================================================

@Composable
private fun ChatBubble(
    message: MessageEntity,
    onCopyAssistantMessage: (String) -> Unit
) {
    val isUser = message.role == "user"

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        // 对侧留白，气泡最大宽度约 85%
        if (isUser) {
            Spacer(modifier = Modifier.width(48.dp))
        }

        Column(
            modifier = Modifier.weight(1f, fill = false),
            horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
        ) {
            Surface(
                shape = RoundedCornerShape(
                    topStart = 20.dp,
                    topEnd = 20.dp,
                    bottomStart = if (isUser) 20.dp else 6.dp,
                    bottomEnd = if (isUser) 6.dp else 20.dp
                ),
                color = if (isUser) {
                    MaterialTheme.colorScheme.primary
                } else {
                    MaterialTheme.colorScheme.surfaceContainerHigh
                }
            ) {
                Box {
                    Text(
                        text = message.content,
                        modifier = Modifier.padding(
                            start = 14.dp,
                            top = 10.dp,
                            end = if (!isUser && message.content.isNotBlank()) 44.dp else 14.dp,
                            bottom = 10.dp
                        ),
                        color = if (isUser) {
                            MaterialTheme.colorScheme.onPrimary
                        } else {
                            MaterialTheme.colorScheme.onSurface
                        },
                        style = MaterialTheme.typography.bodyLarge
                    )
                    if (!isUser && message.content.isNotBlank()) {
                        IconButton(
                            onClick = { onCopyAssistantMessage(message.content) },
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .size(36.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.ContentCopy,
                                contentDescription = "复制回答",
                                modifier = Modifier.size(15.dp),
                                tint = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }

            // 生成指标（引擎 / tok/s 等）— 弱化为小标签
            if (!isUser && !message.metrics.isNullOrBlank()) {
                Text(
                    text = formatMetrics(message.metrics),
                    modifier = Modifier.padding(top = 4.dp, start = 4.dp, end = 4.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // 时间戳
            Text(
                text = formatTime(message.timestamp),
                modifier = Modifier.padding(top = 2.dp, start = 4.dp, end = 4.dp),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline
            )
        }

        if (!isUser) {
            Spacer(modifier = Modifier.width(48.dp))
        }
    }
}

// ================================================================
// 加载指示器
// ================================================================

@Composable
private fun LoadingBubble() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start
    ) {
        Surface(
            shape = RoundedCornerShape(
                topStart = 20.dp,
                topEnd = 20.dp,
                bottomStart = 6.dp,
                bottomEnd = 20.dp
            ),
            color = MaterialTheme.colorScheme.surfaceContainerHigh
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.width(10.dp))
                Text(
                    text = "思考中…",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

// ================================================================
// 错误提示
// ================================================================

@Composable
private fun ErrorBubble(error: String, onRetry: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center
    ) {
        Surface(
            shape = MaterialTheme.shapes.medium,
            color = MaterialTheme.colorScheme.errorContainer
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.ErrorOutline,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onErrorContainer,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "发送失败",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
                Spacer(modifier = Modifier.width(12.dp))
                Button(
                    onClick = onRetry,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error,
                        contentColor = MaterialTheme.colorScheme.onError
                    ),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)
                ) {
                    Icon(
                        Icons.Default.Refresh,
                        contentDescription = null,
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("重试", style = MaterialTheme.typography.labelLarge)
                }
            }
        }
    }
}

// ================================================================
// 空状态提示
// ================================================================

@Composable
private fun EmptyChatHint() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 32.dp, vertical = 72.dp),
        contentAlignment = Alignment.Center
    ) {
        EmptyState(
            icon = Icons.AutoMirrored.Filled.Chat,
            title = "开始新的对话",
            subtitle = "输入消息后发送，AI 将为你生成回复"
        )
    }
}

// ================================================================
// 输入栏
// ================================================================

@Composable
private fun ChatInputBar(
    text: String,
    onTextChange: (String) -> Unit,
    onSend: () -> Unit,
    enabled: Boolean
) {
    Surface(
        modifier = Modifier.navigationBarsPadding(),
        color = MaterialTheme.colorScheme.surfaceContainerLow
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            OutlinedTextField(
                value = text,
                onValueChange = onTextChange,
                modifier = Modifier
                    .weight(1f)
                    .testTag("chat_input"),
                placeholder = {
                    Text(
                        "输入消息…",
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                },
                maxLines = 4,
                enabled = enabled,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { onSend() }),
                shape = RoundedCornerShape(24.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                    unfocusedBorderColor = Color.Transparent,
                    focusedContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh
                )
            )
            Spacer(modifier = Modifier.width(10.dp))
            IconButton(
                onClick = onSend,
                enabled = enabled && text.isNotBlank(),
                modifier = Modifier
                    .size(48.dp)
                    .testTag("chat_send")
                    .clip(CircleShape)
                    .background(
                        if (text.isNotBlank() && enabled) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.surfaceContainerHighest
                        }
                    )
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.Send,
                    contentDescription = "发送",
                    tint = if (text.isNotBlank() && enabled) {
                        MaterialTheme.colorScheme.onPrimary
                    } else {
                        MaterialTheme.colorScheme.outline
                    }
                )
            }
        }
    }
}

// ================================================================
// 工具函数
// ================================================================

private fun formatTime(timestamp: Long): String {
    val sdf = SimpleDateFormat("HH:mm", Locale.getDefault())
    return sdf.format(Date(timestamp))
}

private fun formatMetrics(metricsJson: String): String {
    return try {
        val obj = JSONObject(metricsJson)
        val parts = mutableListOf<String>()
        val engine = obj.optString("engine", obj.optString("execution_mode", ""))
        if (engine.isNotBlank()) parts += engine.replace("distributed_pipeline", "Pipeline")
        if (obj.has("distributed_used")) {
            parts += if (obj.optBoolean("distributed_used")) "分布式: 是" else "分布式: 否"
        }
        val tokens = listOf("generated_tokens", "new_tokens", "completion_tokens", "tokens_generated")
            .firstNotNullOfOrNull { key -> obj.optInt(key, 0).takeIf { it > 0 } }
        if (tokens != null) parts += "$tokens tokens"
        val tps = when {
            obj.optDouble("tokens_per_second", 0.0) > 0 -> obj.optDouble("tokens_per_second")
            obj.optDouble("tokens_per_sec", 0.0) > 0 -> obj.optDouble("tokens_per_sec")
            else -> 0.0
        }
        if (tps > 0) parts += "%.1f tok/s".format(tps)
        val route = obj.optString("route", "")
        if (route.isNotBlank()) parts += route
        val fallback = obj.optString("fallback_reason", "")
        if (fallback.isNotBlank()) parts += "回退: $fallback"
        parts.joinToString(" · ").ifBlank { metricsJson }
    } catch (_: Exception) {
        metricsJson
    }
}
