package com.qlh.inference.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.qlh.inference.DiffusionUiState
import com.qlh.inference.media.DiffusionImageDecoder
import com.qlh.inference.media.ImageAttachmentEncoder
import com.qlh.inference.network.DiffusionBlobUpload
import com.qlh.inference.network.DiffusionGenerateRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun DiffusionScreen(
    state: DiffusionUiState,
    onSubmit: (DiffusionGenerateRequest, DiffusionBlobUpload?) -> Unit,
    onCancel: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var prompt by rememberSaveable { mutableStateOf("") }
    var negativePrompt by rememberSaveable { mutableStateOf("") }
    var stepsText by rememberSaveable { mutableStateOf("28") }
    var referenceImage by remember { mutableStateOf<DiffusionBlobUpload?>(null) }
    var referenceError by remember { mutableStateOf<String?>(null) }
    val referencePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent(),
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        referenceError = null
        scope.launch {
            val encoded = withContext(Dispatchers.IO) {
                ImageAttachmentEncoder.encode(context.contentResolver, uri)
            }
            encoded.onSuccess { image ->
                referenceImage = DiffusionBlobUpload(
                    data = image.previewBytes,
                    fileName = "reference.jpg",
                    contentType = "image/jpeg",
                )
            }.onFailure { error ->
                referenceError = error.message ?: "参考图读取失败"
            }
        }
    }

    val referencePreview = remember(referenceImage?.data) {
        referenceImage?.data?.let(DiffusionImageDecoder::decodeThumbnail)?.asImageBitmap()
    }
    val resultPreview = remember(state.imageBytes) {
        state.imageBytes?.let(DiffusionImageDecoder::decodeThumbnail)?.asImageBitmap()
    }
    val steps = stepsText.toIntOrNull()?.coerceIn(1, 100) ?: 28
    val canSubmit = prompt.trim().isNotEmpty() && !state.isBusy

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .imePadding()
            .navigationBarsPadding()
            .padding(horizontal = 16.dp, vertical = 12.dp)
            .testTag("diffusion_screen"),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(Icons.Default.Image, contentDescription = null)
            Spacer(Modifier.width(10.dp))
            Text("图像生成", style = MaterialTheme.typography.headlineSmall)
            Spacer(Modifier.weight(1f))
            if (state.canCancel) {
                IconButton(
                    onClick = onCancel,
                    enabled = !state.isCancelling,
                    modifier = Modifier.testTag("diffusion_cancel"),
                ) {
                    Icon(Icons.Default.Close, contentDescription = "取消任务")
                }
            }
        }

        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            modifier = Modifier.fillMaxWidth().testTag("diffusion_prompt"),
            enabled = !state.isBusy,
            label = { Text("提示词") },
            minLines = 3,
        )
        OutlinedTextField(
            value = negativePrompt,
            onValueChange = { negativePrompt = it },
            modifier = Modifier.fillMaxWidth().testTag("diffusion_negative_prompt"),
            enabled = !state.isBusy,
            label = { Text("反向提示词") },
            minLines = 2,
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value = stepsText,
                onValueChange = { stepsText = it.filter(Char::isDigit).take(3) },
                modifier = Modifier.width(120.dp).testTag("diffusion_steps"),
                enabled = !state.isBusy,
                label = { Text("步数") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                singleLine = true,
            )
            Text("512 × 512", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(
                onClick = { referencePicker.launch("image/*") },
                enabled = !state.isBusy,
                modifier = Modifier.testTag("diffusion_reference_pick"),
            ) {
                Icon(Icons.Default.AddPhotoAlternate, contentDescription = "选择参考图")
            }
            Text(
                if (referenceImage == null) "可选参考图" else "已选择参考图",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (referencePreview != null) {
                Image(
                    bitmap = referencePreview,
                    contentDescription = "参考图预览",
                    modifier = Modifier.size(52.dp).clip(RoundedCornerShape(8.dp)),
                )
                IconButton(
                    onClick = { referenceImage = null },
                    enabled = !state.isBusy,
                    modifier = Modifier.testTag("diffusion_reference_clear"),
                ) {
                    Icon(Icons.Default.Close, contentDescription = "移除参考图")
                }
            }
        }
        if (referenceError != null) {
            Text(
                referenceError!!,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.testTag("diffusion_reference_error"),
            )
        }

        Button(
            onClick = {
                onSubmit(
                    DiffusionGenerateRequest(
                        prompt = prompt.trim(),
                        negativePrompt = negativePrompt.trim().ifEmpty { null },
                        steps = steps,
                        width = 512,
                        height = 512,
                    ),
                    referenceImage,
                )
            },
            enabled = canSubmit,
            modifier = Modifier.fillMaxWidth().testTag("diffusion_submit"),
            contentPadding = ButtonDefaults.ButtonWithIconContentPadding,
        ) {
            if (state.state == "submitting" || state.state == "uploading") {
                CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
            } else {
                Icon(Icons.Default.AutoAwesome, contentDescription = null)
            }
            Spacer(Modifier.width(8.dp))
            Text(if (referenceImage == null) "生成图像" else "生成变体")
        }

        DiffusionStatus(state)

        HorizontalDivider()
        Text("结果", style = MaterialTheme.typography.titleMedium)
        if (resultPreview != null) {
            Image(
                bitmap = resultPreview,
                contentDescription = "生成结果",
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
                    .testTag("diffusion_result"),
            )
        } else {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(12.dp))
                    .testTag("diffusion_result_empty"),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    if (state.imageBytes == null) "暂无结果" else "结果图片解码失败",
                    color = if (state.imageBytes == null) {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    } else {
                        MaterialTheme.colorScheme.error
                    },
                )
            }
        }
    }
}

@Composable
private fun DiffusionStatus(state: DiffusionUiState) {
    val label = when (state.state) {
        "uploading" -> "上传参考图…"
        "submitting" -> "提交任务…"
        "queued" -> "排队中"
        "running" -> "生成中"
        "cancelling" -> "取消中"
        "downloading" -> "下载结果…"
        "completed" -> "已完成"
        "cancelled" -> "已取消"
        "failed" -> "失败"
        else -> "待命"
    }
    Surface(
        modifier = Modifier.fillMaxWidth().testTag("diffusion_status"),
        color = if (state.state == "failed") {
            MaterialTheme.colorScheme.errorContainer
        } else {
            MaterialTheme.colorScheme.surfaceContainerHighest
        },
        shape = RoundedCornerShape(8.dp),
    ) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (state.isBusy) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                }
                Text(label)
                if (state.jobId != null) {
                    Spacer(Modifier.width(8.dp))
                    Text(
                        state.jobId,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
            }
            if (state.progressTotal > 0 && state.state == "running") {
                LinearProgressIndicator(
                    progress = { (state.progressStep.toFloat() / state.progressTotal).coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            if (!state.error.isNullOrBlank()) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.ErrorOutline, contentDescription = null)
                    Spacer(Modifier.width(6.dp))
                    Text(state.error, color = MaterialTheme.colorScheme.error)
                }
            }
        }
    }
}
