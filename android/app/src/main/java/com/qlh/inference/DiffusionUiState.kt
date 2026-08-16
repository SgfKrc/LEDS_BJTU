package com.qlh.inference

import com.qlh.inference.network.DiffusionJob

/** UI-facing state for a remote PC Stable Diffusion request. */
data class DiffusionUiState(
    val jobId: String? = null,
    val state: String = "idle",
    val progressStep: Int = 0,
    val progressTotal: Int = 0,
    val outputBlobId: String? = null,
    val imageBytes: ByteArray? = null,
    val imageContentType: String? = null,
    val error: String? = null,
    val isCancelling: Boolean = false,
) {
    val isBusy: Boolean
        get() = state == "uploading" || state == "submitting" || state == "queued" || state == "running" ||
            state == "cancelling" || state == "downloading"

    val canCancel: Boolean
        get() = jobId != null && !isCancelling && state in setOf("queued", "running", "cancelling")
}

fun startDiffusionSubmission(state: DiffusionUiState, usesReferenceImage: Boolean): DiffusionUiState =
    state.copy(
        jobId = null,
        state = if (usesReferenceImage) "uploading" else "submitting",
        progressStep = 0,
        progressTotal = 0,
        outputBlobId = null,
        imageBytes = null,
        imageContentType = null,
        error = null,
        isCancelling = false,
    )

fun applyDiffusionJob(state: DiffusionUiState, job: DiffusionJob): DiffusionUiState =
    state.copy(
        jobId = job.jobId,
        state = job.state,
        progressStep = job.progress.step,
        progressTotal = job.progress.total,
        outputBlobId = job.outputBlobId ?: job.blob?.blobId,
        error = job.error,
        isCancelling = job.cancelRequested && !job.isTerminal,
    )

fun beginDiffusionResultDownload(state: DiffusionUiState, blobId: String): DiffusionUiState =
    state.copy(
        state = "downloading",
        outputBlobId = blobId,
        error = null,
        isCancelling = false,
    )

fun completeDiffusionResultDownload(
    state: DiffusionUiState,
    bytes: ByteArray,
    contentType: String,
): DiffusionUiState = state.copy(
    state = "completed",
    imageBytes = bytes,
    imageContentType = contentType,
    error = null,
    isCancelling = false,
)

fun failDiffusion(state: DiffusionUiState, error: String): DiffusionUiState = state.copy(
    state = "failed",
    error = error,
    isCancelling = false,
)

fun markDiffusionCancelling(state: DiffusionUiState): DiffusionUiState =
    if (state.canCancel) state.copy(state = "cancelling", isCancelling = true, error = null) else state
