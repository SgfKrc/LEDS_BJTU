package com.qlh.inference

import com.qlh.inference.network.DiffusionJob
import com.qlh.inference.network.DiffusionProgress
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DiffusionUiStateTest {
    @Test
    fun `job snapshots expose progress and terminal state`() {
        val started = startDiffusionSubmission(DiffusionUiState(), usesReferenceImage = true)
        assertEquals("uploading", started.state)
        assertTrue(started.isBusy)

        val running = applyDiffusionJob(
            started,
            DiffusionJob(
                jobId = "sdjob_1",
                state = "running",
                progress = DiffusionProgress(step = 4, total = 28),
            ),
        )
        assertEquals("sdjob_1", running.jobId)
        assertEquals(4, running.progressStep)
        assertEquals(28, running.progressTotal)
        assertTrue(running.canCancel)
    }

    @Test
    fun `result download and failure are mutually visible states`() {
        val running = applyDiffusionJob(
            DiffusionUiState(),
            DiffusionJob(jobId = "sdjob_2", state = "completed", outputBlobId = "out_2"),
        )
        val downloading = beginDiffusionResultDownload(running, "out_2")
        assertEquals("downloading", downloading.state)
        val completed = completeDiffusionResultDownload(
            downloading,
            "image".toByteArray(),
            "image/png",
        )
        assertEquals("completed", completed.state)
        assertEquals("image/png", completed.imageContentType)
        assertEquals("image", completed.imageBytes!!.toString(Charsets.UTF_8))
        assertFalse(completed.isBusy)

        val failed = failDiffusion(running, "server rejected request")
        assertEquals("failed", failed.state)
        assertEquals("server rejected request", failed.error)
        assertNull(failed.imageBytes)
    }

    @Test
    fun `cancelling fences duplicate cancel actions`() {
        val running = applyDiffusionJob(
            DiffusionUiState(),
            DiffusionJob(jobId = "sdjob_3", state = "running"),
        )
        val cancelling = markDiffusionCancelling(running)
        assertEquals("cancelling", cancelling.state)
        assertTrue(cancelling.isCancelling)
        assertFalse(cancelling.canCancel)
        assertEquals(cancelling, markDiffusionCancelling(cancelling))
    }
}
