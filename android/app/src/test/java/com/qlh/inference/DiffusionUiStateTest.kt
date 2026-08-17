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

    // ---- T9：状态矩阵补齐 + failDiffusion 残留语义决策 ----

    @Test
    fun `T9 isBusy covers all in-flight states`() {
        for (s in listOf("uploading", "submitting", "queued", "running",
                         "cancelling", "downloading")) {
            val st = applyDiffusionJob(DiffusionUiState(), DiffusionJob(
                jobId = "j", state = s))
            assertTrue("$s 应视为 busy", st.isBusy)
        }
        for (s in listOf("idle", "completed", "failed", "cancelled")) {
            val st = applyDiffusionJob(DiffusionUiState(), DiffusionJob(
                jobId = "j", state = s))
            assertFalse("$s 不应 busy", st.isBusy)
        }
    }

    @Test
    fun `T9 canCancel requires job plus queued or running state`() {
        // queued/running 可取消
        for (s in listOf("queued", "running")) {
            val st = applyDiffusionJob(DiffusionUiState(), DiffusionJob(
                jobId = "j", state = s))
            assertTrue("$s 应可取消", st.canCancel)
        }
        // uploading/submitting/downloading/completed/failed 不可取消
        for (s in listOf("uploading", "submitting", "downloading",
                         "completed", "failed", "cancelled")) {
            val st = applyDiffusionJob(DiffusionUiState(), DiffusionJob(
                jobId = "j", state = s))
            assertFalse("$s 不应可取消", st.canCancel)
        }
        // 无 jobId 不可取消
        assertFalse(DiffusionUiState().canCancel)
        // 已取消中不可重复取消
        val cancelling = applyDiffusionJob(DiffusionUiState(), DiffusionJob(
            jobId = "j", state = "running", cancelRequested = true))
        assertFalse(cancelling.canCancel)
    }

    @Test
    fun `T9 applyDiffusionJob isCancelling requires non-terminal`() {
        // cancelRequested + running -> isCancelling
        val cancelling = applyDiffusionJob(DiffusionUiState(), DiffusionJob(
            jobId = "j", state = "running", cancelRequested = true))
        assertTrue(cancelling.isCancelling)
        // cancelRequested + 终态 -> 不 isCancelling（applyDiffusionJob: cancelRequested && !isTerminal）
        for (terminal in listOf("completed", "failed", "cancelled")) {
            val st = applyDiffusionJob(DiffusionUiState(), DiffusionJob(
                jobId = "j", state = terminal, cancelRequested = true))
            assertFalse("$terminal 终态不应 isCancelling", st.isCancelling)
        }
    }

    @Test
    fun `T9 restart from failed or completed resets state`() {
        for (prior in listOf("failed", "completed")) {
            val done = applyDiffusionJob(DiffusionUiState(), DiffusionJob(
                jobId = "old", state = prior))
            val restarted = startDiffusionSubmission(done, usesReferenceImage = true)
            assertEquals("uploading", restarted.state)
            assertNull(restarted.jobId)
            assertNull(restarted.imageBytes)
            assertNull(restarted.error)
            assertFalse(restarted.isCancelling)
        }
    }

    @Test
    fun `T9 fail after download retains previous result by design`() {
        // 决策（2026-08-17）：startDiffusionSubmission 已清 imageBytes；若跳过 start
        // 直接 fail（polling 中途失败），保留上次成功图供参考——不留残留语义缺口
        val completed = completeDiffusionResultDownload(
            beginDiffusionResultDownload(DiffusionUiState(), "out_1"),
            "prev".toByteArray(),
            "image/png",
        )
        val failed = failDiffusion(completed, "poll timeout")
        assertEquals("failed", failed.state)
        // 保留上次结果（设计如此，测试锁定该语义防误改）
        assertEquals("prev", failed.imageBytes!!.toString(Charsets.UTF_8))
        // 但正常流程（start 后）失败时图已被清空
        val started = startDiffusionSubmission(DiffusionUiState(), usesReferenceImage = false)
        assertNull(started.imageBytes)
    }
}
