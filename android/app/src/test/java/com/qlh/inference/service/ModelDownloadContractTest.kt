package com.qlh.inference.service

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelDownloadContractTest {
    @Test
    fun `range response appends only at exact manifest offset`() {
        val plan = planModelDownloadWrite(
            existingBytes = 4L,
            statusCode = 206,
            contentLength = 6L,
            contentRange = "bytes 4-9/10",
            expectedTotalBytes = 10L,
        )
        assertEquals(4L, plan.startBytes)
        assertEquals(10L, plan.totalBytes)
        assertTrue(plan.append)
    }

    @Test
    fun `full response restarts an existing partial file`() {
        val plan = planModelDownloadWrite(
            existingBytes = 4L,
            statusCode = 200,
            contentLength = 10L,
            contentRange = null,
            expectedTotalBytes = 10L,
        )
        assertEquals(0L, plan.startBytes)
        assertFalse(plan.append)
    }

    @Test
    fun `mismatched range and model size fail closed`() {
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(4L, 206, 5L, "bytes 5-9/10", 10L)
        }
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(0L, 200, 9L, null, 10L)
        }
        assertThrows(IllegalArgumentException::class.java) {
            planModelDownloadWrite(4L, 206, 7L, "bytes 4-10/10", 10L)
        }
    }
}
