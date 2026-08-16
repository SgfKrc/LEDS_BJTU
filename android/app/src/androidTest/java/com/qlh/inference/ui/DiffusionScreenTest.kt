package com.qlh.inference.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import com.qlh.inference.DiffusionUiState
import com.qlh.inference.network.DiffusionBlobUpload
import com.qlh.inference.network.DiffusionGenerateRequest
import com.qlh.inference.ui.theme.QlhTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

class DiffusionScreenTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun `prompt submission emits generation request`() {
        var request: DiffusionGenerateRequest? = null
        setContent(state = DiffusionUiState(), onSubmit = { value, _ -> request = value })

        composeRule.onNodeWithTag("diffusion_prompt").performTextInput("  lighthouse  ")
        composeRule.onNodeWithTag("diffusion_submit").assertIsEnabled().performClick()

        composeRule.runOnIdle {
            assertEquals("lighthouse", request?.prompt)
            assertEquals(28, request?.steps)
            assertEquals(512, request?.width)
            assertEquals(512, request?.height)
        }
    }

    @Test
    fun `busy state disables editing and shows result placeholder`() {
        var cancelled = false
        setContent(
            state = DiffusionUiState(jobId = "sdjob_1", state = "running"),
            onCancel = { cancelled = true },
        )

        composeRule.onNodeWithTag("diffusion_screen").assertIsDisplayed()
        composeRule.onNodeWithTag("diffusion_submit").assertIsNotEnabled()
        composeRule.onNodeWithTag("diffusion_cancel").assertIsDisplayed().performClick()
        composeRule.onNodeWithText("生成中").assertIsDisplayed()
        composeRule.onNodeWithTag("diffusion_result_empty").assertIsDisplayed()
        composeRule.runOnIdle { assertTrue(cancelled) }
    }

    @Test
    fun `failed state displays actionable error`() {
        setContent(
            state = DiffusionUiState(state = "failed", error = "主节点不可用"),
        )
        composeRule.onNodeWithText("失败").assertIsDisplayed()
        composeRule.onNodeWithText("主节点不可用").assertIsDisplayed()
    }

    private fun setContent(
        state: DiffusionUiState,
        onSubmit: (DiffusionGenerateRequest, DiffusionBlobUpload?) -> Unit = { _, _ -> },
        onCancel: () -> Unit = {},
    ) {
        composeRule.setContent {
            QlhTheme(darkTheme = false) {
                DiffusionScreen(
                    state = state,
                    onSubmit = onSubmit,
                    onCancel = onCancel,
                )
            }
        }
    }
}
