package com.qlh.inference.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import com.qlh.inference.network.GgufModelInfo
import com.qlh.inference.service.ModelManager
import com.qlh.inference.status.AndroidRuntimeStatus
import com.qlh.inference.ui.theme.QlhTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

class SettingsScreenTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun themeModeButtonsAreVisibleAndEmitExpectedModes() {
        val selectedModes = mutableListOf<String>()
        var refreshCalled = false

        setSettingsContent(
            themeMode = "system",
            onThemeModeChange = { selectedModes += it },
            onRefreshRuntimeStatus = { refreshCalled = true },
        )

        composeRule.onNodeWithTag("settings_screen").assertIsDisplayed()
        composeRule.onNodeWithTag("settings_theme_system").assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_light").assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_dark").assertIsDisplayed().assertIsEnabled()

        composeRule.onNodeWithTag("settings_theme_light").performClick()
        composeRule.onNodeWithTag("settings_theme_dark").performClick()
        composeRule.onNodeWithTag("settings_theme_system").performClick()

        composeRule.runOnIdle {
            assertEquals(listOf("light", "dark", "system"), selectedModes)
            assertTrue(refreshCalled)
        }
    }

    @Test
    fun settingsScreenRendersThemeControlsInDarkTheme() {
        setSettingsContent(themeMode = "dark", darkTheme = true)

        composeRule.onNodeWithTag("settings_screen").assertIsDisplayed()
        composeRule.onNodeWithTag("settings_theme_dark").assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_light").assertIsDisplayed().assertIsEnabled()
        composeRule.onNodeWithTag("settings_theme_system").assertIsDisplayed().assertIsEnabled()
    }

    @Test
    fun remoteModelCatalogShowsVerifiedModelAndEmitsDownload() {
        var selected: GgufModelInfo? = null
        val model = GgufModelInfo(
            filename = "tiny.gguf",
            sizeBytes = 1_048_576L,
            sizeMb = 1.0,
            sha256 = "a".repeat(64),
            downloadUrl = "/api/models/download/tiny.gguf",
        )

        setSettingsContent(
            themeMode = "system",
            inferenceMode = "full",
            modelTreeUri = "content://example/tree/models",
            remoteModels = listOf(model),
            onDownloadRemoteModel = { selected = it },
        )

        composeRule.onNodeWithTag("remote_model_panel").assertIsDisplayed()
        composeRule.onNodeWithTag("remote_model_download_tiny.gguf").assertIsEnabled().performClick()
        composeRule.runOnIdle { assertEquals(model, selected) }
    }

    @Test
    fun highFrequencyStatusAndThinkingControlStayAvailable() {
        var showThinking = false
        setSettingsContent(
            themeMode = "system",
            onShowThinkingChange = { showThinking = it }
        )

        composeRule.onNodeWithTag("settings_device_details").assertIsDisplayed()
        composeRule.onNodeWithTag("settings_show_thinking").assertIsDisplayed().performClick()
        composeRule.runOnIdle { assertTrue(showThinking) }
    }

    private fun setSettingsContent(
        themeMode: String,
        darkTheme: Boolean = false,
        inferenceMode: String = "thin",
        modelTreeUri: String = "",
        remoteModels: List<GgufModelInfo> = emptyList(),
        onThemeModeChange: (String) -> Unit = {},
        onRefreshRuntimeStatus: () -> Unit = {},
        onDownloadRemoteModel: (GgufModelInfo) -> Unit = {},
        onShowThinkingChange: (Boolean) -> Unit = {},
    ) {
        composeRule.setContent {
            QlhTheme(darkTheme = darkTheme) {
                SettingsScreen(
                    serverHost = "100.64.0.1",
                    serverPort = 8000,
                    inferenceMode = inferenceMode,
                    maxTokens = 512,
                    temperature = 0.7f,
                    topP = 0.9f,
                    contextSize = 2048,
                    showThinking = false,
                    themeMode = themeMode,
                    modelTreeUri = modelTreeUri,
                    selectedModelUri = "",
                    modelStorageMode = "saf",
                    availableModels = emptyList<ModelManager.ModelDocument>(),
                    selectedModelName = "",
                    selectedModelSizeBytes = 0L,
                    isScanningModels = false,
                    modelMessage = null,
                    onServerHostChange = {},
                    onServerPortChange = {},
                    onInferenceModeChange = {},
                    onMaxTokensChange = {},
                    onTemperatureChange = {},
                    onTopPChange = {},
                    onContextSizeChange = {},
                    onShowThinkingChange = onShowThinkingChange,
                    onThemeModeChange = onThemeModeChange,
                    onChooseModelDirectory = {},
                    onRefreshModels = {},
                    onModelSelected = {},
                    onDeleteSelectedModel = {},
                    runtimeStatus = AndroidRuntimeStatus(),
                    runtimeStatusLoading = false,
                    runtimeStatusError = null,
                    onRefreshRuntimeStatus = onRefreshRuntimeStatus,
                    remoteModels = remoteModels,
                    onDownloadRemoteModel = onDownloadRemoteModel,
                )
            }
        }
    }
}
