package com.qlh.inference.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.junit.Assert.assertEquals
import org.junit.Test

class ThemeTokenTest {
    @Test
    fun neutralSchemesKeepBlackAndWhiteAsThePrimaryReadingContrast() {
        assertEquals(Color(0xFFFFFFFF), BackgroundLight)
        assertEquals(Color(0xFF151515), OnBackgroundLight)
        assertEquals(Color(0xFF000000), BackgroundDark)
        assertEquals(Color(0xFFF5F5F5), OnBackgroundDark)
        assertEquals(Color(0xFF151515), PrimaryLight)
        assertEquals(Color(0xFFF3F3F3), PrimaryDark)
    }

    @Test
    fun sharedTokensKeepCompactCornersAndZeroLetterSpacing() {
        assertEquals(8.dp, QlhShapeTokens.control)
        assertEquals(12.dp, QlhShapeTokens.dialog)
        assertEquals(0.sp, QlhTypography.titleMedium.letterSpacing)
        assertEquals(0.sp, QlhTypography.bodyMedium.letterSpacing)
        assertEquals(0.sp, QlhTypography.labelMedium.letterSpacing)
    }
}
