package com.qlh.inference.ui.theme

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val LightColorScheme = lightColorScheme(
    primary = Ink,
    onPrimary = Paper,
    primaryContainer = Gray90,
    onPrimaryContainer = Ink,
    secondary = Gray40,
    onSecondary = Paper,
    secondaryContainer = SurfaceVariantLight,
    onSecondaryContainer = Ink,
    tertiary = Gray60,
    background = Gray95,
    onBackground = Ink,
    surface = SurfaceLight,
    onSurface = Ink,
    surfaceVariant = SurfaceVariantLight,
    onSurfaceVariant = Gray40,
    outline = Gray80,
    error = ErrorRed,
    errorContainer = ErrorRedLight
)

private val DarkColorScheme = darkColorScheme(
    primary = Paper,
    onPrimary = PaperDark,
    primaryContainer = Gray30,
    onPrimaryContainer = Paper,
    secondary = Gray80,
    onSecondary = PaperDark,
    secondaryContainer = SurfaceVariantDark,
    onSecondaryContainer = Paper,
    tertiary = Gray90,
    background = PaperDark,
    onBackground = Gray95,
    surface = SurfaceDark,
    onSurface = Gray95,
    surfaceVariant = SurfaceVariantDark,
    onSurfaceVariant = Gray80,
    outline = Gray40,
    error = ErrorRed,
    errorContainer = Gray30
)

@Composable
fun QlhTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    // 设置状态栏颜色
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            @Suppress("DEPRECATION")
            window.statusBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = QlhTypography,
        content = content
    )
}
