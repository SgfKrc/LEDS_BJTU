package com.qlh.inference.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val LightColorScheme = lightColorScheme(
    primary = Blue40,
    onPrimary = Color.White,
    primaryContainer = Blue90,
    onPrimaryContainer = Blue10,
    secondary = Teal40,
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFB2DFDB),
    onSecondaryContainer = Color(0xFF00332E),
    tertiary = Blue60,
    background = Gray95,
    onBackground = Gray10,
    surface = Color.White,
    onSurface = Gray10,
    surfaceVariant = Gray90,
    onSurfaceVariant = Gray30,
    outline = Gray80,
    error = ErrorRed,
    errorContainer = ErrorRedLight
)

private val DarkColorScheme = darkColorScheme(
    primary = Blue60,
    onPrimary = Blue10,
    primaryContainer = Blue30,
    onPrimaryContainer = Blue90,
    secondary = Teal60,
    onSecondary = Color(0xFF00332E),
    secondaryContainer = Color(0xFF00594D),
    onSecondaryContainer = Color(0xFFB2DFDB),
    tertiary = Blue80,
    background = Gray10,
    onBackground = Gray95,
    surface = Gray20,
    onSurface = Gray95,
    surfaceVariant = Gray30,
    onSurfaceVariant = Gray80,
    outline = Gray40,
    error = Color(0xFFEF5350),
    errorContainer = Color(0xFF5C1A1A)
)

@Composable
fun QlhTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

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
