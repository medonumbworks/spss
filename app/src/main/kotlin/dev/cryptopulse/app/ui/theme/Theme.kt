package dev.cryptopulse.app.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

/** Up/down colours that stay readable in light and dark mode (not tied to the dynamic palette). */
@Immutable
data class MarketColors(val up: Color, val upContainer: Color, val down: Color, val downContainer: Color, val neutral: Color, val warn: Color)

val LocalMarketColors = staticCompositionLocalOf {
    MarketColors(Color(0xFF138A55), Color(0xFFDDF5E8), Color(0xFFC62E3A), Color(0xFFFBE2E4), Color(0xFF6B7280), Color(0xFFB7791F))
}

private val LightMarket = MarketColors(Color(0xFF138A55), Color(0xFFDDF5E8), Color(0xFFC62E3A), Color(0xFFFBE2E4), Color(0xFF6B7280), Color(0xFFB7791F))
private val DarkMarket = MarketColors(Color(0xFF4ADE80), Color(0xFF12321F), Color(0xFFF87171), Color(0xFF3B1518), Color(0xFF9CA3AF), Color(0xFFFBBF24))

private val LightScheme = lightColorScheme(
    primary = Color(0xFF1F6FEB),
    secondary = Color(0xFF0F766E),
    tertiary = Color(0xFFB45309),
    background = Color(0xFFF7F8FA),
    surface = Color(0xFFFFFFFF),
)
private val DarkScheme = darkColorScheme(
    primary = Color(0xFF79A8FF),
    secondary = Color(0xFF5EEAD4),
    tertiary = Color(0xFFFCD34D),
    background = Color(0xFF0E1116),
    surface = Color(0xFF151A21),
)

@Composable
fun CryptoPulseTheme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    val context = LocalContext.current
    val scheme = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        darkTheme -> DarkScheme
        else -> LightScheme
    }
    androidx.compose.runtime.CompositionLocalProvider(LocalMarketColors provides if (darkTheme) DarkMarket else LightMarket) {
        MaterialTheme(colorScheme = scheme, content = content)
    }
}
