package dev.cryptopulse.app.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.cryptopulse.app.ui.theme.LocalMarketColors

/**
 * Line chart drawn on a Canvas. When [bandLow]/[bandHigh] are set, the forecast range is shown as
 * dashed lines so the user sees where the price may go next.
 */
@Composable
fun PriceChart(
    values: List<Double>,
    modifier: Modifier = Modifier,
    height: Dp = 180.dp,
    bandLow: Double? = null,
    bandHigh: Double? = null,
    showMinMax: Boolean = true,
) {
    val c = LocalMarketColors.current
    val outline = MaterialTheme.colorScheme.outlineVariant
    val labelColor = MaterialTheme.colorScheme.onSurfaceVariant
    if (values.size < 2) {
        Box(modifier.fillMaxWidth().height(height), contentAlignment = Alignment.Center) {
            Text("No chart data yet", style = MaterialTheme.typography.bodySmall, color = labelColor)
        }
        return
    }
    val lineColor = if (values.last() >= values.first()) c.up else c.down
    val (minV, maxV) = remember(values, bandLow, bandHigh) {
        val all = values + listOfNotNull(bandLow, bandHigh)
        all.min() to all.max()
    }
    Box(modifier.fillMaxWidth().height(height)) {
        Canvas(Modifier.fillMaxWidth().height(height)) {
            val w = size.width
            val h = size.height
            val pad = 6f
            val span = (maxV - minV).takeIf { it > 0 } ?: 1.0
            fun y(v: Double) = (pad + (h - 2 * pad) * (1 - (v - minV) / span)).toFloat()
            val stepX = w / (values.size - 1)
            val path = Path()
            values.forEachIndexed { i, v -> if (i == 0) path.moveTo(0f, y(v)) else path.lineTo(i * stepX, y(v)) }
            val fill = Path().apply {
                addPath(path)
                lineTo(w, h)
                lineTo(0f, h)
                close()
            }
            drawPath(fill, Brush.verticalGradient(listOf(lineColor.copy(alpha = 0.25f), Color.Transparent)))
            drawPath(path, lineColor, style = Stroke(width = 4f, cap = StrokeCap.Round, join = StrokeJoin.Round))
            val dash = PathEffect.dashPathEffect(floatArrayOf(10f, 10f))
            bandHigh?.let { drawLine(c.up, Offset(w * 0.6f, y(it)), Offset(w, y(it)), strokeWidth = 3f, pathEffect = dash) }
            bandLow?.let { drawLine(c.down, Offset(w * 0.6f, y(it)), Offset(w, y(it)), strokeWidth = 3f, pathEffect = dash) }
            drawLine(outline, Offset(0f, h - 1), Offset(w, h - 1), strokeWidth = 1f)
        }
        if (showMinMax) {
            Text(priceText(maxV, ""), style = MaterialTheme.typography.labelSmall, color = labelColor, modifier = Modifier.align(Alignment.TopStart))
            Text(priceText(minV, ""), style = MaterialTheme.typography.labelSmall, color = labelColor, modifier = Modifier.align(Alignment.BottomStart))
        }
    }
}

/** Tiny 7-day sparkline for list rows. */
@Composable
fun Sparkline(values: List<Double>, modifier: Modifier = Modifier) {
    val c = LocalMarketColors.current
    if (values.size < 2) return
    val color = if (values.last() >= values.first()) c.up else c.down
    Canvas(modifier) {
        val minV = values.min()
        val maxV = values.max()
        val span = (maxV - minV).takeIf { it > 0 } ?: 1.0
        val stepX = size.width / (values.size - 1)
        val path = Path()
        values.forEachIndexed { i, v ->
            val y = (size.height * (1 - (v - minV) / span)).toFloat()
            if (i == 0) path.moveTo(0f, y) else path.lineTo(i * stepX, y)
        }
        drawPath(path, color, style = Stroke(width = 2.5f, cap = StrokeCap.Round, join = StrokeJoin.Round))
    }
}
