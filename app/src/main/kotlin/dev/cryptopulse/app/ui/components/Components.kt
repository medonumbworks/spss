package dev.cryptopulse.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.cryptopulse.app.ui.theme.LocalMarketColors
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.RiskAssessment
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.util.Text as Fmt
import kotlin.math.abs

/** Coloured circle with the first letters of the symbol (no image downloads needed). */
@Composable
fun CoinAvatar(symbol: String, size: Dp = 40.dp) {
    val hue = (abs(symbol.hashCode()) % 360).toFloat()
    val bg = Color.hsl(hue, 0.55f, 0.45f)
    Box(Modifier.size(size).clip(CircleShape).background(bg), contentAlignment = Alignment.Center) {
        Text(
            text = symbol.take(if (symbol.length <= 3) 3 else 2),
            color = Color.White,
            fontWeight = FontWeight.Bold,
            fontSize = if (symbol.length >= 3) 11.sp else 14.sp,
            maxLines = 1,
        )
    }
}

@Composable
fun ChangePill(pct: Double?, digits: Int = 2) {
    val c = LocalMarketColors.current
    val (fg, bg) = when {
        pct == null -> c.neutral to MaterialTheme.colorScheme.surfaceVariant
        pct >= 0 -> c.up to c.upContainer
        else -> c.down to c.downContainer
    }
    val arrow = when {
        pct == null -> ""
        pct >= 0 -> "▲ "
        else -> "▼ "
    }
    Text(
        text = if (pct == null) "—" else arrow + String.format(java.util.Locale.ROOT, "%.${digits}f%%", abs(pct)),
        color = fg,
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier.clip(RoundedCornerShape(6.dp)).background(bg).padding(horizontal = 6.dp, vertical = 2.dp),
    )
}

/** "24h ▲ 58%" chip coloured by direction. */
@Composable
fun ProbabilityChip(label: String, h: HorizonForecast?) {
    val c = LocalMarketColors.current
    val (fg, bg, arrow) = when (h?.direction) {
        Direction.UP -> Triple(c.up, c.upContainer, "▲")
        Direction.DOWN -> Triple(c.down, c.downContainer, "▼")
        Direction.FLAT -> Triple(c.neutral, MaterialTheme.colorScheme.surfaceVariant, "■")
        null -> Triple(c.neutral, MaterialTheme.colorScheme.surfaceVariant, "·")
    }
    // no number at all for red-flagged, unchecked or too-young tokens (see RiskScorer)
    val noCall = h?.signals?.firstOrNull()?.name in NO_CALL_SIGNALS
    val pct = if (noCall) "—" else h?.let { "${(it.probabilityUp * 100).toInt()}%" } ?: "—"
    Text(
        text = if (noCall) "$label —" else "$label $arrow $pct",
        color = fg,
        style = MaterialTheme.typography.labelSmall,
        fontWeight = FontWeight.Medium,
        modifier = Modifier.clip(RoundedCornerShape(6.dp)).background(bg).padding(horizontal = 6.dp, vertical = 2.dp),
    )
}

private val NO_CALL_SIGNALS = setOf("risk_blocked", "history")

/** "MEME" / "TRENDING" / "DEX" label for coins that Bitpanda does not list. */
@Composable
fun TierBadge(tier: Tier) {
    if (tier == Tier.BITPANDA) return
    val c = LocalMarketColors.current
    val (fg, text) = when (tier) {
        Tier.MEME -> c.warn to "MEME"
        Tier.TRENDING -> c.warn to "TRENDING"
        Tier.DEX -> c.down to "DEX"
        Tier.BITPANDA -> c.neutral to ""
    }
    Text(
        text = text,
        color = fg,
        style = MaterialTheme.typography.labelSmall,
        fontWeight = FontWeight.Bold,
        modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(fg.copy(alpha = 0.14f)).padding(horizontal = 4.dp, vertical = 1.dp),
    )
}

/** Risk score of a non-Bitpanda coin, coloured from amber (elevated) to red (extreme / red flags). */
@Composable
fun RiskPill(risk: RiskAssessment?, long: Boolean = false) {
    risk ?: return
    val c = LocalMarketColors.current
    val color = when {
        risk.blocked || risk.score >= 50 -> c.down
        risk.score >= 25 -> c.warn
        else -> c.neutral
    }
    Text(
        text = if (long) "Risk ${risk.score}/100 · ${risk.label}" else if (risk.blocked) risk.label else "Risk ${risk.score}",
        color = color,
        style = MaterialTheme.typography.labelSmall,
        fontWeight = FontWeight.SemiBold,
        maxLines = 1,
        modifier = Modifier.clip(RoundedCornerShape(6.dp)).background(color.copy(alpha = 0.12f)).padding(horizontal = 6.dp, vertical = 2.dp),
    )
}

/** Small 0-100 bar: green near a local low, red near a local high. */
@Composable
fun BottomMeter(score: Int?, modifier: Modifier = Modifier, width: Dp = 64.dp, height: Dp = 6.dp) {
    val c = LocalMarketColors.current
    Box(modifier.width(width).height(height).clip(RoundedCornerShape(3.dp)).background(Brush.horizontalGradient(listOf(c.down, c.warn, c.up)))) {
        if (score != null) {
            val x = (width - 4.dp) * (score.coerceIn(0, 100) / 100f)
            Box(Modifier.padding(start = x).width(4.dp).height(height).background(MaterialTheme.colorScheme.onSurface))
        }
    }
}

@Composable
fun SentimentDot(composite: Double?, samples: Int?) {
    val c = LocalMarketColors.current
    val color = when {
        composite == null || (samples ?: 0) == 0 -> MaterialTheme.colorScheme.outlineVariant
        composite >= 0.08 -> c.up
        composite <= -0.08 -> c.down
        else -> c.neutral
    }
    Box(Modifier.size(8.dp).clip(CircleShape).background(color))
}

/** Horizontal bar used for confidence and source shares. */
@Composable
fun FractionBar(fraction: Double, color: Color, modifier: Modifier = Modifier, height: Dp = 6.dp) {
    Box(modifier.fillMaxWidth().height(height).clip(RoundedCornerShape(3.dp)).background(MaterialTheme.colorScheme.surfaceVariant)) {
        Box(Modifier.fillMaxWidth(fraction.toFloat().coerceIn(0f, 1f)).height(height).background(color))
    }
}

/** Signed bar centred at zero for a signal's contribution. */
@Composable
fun ContributionBar(value: Double, maxAbs: Double, modifier: Modifier = Modifier) {
    val c = LocalMarketColors.current
    val frac = (abs(value) / maxAbs.coerceAtLeast(1e-9)).toFloat().coerceIn(0f, 1f)
    Row(modifier.width(80.dp).height(8.dp)) {
        Box(Modifier.weight(1f).height(8.dp), contentAlignment = Alignment.CenterEnd) {
            if (value < 0) Box(Modifier.fillMaxWidth(frac).height(8.dp).clip(RoundedCornerShape(topStart = 3.dp, bottomStart = 3.dp)).background(c.down))
        }
        Box(Modifier.width(1.dp).height(8.dp).background(MaterialTheme.colorScheme.outline))
        Box(Modifier.weight(1f).height(8.dp), contentAlignment = Alignment.CenterStart) {
            if (value > 0) Box(Modifier.fillMaxWidth(frac).height(8.dp).clip(RoundedCornerShape(topEnd = 3.dp, bottomEnd = 3.dp)).background(c.up))
        }
    }
}

@Composable
fun SectionCard(title: String, modifier: Modifier = Modifier, subtitle: String? = null, content: @Composable () -> Unit) {
    Card(
        modifier = modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            if (subtitle != null) Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            content()
        }
    }
}

@Composable
fun KeyValue(key: String, value: String, modifier: Modifier = Modifier) {
    Row(modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(key, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.width(12.dp))
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

fun priceText(value: Double?, currency: String): String = Fmt.formatPrice(value, currency)
fun capText(value: Double?, currency: String): String = if (value == null) "—" else "${Fmt.formatCompact(value)} $currency"
