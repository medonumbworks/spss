package dev.cryptopulse.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.cryptopulse.app.ui.components.BottomMeter
import dev.cryptopulse.app.ui.components.ChangePill
import dev.cryptopulse.app.ui.components.CoinIcon
import dev.cryptopulse.app.ui.components.ContributionBar
import dev.cryptopulse.app.ui.components.FractionBar
import dev.cryptopulse.app.ui.components.KeyValue
import dev.cryptopulse.app.ui.components.PriceChart
import dev.cryptopulse.app.ui.components.RiskPill
import dev.cryptopulse.app.ui.components.SectionCard
import dev.cryptopulse.app.ui.components.capText
import dev.cryptopulse.app.ui.components.priceText
import dev.cryptopulse.app.ui.theme.LocalMarketColors
import dev.cryptopulse.core.analysis.forecast.DefaultForecastEngine
import dev.cryptopulse.core.model.AssetDetail
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.BottomAssessment
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.model.SentimentSummary
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.util.Text as Fmt
import kotlin.math.abs

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetailScreen(vm: AppViewModel, symbol: String) {
    val detailState by vm.detail.collectAsStateWithLifecycle()
    // a load for another coin may still be finishing; never show its data on this screen
    val state = detailState?.takeIf { it.symbol == symbol }
    val snapshot by vm.snapshot.collectAsStateWithLifecycle()
    val listRow: AssetRow? = snapshot?.rows?.firstOrNull { it.asset.symbol == symbol }
    // beyond Bitpanda a ticker can move to another token while a detail loads: only show the detail of this one
    val row: AssetRow? = state?.detail?.row?.takeIf { listRow == null || it.asset.tokenKey == listRow.asset.tokenKey } ?: listRow
    // DEX prices stay in USD when there is no USD rate for the list currency; label them with their own
    val currency = row?.market?.currency ?: snapshot?.currency ?: "EUR"

    Scaffold(
        topBar = {
            TopAppBar(
                navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } },
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CoinIcon(symbol, row?.market?.imageUrl, 32.dp)
                        Spacer(Modifier.width(8.dp))
                        Column {
                            Text(row?.asset?.name ?: symbol, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(symbol, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                },
                actions = { IconButton(onClick = { vm.loadDetail(symbol, force = true) }) { Icon(Icons.Filled.Refresh, contentDescription = "Reload") } },
            )
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState())) {
            if (state?.loading == true) {
                LinearProgressIndicator(Modifier.fillMaxWidth())
                Text(
                    if (row?.asset?.tier == Tier.DEX) "Loading hourly pool prices, contract checks, Telegram and news for $symbol…"
                    else "Loading 1-year history, Reddit, StockTwits, Telegram and news for $symbol…",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
                )
            }
            state?.error?.let { Text(it, color = LocalMarketColors.current.down, modifier = Modifier.padding(16.dp)) }
            if (row == null) {
                Text("Unknown coin", modifier = Modifier.padding(16.dp))
                return@Column
            }
            PriceHeader(row, currency, state?.detail)
            if (!row.asset.onBitpanda) SpeculativeCard(row)
            ChartCard(row, state?.detail)
            row.forecast?.let { f ->
                ForecastCard("Next 24 hours", f.h24, noCallReason(row, f.h24), currency)
                ForecastCard("Next 7 days", f.d7, noCallReason(row, f.d7), currency)
                BottomCard(f.bottom, row.upOrNull(f.d7))
            }
            SentimentCard(row.sentiment)
            state?.detail?.let { MoreDataCard(it, currency) }
            val notes = (row.forecast?.coverage?.notes.orEmpty() + state?.detail?.warnings.orEmpty()).distinct()
            if (notes.isNotEmpty()) {
                SectionCard("Data notes") { notes.forEach { Text("• $it", style = MaterialTheme.typography.bodySmall) } }
            }
            Text(Forecast.DISCLAIMER, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(16.dp))
            Spacer(Modifier.height(24.dp))
        }
    }
}

/** Risk, contract checks and where a non-Bitpanda coin trades. */
@Composable
private fun SpeculativeCard(row: AssetRow) {
    val c = LocalMarketColors.current
    val uri = LocalUriHandler.current
    val title = when (row.asset.tier) {
        Tier.MEME -> "Memecoin, not on Bitpanda"
        Tier.TRENDING -> "Trending coin, not on Bitpanda"
        Tier.DEX -> "DEX-only token, not on Bitpanda"
        Tier.BITPANDA -> return
    }
    SectionCard(title, subtitle = "Information only. Most memecoins and DEX tokens lose most of their value; no red flag does not mean safe.") {
        row.risk?.let { r ->
            RiskPill(r, long = true)
            FractionBar(r.score / 100.0, if (r.blocked || r.score >= 50) c.down else c.warn)
            r.reasons.forEach { Text("• $it", style = MaterialTheme.typography.bodySmall) }
        }
        row.dex?.let { d ->
            HorizontalDivider()
            KeyValue("Chain / exchange", listOfNotNull(d.chainId, d.dexId).joinToString(" · "))
            KeyValue("Pool liquidity", capText(d.liquidityUsd, "USD"))
            KeyValue("Volume 24h", capText(d.volumeUsd.h24, "USD"))
            d.pairCreatedAtMs?.let { KeyValue("Pool age", ageText(System.currentTimeMillis() - it)) }
            KeyValue("Trades 24h", "${d.txnsH24.buys} buys / ${d.txnsH24.sells} sells")
            if (d.boostsActive > 0) KeyValue("Paid promotion", "${d.boostsActive} DexScreener boosts")
            if (d.pumpFunGraduated != null) KeyValue("pump.fun", if (d.pumpFunGraduated == true) "graduated to a pool" else "still on the bonding curve")
            row.asset.contract?.let { addr ->
                SelectionContainer { Text("Contract: $addr", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            }
            d.url?.let { u -> Text("Open pool page", color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.labelLarge, modifier = Modifier.clickable { uri.openUri(u) }) }
        }
        row.safety?.let { s ->
            HorizontalDivider()
            Text("Contract checks (${s.source})", style = MaterialTheme.typography.labelLarge)
            s.honeypot?.let { KeyValue("Honeypot", if (it) "YES – cannot sell" else "no") }
            if (s.buyTaxPct != null || s.sellTaxPct != null) KeyValue("Tax buy / sell", "${"%.0f".format(s.buyTaxPct ?: 0.0)}% / ${"%.0f".format(s.sellTaxPct ?: 0.0)}%")
            s.mintable?.let { KeyValue("New tokens can be minted", if (it) "yes" else "no") }
            s.freezable?.let { KeyValue("Holders can be frozen", if (it) "yes" else "no") }
            s.holderCount?.let { KeyValue("Holders", "%,d".format(it)) }
            s.top10HolderPct?.let { KeyValue("Top 10 wallets (pools incl.)", "${"%.0f".format(it)}%") }
            if (s.cexListed.isNotEmpty()) KeyValue("Also listed on", s.cexListed.take(3).joinToString())
            s.ownerPowers.forEach { Text("• $it", style = MaterialTheme.typography.bodySmall, color = c.down) }
        }
    }
}

private fun ageText(ms: Long): String {
    val h = ms / 3_600_000.0
    return if (h < 48) "${h.toInt().coerceAtLeast(0)} h" else "${(h / 24).toInt()} days"
}

@Composable
private fun PriceHeader(row: AssetRow, currency: String, detail: AssetDetail?) {
    val m = row.market
    Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
        Text(priceText(m?.bitpandaPrice ?: m?.price, currency), style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("24h", style = MaterialTheme.typography.labelMedium); ChangePill(m?.change24hPct)
            Text("7d", style = MaterialTheme.typography.labelMedium); ChangePill(m?.change7dPct)
            Text("30d", style = MaterialTheme.typography.labelMedium); ChangePill(m?.change30dPct, 1)
        }
        row.forecast?.summary?.let { Text(it, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(top = 8.dp)) }
        if (m?.bitpandaPrice != null && m.price != null && m.source != "bitpanda") {
            val diff = (m.bitpandaPrice!! / m.price!! - 1) * 100
            if (abs(diff) > 0.3) Text("Bitpanda price differs from the global average by ${Fmt.formatPct(diff, 2)} (spread)", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        detail?.bitpandaPrices?.takeIf { it.size > 1 }?.let { prices ->
            Text(prices.entries.joinToString("  ·  ") { "${Fmt.formatPrice(it.value)} ${it.key}" }, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun ChartCard(row: AssetRow, detail: AssetDetail?) {
    var range by remember { mutableIntStateOf(0) }
    val labels = listOf("7D", "90D", "1Y")
    val values = when (range) {
        0 -> (detail?.hourly?.map { it.value } ?: row.market?.sparkline7d).orEmpty()
        1 -> detail?.daily?.takeLast(90)?.map { it.value }.orEmpty()
        else -> detail?.daily?.map { it.value }.orEmpty()
    }
    SectionCard("Price", subtitle = if (range == 0) "Dashed lines: likely 24h range" else null) {
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            labels.forEachIndexed { i, label ->
                SegmentedButton(selected = range == i, onClick = { range = i }, shape = SegmentedButtonDefaults.itemShape(index = i, count = labels.size), label = { Text(label) })
            }
        }
        PriceChart(
            values,
            bandLow = if (range == 0) row.forecast?.h24?.expectedLow else null,
            bandHigh = if (range == 0) row.forecast?.h24?.expectedHigh else null,
        )
        if (range > 0 && detail == null) Text("Loading history…", style = MaterialTheme.typography.labelSmall)
    }
}

/** Why [h] gets no number at all (red flags, unchecked contract, too little history), or null when it has a call. */
internal fun noCallReason(row: AssetRow, h: HorizonForecast): String? = when {
    row.upOrNull(h) != null -> null
    !h.hasCall -> h.signals.firstOrNull()?.description ?: "No forecast"
    else -> "No forecast (${row.risk?.label?.lowercase() ?: "red flags"})"
}

/** [noCall]: see [noCallReason]; the 0.5 such a horizon carries is a placeholder, not a coin flip. */
@Composable
private fun ForecastCard(title: String, h: HorizonForecast, noCall: String?, currency: String) {
    val c = LocalMarketColors.current
    if (noCall != null) {
        SectionCard(title) {
            Row(verticalAlignment = Alignment.Bottom) {
                Text("—", style = MaterialTheme.typography.displaySmall, color = c.neutral, fontWeight = FontWeight.Bold)
                Spacer(Modifier.width(8.dp))
                Column {
                    Text("chance of going up", style = MaterialTheme.typography.bodyMedium)
                    Text("No forecast", style = MaterialTheme.typography.labelLarge, color = c.neutral)
                }
            }
            Text(noCall, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        return
    }
    val color = when (h.direction) {
        Direction.UP -> c.up
        Direction.DOWN -> c.down
        Direction.FLAT -> c.neutral
    }
    val word = when (h.direction) {
        Direction.UP -> "Likely up"
        Direction.DOWN -> "Likely down"
        Direction.FLAT -> "No clear direction"
    }
    SectionCard(title) {
        Row(verticalAlignment = Alignment.Bottom) {
            Text("${(h.probabilityUp * 100).toInt()}%", style = MaterialTheme.typography.displaySmall, color = color, fontWeight = FontWeight.Bold)
            Spacer(Modifier.width(8.dp))
            Column {
                Text("chance of going up", style = MaterialTheme.typography.bodyMedium)
                Text(word, style = MaterialTheme.typography.labelLarge, color = color)
            }
        }
        Text("Confidence ${(h.confidence * 100).toInt()}%", style = MaterialTheme.typography.labelMedium)
        FractionBar(h.confidence, MaterialTheme.colorScheme.primary)
        if (h.expectedLow != null && h.expectedHigh != null) {
            KeyValue("Likely range", "${priceText(h.expectedLow, "")} – ${priceText(h.expectedHigh, currency)}")
        }
        h.expectedMovePct?.let { KeyValue("Expected move", Fmt.formatPct(it, 2)) }
        h.backtest?.let { bt ->
            KeyValue("Backtest", "${(bt.hitRate * 100).toInt()}% right on ~${bt.effectiveSamples} independent past cases (chance ${(bt.baselineHitRate * 100).toInt()}%)")
        }
        if (h.signals.isNotEmpty()) {
            HorizontalDivider()
            Text("What drives it", style = MaterialTheme.typography.labelLarge)
            val maxAbs = h.signals.maxOf { abs(it.contribution) }
            h.signals.take(8).forEach { s ->
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 2.dp)) {
                    ContributionBar(s.contribution, maxAbs)
                    Spacer(Modifier.width(8.dp))
                    Column(Modifier.weight(1f)) {
                        Text(s.label, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
                        Text(s.description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }
    }
}

/** [d7Up]: the 7-day chance of going up, null when that horizon has no call. */
@Composable
private fun BottomCard(b: BottomAssessment, d7Up: Double?) {
    SectionCard("Is this the lowest price?", subtitle = "Nobody knows the absolute bottom. This measures how close the price is to its recent lows and whether rebound conditions are present.") {
        // blocked tokens (red flags, too new, unchecked) get no score: their 50 is a placeholder, the reasons say why
        if (b.label != DefaultForecastEngine.NOT_ASSESSED && b.label != DefaultForecastEngine.NO_HISTORY) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("${b.score}", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
                Text("/100  ", style = MaterialTheme.typography.bodyMedium)
                Text(b.label, style = MaterialTheme.typography.titleSmall)
            }
            BottomMeter(b.score, Modifier.fillMaxWidth(), width = 300.dp, height = 10.dp)
        }
        d7Up?.let { Text("Chance of trading higher in 7 days: ${(it * 100).toInt()}%", style = MaterialTheme.typography.bodyMedium) }
        b.reasons.forEach { Text("• $it", style = MaterialTheme.typography.bodySmall) }
        b.distanceTo7dLowPct?.let { KeyValue("Above 7-day low", "+${"%.1f".format(it)}%") }
        b.distanceTo30dLowPct?.let { KeyValue("Above 30-day low", "+${"%.1f".format(it)}%") }
        b.distanceTo90dLowPct?.let { KeyValue("Above 90-day low", "+${"%.1f".format(it)}%") }
        b.distanceTo1yLowPct?.let { KeyValue("Above 1-year low", "+${"%.1f".format(it)}%") }
        b.drawdownFromAthPct?.let { KeyValue("From all-time high", "${"%.0f".format(it)}%") }
        b.rsi14?.let { KeyValue("RSI (14)", "%.0f".format(it)) }
    }
}

@Composable
private fun SentimentCard(s: SentimentSummary?) {
    val c = LocalMarketColors.current
    val uri = LocalUriHandler.current
    SectionCard("Sentiment", subtitle = "Reddit, StockTwits, Telegram, news and the cloud hub (FinBERT, CryptoBERT, VADER)") {
        if (s == null || s.totalSamples == 0) {
            Text("No posts yet. Open again in a moment; sources are read slowly to stay within free limits.", style = MaterialTheme.typography.bodySmall)
        }
        if (s != null) {
            if (s.totalSamples > 0) {
                val color = if (s.composite >= 0.08) c.up else if (s.composite <= -0.08) c.down else c.neutral
                Text("${s.label} (${"%+.2f".format(s.composite)}) · ${s.totalSamples} items · confidence ${(s.confidence * 100).toInt()}%", color = color, fontWeight = FontWeight.SemiBold)
            }
            s.perSource.forEach { src ->
                HorizontalDivider()
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(sourceName(src.source), style = MaterialTheme.typography.labelLarge, modifier = Modifier.weight(1f))
                    Text(
                        if (src.status == SourceStatus.OK) "${src.sampleCount} · ${"%+.2f".format(src.composite)}" else statusText(src.status),
                        style = MaterialTheme.typography.labelMedium,
                        color = if (src.status == SourceStatus.OK) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                if (src.status == SourceStatus.OK && src.sampleCount > 0) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Column(Modifier.weight(1f)) { Text("bullish ${(src.bullishShare * 100).toInt()}%", style = MaterialTheme.typography.labelSmall); FractionBar(src.bullishShare, c.up) }
                        Column(Modifier.weight(1f)) { Text("bearish ${(src.bearishShare * 100).toInt()}%", style = MaterialTheme.typography.labelSmall); FractionBar(src.bearishShare, c.down) }
                    }
                }
                if (src.modelScores.isNotEmpty()) {
                    Text(src.modelScores.entries.joinToString("  ·  ") { "${modelName(it.key)} ${"%+.2f".format(it.value)}" }, style = MaterialTheme.typography.labelSmall)
                }
                src.note?.let { Text(it, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                src.topSamples.take(3).forEach { t ->
                    Text(
                        "${if (t.score >= 0.1) "🟢" else if (t.score <= -0.1) "🔴" else "⚪"} ${t.text}${t.community?.let { " — $it" } ?: ""}",
                        style = MaterialTheme.typography.bodySmall,
                        maxLines = 3,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.clickable(enabled = t.url != null) { t.url?.let { runCatching { uri.openUri(it) } } },
                    )
                }
            }
        }
    }
}

@Composable
private fun MoreDataCard(d: AssetDetail, currency: String) {
    val m = d.row.market
    val uri = LocalUriHandler.current
    SectionCard("Market and crowd data") {
        KeyValue("Market cap", capText(m?.marketCap, currency))
        KeyValue("Volume 24h", capText(m?.volume24h, currency))
        m?.marketCapRank?.let { KeyValue("Global rank", "#$it") }
        m?.ath?.let { KeyValue("All-time high", priceText(it, currency)) }
        m?.atl?.let { KeyValue("All-time low", priceText(it, currency)) }
        d.derivatives?.fundingRate?.let { KeyValue("Futures funding (8h)", "%.4f%%".format(it * 100)) }
        d.derivatives?.longShortRatio?.let { KeyValue("Long/short accounts", "%.2f".format(it)) }
        d.community?.sentimentVotesUpPct?.let { KeyValue("CoinGecko votes up", "%.0f%%".format(it)) }
        d.community?.watchlistUsers?.let { KeyValue("Watchlists", Fmt.formatCompact(it.toDouble())) }
        d.community?.telegramUsers?.let { KeyValue("Telegram members", Fmt.formatCompact(it.toDouble())) }
        d.community?.description?.let { Text(it.take(400) + if (it.length > 400) "…" else "", style = MaterialTheme.typography.bodySmall) }
        d.community?.homepage?.let { url -> Text(url, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary, modifier = Modifier.clickable { runCatching { uri.openUri(url) } }) }
    }
}

private fun sourceName(s: SentimentSource) = when (s) {
    SentimentSource.REDDIT -> "Reddit"
    SentimentSource.STOCKTWITS -> "StockTwits"
    SentimentSource.TELEGRAM -> "Telegram (public channels)"
    SentimentSource.NEWS -> "News"
    SentimentSource.FEAR_GREED -> "Fear & Greed"
    SentimentSource.COINGECKO_VOTES -> "CoinGecko votes"
    SentimentSource.DERIVATIVES -> "Futures"
    SentimentSource.SENTIMENT_HUB -> "Cloud hub (Telegram + AI models)"
}

private fun modelName(key: String) = when (key.lowercase()) {
    "vader" -> "VADER"
    "finbert" -> "FinBERT"
    "cryptobert" -> "CryptoBERT"
    else -> key
}

private fun statusText(s: SourceStatus) = when (s) {
    SourceStatus.OK -> "ok"
    SourceStatus.EMPTY -> "no mentions"
    SourceStatus.RATE_LIMITED -> "rate limited, retry later"
    SourceStatus.UNAUTHORIZED -> "blocked"
    SourceStatus.UNAVAILABLE -> "unreachable"
    SourceStatus.GEO_BLOCKED -> "not available in your region"
    SourceStatus.PARSE_ERROR -> "format changed"
    SourceStatus.DISABLED -> "off"
}
