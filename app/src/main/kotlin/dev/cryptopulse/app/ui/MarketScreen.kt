package dev.cryptopulse.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.cryptopulse.app.ui.components.BottomMeter
import dev.cryptopulse.app.ui.components.ChangePill
import dev.cryptopulse.app.ui.components.CoinIcon
import dev.cryptopulse.app.ui.components.ProbabilityChip
import dev.cryptopulse.app.ui.components.RiskPill
import dev.cryptopulse.app.ui.components.SentimentDot
import dev.cryptopulse.app.ui.components.Sparkline
import dev.cryptopulse.app.ui.components.TierBadge
import dev.cryptopulse.app.ui.components.capText
import dev.cryptopulse.app.ui.components.priceText
import dev.cryptopulse.app.ui.theme.LocalMarketColors
import dev.cryptopulse.core.analysis.forecast.DefaultForecastEngine
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.ListingSource
import dev.cryptopulse.core.model.MarketSnapshot
import dev.cryptopulse.core.model.RefreshStatus
import dev.cryptopulse.core.model.SourceStatus

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MarketScreen(vm: AppViewModel) {
    val snapshot by vm.snapshot.collectAsStateWithLifecycle()
    val status by vm.status.collectAsStateWithLifecycle()
    val list by vm.list.collectAsStateWithLifecycle()
    val enrich by vm.enrichProgress.collectAsStateWithLifecycle()
    val hub by vm.hub.collectAsStateWithLifecycle()
    val refreshing = status is RefreshStatus.Refreshing

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("CryptoPulse", fontWeight = FontWeight.Bold)
                        Text(
                            snapshot?.let { s ->
                                val extra = s.rows.count { !it.asset.onBitpanda }
                                "${s.universeSize} Bitpanda coins" + (if (extra > 0) " + $extra speculative" else "") + " · ${s.currency}"
                            } ?: "Bitpanda crypto overview",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
                actions = {
                    IconButton(onClick = { vm.refresh(force = true) }) { Icon(Icons.Filled.Refresh, contentDescription = "Refresh") }
                    IconButton(onClick = { vm.open(Screen.Settings) }) { Icon(Icons.Filled.Settings, contentDescription = "Settings") }
                },
            )
        },
    ) { padding ->
        PullToRefreshBox(isRefreshing = refreshing, onRefresh = { vm.refresh(force = true) }, modifier = Modifier.padding(padding).fillMaxSize()) {
            LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(bottom = 24.dp)) {
                item { StatusBanner(status, enrich, snapshot) }
                item { MoodCard(snapshot, hub) }
                item { SearchAndFilters(list, vm) }
                if (list.tier != ListTier.BITPANDA && list.rows.any { !it.asset.onBitpanda }) item { SpeculativeNotice() }
                if (snapshot == null && !refreshing) {
                    item {
                        Text(
                            "No data yet. Pull down to load prices.",
                            modifier = Modifier.padding(24.dp),
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                items(list.rows, key = { it.asset.symbol }) { row ->
                    AssetRowItem(row, snapshot?.currency ?: "EUR") { vm.open(Screen.Detail(row.asset.symbol)) }
                    HorizontalDivider(Modifier.padding(start = 72.dp), color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
                }
                item {
                    Text(
                        Forecast.DISCLAIMER,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(16.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun StatusBanner(status: RefreshStatus, enrich: Pair<Int, Int>?, snapshot: MarketSnapshot?) {
    when (status) {
        is RefreshStatus.Refreshing -> Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp)) {
            LinearProgressIndicator(progress = { status.progress }, modifier = Modifier.fillMaxWidth())
            Text(status.step, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        is RefreshStatus.Error -> Row(Modifier.fillMaxWidth().padding(16.dp, 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.Warning, contentDescription = null, tint = LocalMarketColors.current.warn, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(6.dp))
            Text(
                (if (snapshot != null) "Offline, showing saved data. " else "") + status.message,
                style = MaterialTheme.typography.labelMedium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        RefreshStatus.Idle -> if (enrich != null) {
            Text(
                "Reading Reddit, StockTwits, Telegram and news… ${enrich.first}/${enrich.second}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 2.dp),
            )
        }
    }
}

@Composable
private fun MoodCard(snapshot: MarketSnapshot?, hub: HubState) {
    val ctx = snapshot?.context ?: return
    val c = LocalMarketColors.current
    Surface(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                ctx.fearGreed?.let { fg ->
                    val color = when {
                        fg.value <= 25 -> c.down
                        fg.value >= 75 -> c.up
                        else -> c.warn
                    }
                    Text("Fear & Greed ", style = MaterialTheme.typography.labelLarge)
                    Text("${fg.value} ${fg.classification}", style = MaterialTheme.typography.labelLarge, color = color, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.width(12.dp))
                }
                ctx.global?.marketCapChange24hPct?.let {
                    Text("Market 24h ", style = MaterialTheme.typography.labelLarge)
                    ChangePill(it, 1)
                }
            }
            ctx.globalSentiment?.takeIf { it.totalSamples > 0 }?.let {
                Text("Social mood: ${it.label} · ${it.totalSamples} posts and headlines", style = MaterialTheme.typography.labelMedium)
            }
            val source = when (snapshot.universeSource) {
                ListingSource.BITPANDA_LEGACY_TICKER -> "live Bitpanda ticker"
                ListingSource.BITPANDA_PUBLIC_API -> "live Bitpanda API"
                ListingSource.BUNDLED_SNAPSHOT -> "built-in Bitpanda list"
            }
            val hubText = when (hub.status) {
                SourceStatus.OK -> "cloud sentiment hub connected"
                SourceStatus.DISABLED -> "cloud hub off"
                else -> "cloud hub: ${hub.message ?: hub.status.name.lowercase()}"
            }
            Text("Coins from $source · $hubText", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun SearchAndFilters(list: ListState, vm: AppViewModel) {
    var sortOpen by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp)) {
        OutlinedTextField(
            value = list.query,
            onValueChange = vm::setQuery,
            modifier = Modifier.fillMaxWidth(),
            placeholder = { Text("Search coins") },
            leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null) },
            trailingIcon = {
                if (list.query.isNotEmpty()) IconButton(onClick = { vm.setQuery("") }) { Icon(Icons.Filled.Clear, contentDescription = "Clear") }
            },
            singleLine = true,
            shape = MaterialTheme.shapes.large,
        )
        Row(Modifier.horizontalScroll(rememberScrollState()).padding(top = 6.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ListOptions.visibleTiers(list.tierCounts, list.tier).forEach { t ->
                val n = list.tierCounts[t] ?: 0
                FilterChip(selected = list.tier == t, onClick = { vm.setTier(t) }, label = { Text(if (t == ListTier.ALL) t.label else "${t.label} $n") })
            }
        }
        Row(Modifier.horizontalScroll(rememberScrollState()).padding(top = 2.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ListFilter.entries.forEach { f ->
                FilterChip(selected = list.filter == f, onClick = { vm.setFilter(f) }, label = { Text(f.label) })
            }
        }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("${list.rows.size} of ${list.totalRows}", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.weight(1f))
            Box {
                TextButton(onClick = { sortOpen = true }) {
                    Text("Sort: ${list.sort.label}")
                    Icon(Icons.Filled.ArrowDropDown, contentDescription = null)
                }
                DropdownMenu(expanded = sortOpen, onDismissRequest = { sortOpen = false }) {
                    ListSort.entries.forEach { s ->
                        DropdownMenuItem(text = { Text(s.label) }, onClick = { vm.setSort(s); sortOpen = false })
                    }
                }
            }
        }
    }
}

@Composable
private fun SpeculativeNotice() {
    Surface(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
        shape = MaterialTheme.shapes.medium,
        color = LocalMarketColors.current.down.copy(alpha = 0.08f),
    ) {
        Text(
            "Coins marked MEME, TRENDING or DEX are not on Bitpanda and are shown for information only. Most memecoins and DEX tokens lose most of " +
                "their value; tokens with red flags, pools younger than a day or unchecked contracts get no forecast.",
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.padding(10.dp),
        )
    }
}

@Composable
private fun AssetRowItem(row: AssetRow, currency: String, onClick: () -> Unit) {
    val m = row.market
    val f = row.forecast
    Row(
        Modifier.fillMaxWidth().clickable(onClick = onClick).padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        CoinIcon(row.asset.symbol, m?.imageUrl)
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(row.asset.name, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
                Spacer(Modifier.width(6.dp))
                TierBadge(row.asset.tier)
                if (!row.asset.onBitpanda) Spacer(Modifier.width(6.dp))
                SentimentDot(row.sentiment?.composite, row.sentiment?.totalSamples)
            }
            val dexInfo = row.dex?.let { d -> " · ${d.chainId} · liq ${capText(d.liquidityUsd, "USD")}" } ?: ""
            Text(
                "${row.asset.symbol} · Cap ${capText(m?.marketCap, m?.currency ?: currency)}$dexInfo",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
            )
            Row(Modifier.padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                ProbabilityChip("24h", f?.h24)
                ProbabilityChip("7d", f?.d7)
                RiskPill(row.risk)
                if (f != null && f.bottom.label != "Pegged" && f.bottom.label != DefaultForecastEngine.NOT_ASSESSED && row.risk?.blocked != true) BottomMeter(f.bottom.score, width = 40.dp)
            }
        }
        Spacer(Modifier.width(8.dp))
        if (m != null && m.sparkline7d.size > 2) {
            Sparkline(m.sparkline7d, Modifier.width(48.dp).height(28.dp))
            Spacer(Modifier.width(8.dp))
        }
        Column(horizontalAlignment = Alignment.End) {
            // a DEX price stays in USD when there is no USD rate for the list currency; say so
            Text(priceText(m?.bitpandaPrice ?: m?.price, m?.currency?.takeIf { it != currency } ?: ""), style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Medium, maxLines = 1)
            Spacer(Modifier.height(2.dp))
            ChangePill(m?.change24hPct)
        }
    }
}
