package dev.cryptopulse.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.cryptopulse.app.BuildConfig
import dev.cryptopulse.app.ui.components.SectionCard
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.model.Forecast

private const val HUB_DOCS = "https://github.com/medonumbworks/spss/blob/main/tools/sentiment_hub/README.md"

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(vm: AppViewModel) {
    val cfg by vm.config.collectAsStateWithLifecycle()
    val uri = LocalUriHandler.current
    Scaffold(
        topBar = {
            TopAppBar(
                navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } },
                title = { Text("Settings") },
            )
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState())) {
            SectionCard("Currency") {
                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                    AppConfig.SUPPORTED_CURRENCIES.forEachIndexed { i, cur ->
                        SegmentedButton(
                            selected = cfg.currency == cur,
                            onClick = { vm.updateConfig { it.copy(currency = cur) } },
                            shape = SegmentedButtonDefaults.itemShape(index = i, count = AppConfig.SUPPORTED_CURRENCIES.size),
                            label = { Text(cur) },
                        )
                    }
                }
            }

            SectionCard("Sentiment sources", subtitle = "All free. Turn off what you do not want; forecasts adapt to the data that is available.") {
                ToggleRow("Reddit", cfg.redditEnabled) { v -> vm.updateConfig { it.copy(redditEnabled = v) } }
                ToggleRow("StockTwits", cfg.stocktwitsEnabled) { v -> vm.updateConfig { it.copy(stocktwitsEnabled = v) } }
                ToggleRow("Telegram channels", cfg.telegramEnabled) { v -> vm.updateConfig { it.copy(telegramEnabled = v) } }
                ToggleRow("News (Google News, CoinTelegraph, CoinDesk, Decrypt)", cfg.newsEnabled) { v -> vm.updateConfig { it.copy(newsEnabled = v) } }
                ToggleRow("Futures positioning (Binance/Bybit)", cfg.derivativesEnabled) { v -> vm.updateConfig { it.copy(derivativesEnabled = v) } }
                ToggleRow("Fear & Greed index", cfg.fearGreedEnabled) { v -> vm.updateConfig { it.copy(fearGreedEnabled = v) } }
            }

            TelegramCard(vm, cfg) { runCatching { uri.openUri(HUB_DOCS) } }

            SectionCard("Background reading", subtitle = "How many of the largest coins get Reddit/StockTwits/Telegram/news sentiment after each refresh") {
                var topN by remember(cfg.backgroundSentimentTopN) { mutableFloatStateOf(cfg.backgroundSentimentTopN.toFloat()) }
                Text("${topN.toInt()} coins")
                Slider(value = topN, onValueChange = { topN = it }, valueRange = 0f..100f, steps = 19, onValueChangeFinished = { vm.updateConfig { it.copy(backgroundSentimentTopN = topN.toInt()) } })
            }

            SectionCard("Optional keys", subtitle = "Not needed. They only raise free rate limits.") {
                KeyField("CoinGecko demo key (free at coingecko.com/api)", cfg.coingeckoDemoKey.orEmpty()) { v -> vm.updateConfig { it.copy(coingeckoDemoKey = v.ifBlank { null }) } }
                KeyField("Bitpanda API key (only if the public price list stops working)", cfg.bitpandaApiKey.orEmpty()) { v -> vm.updateConfig { it.copy(bitpandaApiKey = v.ifBlank { null }) } }
            }

            SectionCard("About") {
                Text("CryptoPulse ${BuildConfig.VERSION_NAME}", fontWeight = FontWeight.SemiBold)
                Text("Prices: Bitpanda (ticker), CoinGecko. Mood: alternative.me Fear & Greed. Crowd: Reddit, StockTwits, Telegram, news, Binance/Bybit futures. Text sentiment: VADER with a crypto slang lexicon on the phone; FinBERT and CryptoBERT in the cloud hub.", style = MaterialTheme.typography.bodySmall)
                Text(Forecast.DISCLAIMER, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun TelegramCard(vm: AppViewModel, cfg: AppConfig, openDocs: () -> Unit) {
    val (scanned, total) = vm.telegramChannelCount()
    SectionCard("Telegram", subtitle = "Three levels, all free") {
        Text("1. Public channels on this phone: $scanned of $total directory entries (no login).", style = MaterialTheme.typography.bodyMedium)
        ToggleRow("Scan every public channel on the phone (more data use)", cfg.telegramScanAll) { v -> vm.updateConfig { it.copy(telegramScanAll = v) } }
        if (!cfg.telegramScanAll) {
            var max by remember(cfg.telegramMaxChannels) { mutableFloatStateOf(cfg.telegramMaxChannels.toFloat()) }
            Text("Channels scanned on the phone: ${max.toInt()}", style = MaterialTheme.typography.labelMedium)
            Slider(value = max, onValueChange = { max = it }, valueRange = 5f..150f, onValueChangeFinished = { vm.updateConfig { it.copy(telegramMaxChannels = max.toInt()) } })
        }
        Text("2. Cloud sentiment hub: a free hourly GitHub Actions job reads every channel in the directory, scores messages with FinBERT and CryptoBERT, and publishes one small file this app downloads.", style = MaterialTheme.typography.bodyMedium)
        KeyField("Hub feed URL (empty = off)", cfg.sentimentFeedUrl) { v -> vm.updateConfig { it.copy(sentimentFeedUrl = v.trim()) } }
        Text("3. Your own Telegram account: add three GitHub secrets once and the hub also reads every group and channel you are in (including private ones). Only totals per coin are published, never message texts or chat names from private chats.", style = MaterialTheme.typography.bodyMedium)
        OutlinedButton(onClick = openDocs) { Text("How to connect your Telegram account") }
    }
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
private fun KeyField(label: String, value: String, onSave: (String) -> Unit) {
    var text by remember(value) { mutableStateOf(value) }
    OutlinedTextField(
        value = text,
        onValueChange = { text = it },
        label = { Text(label) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
        trailingIcon = { if (text != value) androidx.compose.material3.TextButton(onClick = { onSave(text) }) { Text("Save") } },
    )
}
