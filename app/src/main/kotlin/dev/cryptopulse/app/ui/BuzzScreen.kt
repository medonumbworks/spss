package dev.cryptopulse.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.cryptopulse.app.ui.theme.LocalMarketColors
import dev.cryptopulse.core.sources.feed.BuzzEntry

/** Most-talked-about coins in the cloud hub's Telegram window, loudest spikes first. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BuzzScreen(vm: AppViewModel) {
    val buzz by vm.buzz.collectAsStateWithLifecycle()
    val snapshot by vm.snapshot.collectAsStateWithLifecycle()
    // tapping a symbol opens the detail screen, but only for coins the snapshot knows
    val known = remember(snapshot) { snapshot?.rows?.map { it.asset.symbol }?.toSet() ?: emptySet() }

    Scaffold(
        topBar = {
            TopAppBar(
                navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } },
                title = {
                    Column {
                        Text("Telegram buzz", fontWeight = FontWeight.SemiBold)
                        Text(
                            "Mentions across public channels, from the cloud hub",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
                actions = { IconButton(onClick = { vm.loadBuzz(force = true) }) { Icon(Icons.Filled.Refresh, contentDescription = "Reload") } },
            )
        },
    ) { padding ->
        val entries = buzz
        when {
            entries == null -> Column(Modifier.padding(padding).fillMaxWidth()) {
                LinearProgressIndicator(Modifier.fillMaxWidth())
                Text(
                    "Loading mention data…",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
                )
            }
            entries.isEmpty() -> Text(
                "The cloud hub has not published mention data yet.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(padding).padding(24.dp),
            )
            else -> LazyColumn(Modifier.padding(padding).fillMaxSize(), contentPadding = PaddingValues(bottom = 24.dp)) {
                items(entries, key = { it.symbol }) { e ->
                    BuzzRow(e, known = e.symbol in known) { vm.open(Screen.Detail(e.symbol)) }
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
                }
            }
        }
    }
}

@Composable
private fun BuzzRow(e: BuzzEntry, known: Boolean, onOpen: () -> Unit) {
    val c = LocalMarketColors.current
    val uri = LocalUriHandler.current
    Column(
        Modifier.fillMaxWidth().clickable(enabled = known, onClick = onOpen).padding(horizontal = 16.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(e.symbol, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
            // same colouring as the other sentiment numbers (SentimentCard, SentimentDot)
            val color = if (e.composite >= 0.08) c.up else if (e.composite <= -0.08) c.down else c.neutral
            Text("%+.2f".format(e.composite), style = MaterialTheme.typography.labelMedium, color = color, fontWeight = FontWeight.SemiBold)
        }
        Text(
            "${e.mentions} mentions · ${buzzTrendLabel(e.spike, e.mentionsPrev)}",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        e.samples.take(3).forEach { s ->
            Text(
                "${if (s.score >= 0.1) "🟢" else if (s.score <= -0.1) "🔴" else "⚪"} ${s.text}${s.community?.let { " — $it" } ?: ""}",
                style = MaterialTheme.typography.bodySmall,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.clickable(enabled = s.url != null) { s.url?.let { runCatching { uri.openUri(it) } } },
            )
        }
    }
}
