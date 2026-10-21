package dev.cryptopulse.app.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.cryptopulse.app.CryptoPulseApp
import dev.cryptopulse.app.work.AlertScheduler
import dev.cryptopulse.core.analysis.journal.JournalStats
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.model.AssetDetail
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.MarketSnapshot
import dev.cryptopulse.core.model.RefreshStatus
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.feed.BuzzEntry
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** Which part of the list: Bitpanda coins, or the speculative tiers beyond Bitpanda. */
enum class ListTier(val label: String) {
    ALL("All"),
    BITPANDA("Bitpanda"),
    TRENDING("Trending"),
}

enum class ListFilter(val label: String) {
    ALL("All"),
    UP_24H("Likely up 24h"),
    UP_7D("Likely up 7d"),
    NEAR_BOTTOM("Near a low"),
    GAINERS("Gainers 24h"),
    LOSERS("Losers 24h"),
    BULLISH_SOCIAL("Bullish chatter"),
}

enum class ListSort(val label: String) {
    MARKET_CAP("Market cap"),
    CHANGE_24H("24h change"),
    CHANGE_7D("7d change"),
    UP_24H("Chance up 24h"),
    UP_7D("Chance up 7d"),
    BOTTOM("Bottom score"),
    SENTIMENT("Sentiment"),
    VOLUME("Volume 24h"),
    RISK("Lowest risk"),
    NAME("Name"),
}

sealed interface Screen {
    data object Market : Screen
    data class Detail(val symbol: String) : Screen
    data object Settings : Screen
    data object Buzz : Screen
}

data class DetailState(val symbol: String, val loading: Boolean = true, val detail: AssetDetail? = null, val error: String? = null)

data class ListState(
    val rows: List<AssetRow> = emptyList(),
    val totalRows: Int = 0,
    val query: String = "",
    val filter: ListFilter = ListFilter.ALL,
    val sort: ListSort = ListSort.MARKET_CAP,
    val tier: ListTier = ListTier.ALL,
    /** Rows per tier in the whole snapshot, for the tier chips. */
    val tierCounts: Map<ListTier, Int> = emptyMap(),
)

data class HubState(val status: SourceStatus, val message: String?)

class AppViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as CryptoPulseApp
    private val cp = app.cryptoPulse
    private val repo = cp.repository

    val snapshot: StateFlow<MarketSnapshot?> = repo.snapshot
    val status: StateFlow<RefreshStatus> = repo.status

    private val _screen = MutableStateFlow<Screen>(Screen.Market)
    val screen: StateFlow<Screen> = _screen.asStateFlow()

    private val _query = MutableStateFlow("")
    private val _filter = MutableStateFlow(ListFilter.ALL)
    private val _sort = MutableStateFlow(ListSort.MARKET_CAP)
    private val _tier = MutableStateFlow(ListTier.ALL)

    private val _detail = MutableStateFlow<DetailState?>(null)
    val detail: StateFlow<DetailState?> = _detail.asStateFlow()

    private val _config = MutableStateFlow(cp.config)
    val config: StateFlow<AppConfig> = _config.asStateFlow()

    private val _enrichProgress = MutableStateFlow<Pair<Int, Int>?>(null)
    /** (done, total) while per-coin sentiment is being collected in the background. */
    val enrichProgress: StateFlow<Pair<Int, Int>?> = _enrichProgress.asStateFlow()

    private val _hub = MutableStateFlow(HubState(SourceStatus.DISABLED, null))
    val hub: StateFlow<HubState> = _hub.asStateFlow()

    /** Telegram buzz ranking; null until the first load, empty when the hub has no mention data. */
    private val _buzz = MutableStateFlow<List<BuzzEntry>?>(null)
    val buzz: StateFlow<List<BuzzEntry>?> = _buzz.asStateFlow()

    val list: StateFlow<ListState> = combine(repo.snapshot, _query, _filter, _sort, _tier) { snap, q, f, s, t ->
        val all = snap?.rows.orEmpty()
        val counts = ListTier.entries.associateWith { tier -> all.count { ListOptions.inTier(it, tier) } }
        ListState(rows = ListOptions.apply(all, q, f, s, t), totalRows = all.size, query = q, filter = f, sort = s, tier = t, tierCounts = counts)
    }.stateIn(viewModelScope, SharingStarted.Eagerly, ListState())

    private var refreshJob: Job? = null
    private var enrichJob: Job? = null
    private var detailJob: Job? = null
    private var buzzJob: Job? = null

    init {
        refresh(force = false)
    }

    fun setQuery(q: String) { _query.value = q }
    fun setFilter(f: ListFilter) { _filter.value = f }
    fun setSort(s: ListSort) { _sort.value = s }
    fun setTier(t: ListTier) { _tier.value = t }

    fun refreshIfStale() {
        if (refreshJob?.isActive == true) return
        if (repo.isStale()) refresh(force = false)
    }

    /** [restart]: cancel a running refresh first (it was started with the previous settings). */
    fun refresh(force: Boolean = true, restart: Boolean = false) {
        val previous = refreshJob
        if (!restart && previous?.isActive == true) return
        refreshJob = viewModelScope.launch(Dispatchers.Default) {
            previous?.cancelAndJoin()
            try {
                repo.refresh(force)
                _hub.value = cp.sentiment.feedState().let { HubState(it.first, it.second) }
                // keep the buzz tab in step with the refresh once it was opened at least once
                if (_buzz.value != null || _screen.value == Screen.Buzz) loadBuzz()
                startEnrichment()
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
                // status flow already carries the error for the UI
            }
        }
    }

    private fun startEnrichment() {
        if (enrichJob?.isActive == true) return
        enrichJob = viewModelScope.launch(Dispatchers.Default) {
            try {
                repo.enrichSentiment { done, total -> _enrichProgress.value = done to total }
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
            } finally {
                _enrichProgress.value = null
            }
        }
    }

    fun open(screen: Screen) {
        _screen.value = screen
        if (screen is Screen.Detail) loadDetail(screen.symbol, force = false)
        if (screen is Screen.Buzz) loadBuzz()
    }

    fun loadBuzz(force: Boolean = false) {
        if (buzzJob?.isActive == true) return
        buzzJob = viewModelScope.launch(Dispatchers.Default) {
            try {
                _buzz.value = repo.telegramBuzz(force = force)
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
                // hub unreachable: show the empty state rather than loading forever, keep old data if any
                if (_buzz.value == null) _buzz.value = emptyList()
            }
        }
    }

    /** Realised accuracy of every forecast the engine issued (the forecast journal). */
    fun journalStats(): JournalStats = repo.journalStats()

    fun back(): Boolean = when (_screen.value) {
        Screen.Market -> false
        else -> { _screen.value = Screen.Market; true }
    }

    fun loadDetail(symbol: String, force: Boolean) {
        detailJob?.cancel()
        val previous = _detail.value?.takeIf { it.symbol == symbol }?.detail
        _detail.value = DetailState(symbol, loading = true, detail = previous)
        detailJob = viewModelScope.launch(Dispatchers.Default) {
            val next = try {
                DetailState(symbol, loading = false, detail = repo.detail(symbol, force))
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                DetailState(symbol, loading = false, detail = previous, error = e.message ?: "Could not load $symbol")
            }
            // cancel() cannot stop a load that already returned; only write state for the coin still shown
            ensureActive()
            _detail.update { if (it?.symbol == symbol) next else it }
        }
    }

    fun updateConfig(transform: (AppConfig) -> AppConfig) {
        val old = _config.value
        val new = transform(old)
        if (new == old) return
        _config.value = new
        app.settings.save(new)
        cp.updateConfig(new)
        if (new.alertsEnabled != old.alertsEnabled || new.alertThresholdPct != old.alertThresholdPct) AlertScheduler.sync(app, new)
        if (needsRefresh(old, new)) refresh(force = true, restart = true)
    }

    fun telegramChannelCount(): Pair<Int, Int> {
        val dir = cp.sentiment.telegramDirectory
        return cp.sentiment.channelsToScan(_config.value).size to dir.entries.size
    }
}

/** True when a settings change alters which coins are listed or how they are priced, so the list is rebuilt at once. */
internal fun needsRefresh(old: AppConfig, new: AppConfig): Boolean {
    val tiersChanged = new.showMemecoins != old.showMemecoins || new.showTrending != old.showTrending || new.showDexRadar != old.showDexRadar ||
        new.memecoinCount != old.memecoinCount || new.dexRadarMax != old.dexRadarMax ||
        new.dexMinLiquidityUsd != old.dexMinLiquidityUsd || new.dexMinVolume24hUsd != old.dexMinVolume24hUsd ||
        new.dexIncludeNewPools != old.dexIncludeNewPools || new.dexIncludePumpFun != old.dexIncludePumpFun || new.safetyChecks != old.safetyChecks
    return new.currency != old.currency || new.sentimentFeedUrl != old.sentimentFeedUrl || tiersChanged
}
