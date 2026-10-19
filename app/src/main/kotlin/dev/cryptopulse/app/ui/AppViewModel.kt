package dev.cryptopulse.app.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.cryptopulse.app.CryptoPulseApp
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.model.AssetDetail
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.MarketSnapshot
import dev.cryptopulse.core.model.RefreshStatus
import dev.cryptopulse.core.model.SourceStatus
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

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
    NAME("Name"),
}

sealed interface Screen {
    data object Market : Screen
    data class Detail(val symbol: String) : Screen
    data object Settings : Screen
}

data class DetailState(val symbol: String, val loading: Boolean = true, val detail: AssetDetail? = null, val error: String? = null)

data class ListState(
    val rows: List<AssetRow> = emptyList(),
    val totalRows: Int = 0,
    val query: String = "",
    val filter: ListFilter = ListFilter.ALL,
    val sort: ListSort = ListSort.MARKET_CAP,
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

    private val _detail = MutableStateFlow<DetailState?>(null)
    val detail: StateFlow<DetailState?> = _detail.asStateFlow()

    private val _config = MutableStateFlow(cp.config)
    val config: StateFlow<AppConfig> = _config.asStateFlow()

    private val _enrichProgress = MutableStateFlow<Pair<Int, Int>?>(null)
    /** (done, total) while per-coin sentiment is being collected in the background. */
    val enrichProgress: StateFlow<Pair<Int, Int>?> = _enrichProgress.asStateFlow()

    private val _hub = MutableStateFlow(HubState(SourceStatus.DISABLED, null))
    val hub: StateFlow<HubState> = _hub.asStateFlow()

    val list: StateFlow<ListState> = combine(repo.snapshot, _query, _filter, _sort) { snap, q, f, s ->
        val all = snap?.rows.orEmpty()
        ListState(rows = applyListOptions(all, q, f, s), totalRows = all.size, query = q, filter = f, sort = s)
    }.stateIn(viewModelScope, SharingStarted.Eagerly, ListState())

    private var refreshJob: Job? = null
    private var enrichJob: Job? = null
    private var detailJob: Job? = null

    init {
        refresh(force = false)
    }

    fun setQuery(q: String) { _query.value = q }
    fun setFilter(f: ListFilter) { _filter.value = f }
    fun setSort(s: ListSort) { _sort.value = s }

    fun refreshIfStale() {
        if (refreshJob?.isActive == true) return
        if (repo.isStale()) refresh(force = false)
    }

    fun refresh(force: Boolean = true) {
        if (refreshJob?.isActive == true) return
        refreshJob = viewModelScope.launch(Dispatchers.Default) {
            try {
                repo.refresh(force)
                _hub.value = cp.sentiment.feedState().let { HubState(it.first, it.second) }
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
    }

    fun back(): Boolean = when (_screen.value) {
        Screen.Market -> false
        else -> { _screen.value = Screen.Market; true }
    }

    fun loadDetail(symbol: String, force: Boolean) {
        detailJob?.cancel()
        val previous = _detail.value?.takeIf { it.symbol == symbol }?.detail
        _detail.value = DetailState(symbol, loading = true, detail = previous)
        detailJob = viewModelScope.launch(Dispatchers.Default) {
            _detail.value = try {
                DetailState(symbol, loading = false, detail = repo.detail(symbol, force))
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                DetailState(symbol, loading = false, detail = previous, error = e.message ?: "Could not load $symbol")
            }
        }
    }

    fun updateConfig(transform: (AppConfig) -> AppConfig) {
        val old = _config.value
        val new = transform(old)
        if (new == old) return
        _config.value = new
        app.settings.save(new)
        cp.updateConfig(new)
        if (new.currency != old.currency || new.sentimentFeedUrl != old.sentimentFeedUrl) refresh(force = true)
    }

    fun telegramChannelCount(): Pair<Int, Int> {
        val dir = cp.sentiment.telegramDirectory
        return cp.sentiment.channelsToScan(_config.value).size to dir.entries.size
    }

    companion object {
        fun applyListOptions(rows: List<AssetRow>, query: String, filter: ListFilter, sort: ListSort): List<AssetRow> {
            val q = query.trim().lowercase()
            val filtered = rows.asSequence()
                .filter { q.isEmpty() || it.asset.symbol.lowercase().contains(q) || it.asset.name.lowercase().contains(q) }
                .filter { r ->
                    val f = r.forecast
                    when (filter) {
                        ListFilter.ALL -> true
                        ListFilter.UP_24H -> (f?.h24?.probabilityUp ?: 0.0) >= 0.55
                        ListFilter.UP_7D -> (f?.d7?.probabilityUp ?: 0.0) >= 0.55
                        ListFilter.NEAR_BOTTOM -> (f?.bottom?.score ?: 0) >= 60 && f?.bottom?.label != "Pegged"
                        ListFilter.GAINERS -> (r.market?.change24hPct ?: 0.0) > 0
                        ListFilter.LOSERS -> (r.market?.change24hPct ?: 0.0) < 0
                        ListFilter.BULLISH_SOCIAL -> (r.sentiment?.composite ?: 0.0) >= 0.08 && (r.sentiment?.totalSamples ?: 0) > 0
                    }
                }.toList()
            return when (sort) {
                ListSort.MARKET_CAP -> filtered
                ListSort.CHANGE_24H -> filtered.sortedByDescending { it.market?.change24hPct ?: Double.NEGATIVE_INFINITY }
                ListSort.CHANGE_7D -> filtered.sortedByDescending { it.market?.change7dPct ?: Double.NEGATIVE_INFINITY }
                ListSort.UP_24H -> filtered.sortedByDescending { it.forecast?.h24?.probabilityUp ?: 0.0 }
                ListSort.UP_7D -> filtered.sortedByDescending { it.forecast?.d7?.probabilityUp ?: 0.0 }
                ListSort.BOTTOM -> filtered.sortedByDescending { it.forecast?.bottom?.score ?: -1 }
                ListSort.SENTIMENT -> filtered.sortedByDescending { it.sentiment?.composite ?: -2.0 }
                ListSort.NAME -> filtered.sortedBy { it.asset.name.lowercase() }
            }
        }
    }
}
