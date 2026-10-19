package dev.cryptopulse.mcp

import dev.cryptopulse.core.CryptoPulse
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.MarketSnapshot
import dev.cryptopulse.core.model.RiskAssessment
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.repo.SentimentService
import dev.cryptopulse.core.util.Text
import io.modelcontextprotocol.kotlin.sdk.server.Server
import io.modelcontextprotocol.kotlin.sdk.server.ServerOptions
import io.modelcontextprotocol.kotlin.sdk.types.CallToolResult
import io.modelcontextprotocol.kotlin.sdk.types.GetPromptResult
import io.modelcontextprotocol.kotlin.sdk.types.Implementation
import io.modelcontextprotocol.kotlin.sdk.types.PromptArgument
import io.modelcontextprotocol.kotlin.sdk.types.PromptMessage
import io.modelcontextprotocol.kotlin.sdk.types.Role
import io.modelcontextprotocol.kotlin.sdk.types.ServerCapabilities
import io.modelcontextprotocol.kotlin.sdk.types.TextContent
import io.modelcontextprotocol.kotlin.sdk.types.ToolAnnotations
import io.modelcontextprotocol.kotlin.sdk.types.ToolSchema
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.serialization.KSerializer
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject

/** Registers every CryptoPulse tool on an MCP [Server]. */
class CryptoPulseTools(
    private val app: CryptoPulse,
    private val background: CoroutineScope,
) {
    @Volatile
    private var enrichJob: Job? = null

    fun createServer(): Server {
        val server = Server(
            Implementation(name = "cryptopulse", version = VERSION),
            ServerOptions(capabilities = ServerCapabilities(tools = ServerCapabilities.Tools(listChanged = false), prompts = ServerCapabilities.Prompts(listChanged = false))),
        )
        register(server)
        return server
    }

    private fun register(server: Server) {
        val readOnly = ToolAnnotations(readOnlyHint = true, openWorldHint = true)

        server.addTool(
            name = "market_overview",
            title = "Crypto overview (Bitpanda + memecoins + DEX radar)",
            description = "Every cryptocurrency tradeable on Bitpanda, plus (for information only) the largest memecoins, today's trending coins and speculative DEX-only tokens, " +
                "with live price, 24h/7d change, the probability of going up in the next 24 hours and 7 days, a 0-100 'near the bottom' score, social sentiment and, " +
                "for everything not on Bitpanda, a 0-100 risk score. Free data: Bitpanda, CoinGecko, GeckoTerminal, DexScreener, pump.fun, GoPlus, Reddit, StockTwits, Telegram, news.",
            inputSchema = schema(
                required = emptyList(),
                "currency" to prop("string", "Quote currency: EUR (default), USD, CHF or GBP", enum = AppConfig.SUPPORTED_CURRENCIES),
                "tier" to prop("string", "Which coins: all (default), bitpanda, meme (memecoins incl. Bitpanda ones), trending, dex, speculative (everything not on Bitpanda)", enum = TIER_FILTERS),
                "sort" to prop("string", "Sort order", enum = listOf("market_cap", "change_24h", "change_7d", "up_24h", "up_7d", "bottom_score", "sentiment")),
                "limit" to prop("integer", "Max rows (default 50, max 600)"),
                "query" to prop("string", "Optional filter by name or symbol"),
            ),
            toolAnnotations = readOnly,
        ) { req -> safe { overview(req.arguments) } }

        server.addTool(
            name = "speculative_radar",
            title = "Memecoins and DEX radar",
            description = "Speculative coins that Bitpanda does not list: the largest memecoins, CoinGecko's trending coins, and DEX-only tokens from GeckoTerminal trending/new pools, " +
                "DexScreener boosts/profiles/takeovers and pump.fun, with liquidity, pool age, buyers vs sellers, contract checks (GoPlus/honeypot.is) and a 0-100 risk score. " +
                "Tokens with red flags, pools younger than 24 h or unchecked contracts get no 'up' probability. Information only; most such tokens lose most of their value.",
            inputSchema = schema(
                emptyList(),
                "tier" to prop("string", "meme, trending, dex or speculative (all three, default)", enum = listOf("speculative", "meme", "trending", "dex")),
                "sort" to prop("string", "Sort order (default volume)", enum = listOf("volume", "change_24h", "up_24h", "risk", "liquidity", "newest", "market_cap")),
                "max_risk" to prop("integer", "Only tokens with risk score <= this (0-100)"),
                "hide_blocked" to prop("boolean", "Leave out tokens without a forecast (red flags, too new, unverified); default false"),
                "limit" to prop("integer", "Max rows (default 40, max 300)"),
            ),
            toolAnnotations = readOnly,
        ) { req -> safe { radar(req.arguments) } }

        server.addTool(
            name = "get_forecast",
            title = "Forecast for one coin",
            description = "Explainable forecast for one coin (Bitpanda, memecoin or DEX token): probability of going up in 24h and 7 days, likely price range, the drivers behind it, " +
                "a walk-forward backtest, whether the price is near a local low and, for coins not on Bitpanda, the risk score, contract checks and where it trades. " +
                "deep=true loads history (1 year daily, or a week of hourly DEX candles), fresh contract checks and per-coin sentiment (slower).",
            inputSchema = schema(listOf("symbol"), "symbol" to prop("string", "Ticker, e.g. BTC, SOL, BEST"), "deep" to prop("boolean", "Load 1-year history and per-coin sentiment (default true)")),
            toolAnnotations = readOnly,
        ) { req -> safe { forecast(req.arguments) } }

        server.addTool(
            name = "get_sentiment",
            title = "Social sentiment",
            description = "Sentiment for one coin (or MARKET for the whole crypto market) from Reddit, StockTwits, public Telegram channels, news and the cloud " +
                "sentiment hub (FinBERT/CryptoBERT/VADER over 100+ Telegram channels and your own Telegram account when connected), with example posts.",
            inputSchema = schema(listOf("symbol"), "symbol" to prop("string", "Ticker or MARKET")),
            toolAnnotations = readOnly,
        ) { req -> safe { sentiment(req.arguments) } }

        server.addTool(
            name = "find_opportunities",
            title = "Find coins likely to rise",
            description = "Ranks coins by the probability of rising (weighted by confidence), optionally only those near a local low. Bitpanda coins by default; " +
                "tier=all adds memecoins and DEX tokens that passed the risk checks. Not financial advice.",
            inputSchema = schema(
                emptyList(),
                "tier" to prop("string", "bitpanda (default), all, meme, trending, dex or speculative", enum = TIER_FILTERS),
                "horizon" to prop("string", "24h or 7d (default 7d)", enum = listOf("24h", "7d")),
                "min_probability" to prop("number", "Minimum probability of going up, 0-1 (default 0.55)"),
                "near_bottom" to prop("boolean", "Only coins with bottom score >= 60"),
                "limit" to prop("integer", "Max results (default 15)"),
            ),
            toolAnnotations = readOnly,
        ) { req -> safe { opportunities(req.arguments) } }

        server.addTool(
            name = "get_prices",
            title = "Prices",
            description = "Current prices for one or more Bitpanda coins (Bitpanda ticker price when available, else CoinGecko), with 1h/24h/7d/30d changes.",
            inputSchema = schema(listOf("symbols"), "symbols" to prop("string", "Comma-separated tickers, e.g. BTC,ETH,SOL")),
            toolAnnotations = readOnly,
        ) { req -> safe { prices(req.arguments) } }

        server.addTool(
            name = "list_bitpanda_assets",
            title = "Bitpanda crypto list",
            description = "Lists every cryptocurrency CryptoPulse considers tradeable on Bitpanda, with the source of that list (live Bitpanda API or bundled snapshot).",
            inputSchema = schema(emptyList(), "query" to prop("string", "Optional filter")),
            toolAnnotations = readOnly,
        ) { req -> safe { assets(req.arguments) } }

        server.addTool(
            name = "market_mood",
            title = "Market mood",
            description = "Crypto Fear & Greed index with 30-day history, total market cap change, Bitcoin trend, overall social mood and the status of the cloud sentiment hub.",
            toolAnnotations = readOnly,
        ) { _ -> safe { mood() } }

        server.addTool(
            name = "telegram_sources",
            title = "Telegram sources",
            description = "The public Telegram channels and groups CryptoPulse reads, which ones the phone scans directly, and whether your own Telegram account is connected to the cloud hub.",
            inputSchema = schema(emptyList(), "kind" to prop("string", "channel, group or all", enum = listOf("channel", "group", "all"))),
            toolAnnotations = readOnly,
        ) { req -> safe { telegramSources(req.arguments) } }

        server.addTool(
            name = "daily_digest",
            title = "Digest",
            description = "A short daily briefing (best 24h/7d odds, coins near lows, biggest moves, mood). lang=hr for Croatian.",
            inputSchema = schema(emptyList(), "lang" to prop("string", "en or hr", enum = listOf("en", "hr"))),
            toolAnnotations = readOnly,
        ) { req -> safe { CallToolResult(listOf(TextContent(Digest.build(snapshot(), req.arguments.str("lang") ?: "en")))) } }

        server.addPrompt(
            name = "crypto_briefing",
            description = "Ask for a briefing on the Bitpanda crypto market or one coin",
            arguments = listOf(PromptArgument("symbol", "Optional coin ticker", required = false)),
        ) { req ->
            val symbol = req.params.arguments?.get("symbol")?.takeIf { it.isNotBlank() }
            val text = if (symbol == null) {
                "Use market_mood, then find_opportunities (horizon 24h and 7d) and summarise which Bitpanda coins look most likely to rise, which are near a local low, and the main risks. Mention that these are probabilities, not advice."
            } else {
                "Use get_forecast for $symbol with deep=true and get_sentiment for $symbol. Explain the 24h and 7-day odds, the likely range, whether the price is near a bottom and what drives it. Mention that these are probabilities, not advice."
            }
            GetPromptResult(listOf(PromptMessage(Role.User, TextContent(text))), description = "CryptoPulse briefing")
        }
    }

    // ---- handlers ----------------------------------------------------------------------------

    private suspend fun snapshot(): MarketSnapshot {
        val snap = app.repository.refresh()
        if (enrichJob?.isActive != true) {
            enrichJob = background.launch {
                try {
                    app.repository.enrichSentiment()
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    System.err.println("[cryptopulse] background sentiment failed: ${e.message}")
                }
            }
        }
        return snap
    }

    private fun applyCurrency(args: JsonObject?) {
        val cur = args.str("currency")?.uppercase() ?: return
        require(cur in AppConfig.SUPPORTED_CURRENCIES) { "Unsupported currency $cur; use one of ${AppConfig.SUPPORTED_CURRENCIES}" }
        if (cur != app.config.currency) app.updateConfig(app.config.copy(currency = cur))
    }

    private suspend fun overview(args: JsonObject?): CallToolResult {
        applyCurrency(args)
        val snap = snapshot()
        val query = args.str("query")?.lowercase()
        val limit = (args.int("limit") ?: 50).coerceIn(1, 600)
        val tier = tierFilter(args.str("tier"), "all")
        var rows = snap.rows.filter { inTier(it, tier) && (query == null || it.asset.symbol.lowercase().contains(query) || it.asset.name.lowercase().contains(query)) }
        rows = when (args.str("sort")) {
            "change_24h" -> rows.sortedByDescending { it.market?.change24hPct ?: Double.NEGATIVE_INFINITY }
            "change_7d" -> rows.sortedByDescending { it.market?.change7dPct ?: Double.NEGATIVE_INFINITY }
            // rows without a forecast (red flags, too new) only carry placeholders, so they go last
            "up_24h" -> rows.sortedByDescending { it.upOrNull(it.forecast?.h24) ?: -1.0 }
            "up_7d" -> rows.sortedByDescending { it.upOrNull(it.forecast?.d7) ?: -1.0 }
            "bottom_score" -> rows.sortedByDescending { r -> r.forecast?.bottom?.takeIf { it.label !in Format.NO_BOTTOM_LABELS && r.risk?.blocked != true }?.score ?: -1 }
            "sentiment" -> rows.sortedByDescending { it.sentiment?.composite ?: -2.0 }
            else -> rows
        }.take(limit)
        val header = buildString {
            appendLine("Bitpanda cryptocurrencies: ${snap.universeSize} (list source: ${snap.universeSource.name.lowercase().replace('_', ' ')}), prices in ${snap.currency}, updated ${ageText(snap.fetchedAtMs)}.")
            val extra = snap.rows.groupingBy { it.asset.tier }.eachCount().filterKeys { it != Tier.BITPANDA }
            if (extra.isNotEmpty()) appendLine("Not on Bitpanda (information only): " + extra.entries.joinToString { "${it.value} ${it.key.label.lowercase()}" } + ". Showing: $tier.")
            snap.context?.fearGreed?.let { appendLine("Fear & Greed: ${it.value} (${it.classification}).") }
            if (snap.warnings.isNotEmpty()) appendLine("Notes: ${snap.warnings.take(4).joinToString("; ")}")
            appendLine()
        }
        val structured = buildJsonObject {
            put("currency", snap.currency)
            put("universeSource", snap.universeSource.name)
            put("fetchedAtMs", snap.fetchedAtMs)
            put("rows", json.encodeToJsonElement(ListSerializer(AssetRow.serializer()), rows.map { compact(it) }))
        }
        return CallToolResult(listOf(TextContent(header + Format.overviewTable(rows, snap.currency) + "\n_Probabilities, not certainties. Not financial advice._")), structuredContent = structured)
    }

    private suspend fun forecast(args: JsonObject?): CallToolResult {
        val symbol = requireSymbol(args)
        snapshot()
        val deep = args.bool("deep") ?: true
        app.repository.row(symbol) ?: throw IllegalArgumentException(unknown(symbol))
        val row = if (deep) app.repository.detail(symbol).row else app.repository.row(symbol)!!
        val f = row.forecast ?: return CallToolResult(listOf(TextContent("No market data for $symbol right now, so no forecast.")), isError = true)
        return CallToolResult(listOf(TextContent(Format.forecast(row.asset.name, f, row))), structuredContent = forecastJson(row, f))
    }

    /** The forecast plus the row's risk; a horizon without a call (red flags, too new) gets probabilityUp null, not its 0.5 placeholder. */
    private fun forecastJson(row: AssetRow, f: Forecast): JsonObject {
        val o = encode(Forecast.serializer(), f)
        fun horizon(key: String, h: HorizonForecast): Pair<String, JsonElement> =
            key to if (row.upOrNull(h) != null) o.getValue(key) else JsonObject(o.getValue(key).jsonObject + ("probabilityUp" to JsonNull))
        val risk = row.risk?.let { "risk" to json.encodeToJsonElement(RiskAssessment.serializer(), it) }
        return JsonObject(o + listOfNotNull(horizon("h24", f.h24), horizon("d7", f.d7), risk))
    }

    private suspend fun sentiment(args: JsonObject?): CallToolResult {
        val symbol = args.str("symbol")?.uppercase()?.trim() ?: throw IllegalArgumentException("symbol is required")
        val s = if (symbol == "MARKET" || symbol == SentimentService.GLOBAL_SYMBOL) {
            app.sentiment.globalMood()
        } else {
            snapshot()
            val row = app.repository.row(symbol) ?: throw IllegalArgumentException(unknown(symbol))
            app.sentiment.forAsset(row.asset)
        }
        return CallToolResult(listOf(TextContent(Format.sentiment(s))), structuredContent = encode(dev.cryptopulse.core.model.SentimentSummary.serializer(), s))
    }

    private suspend fun opportunities(args: JsonObject?): CallToolResult {
        val snap = snapshot()
        val horizon = when (args.str("horizon")?.trim()?.lowercase()) {
            null, "", "7d", "1w", "7", "week" -> "7d"
            "24h", "1d", "24", "day" -> "24h"
            else -> throw IllegalArgumentException("horizon must be 24h or 7d")
        }
        val minP = args.dbl("min_probability") ?: 0.55
        val nearBottom = args.bool("near_bottom") ?: false
        val limit = (args.int("limit") ?: 15).coerceIn(1, 100)
        val tier = tierFilter(args.str("tier"), "bitpanda")
        val ranked = snap.rows.mapNotNull { r ->
            if (!inTier(r, tier)) return@mapNotNull null
            val f = r.forecast ?: return@mapNotNull null
            if (f.bottom.label in Format.NO_BOTTOM_LABELS) return@mapNotNull null
            val h = if (horizon == "24h") f.h24 else f.d7
            // red flags or too new: no forecast, whatever min_probability is
            val p = r.upOrNull(h) ?: return@mapNotNull null
            if (p < minP) return@mapNotNull null
            if (nearBottom && f.bottom.score < 60) return@mapNotNull null
            Triple(r, h, p * (0.5 + h.confidence))
        }.sortedByDescending { it.third }.take(limit)
        val text = buildString {
            appendLine("Coins with >= ${Format.pct(minP)} chance of rising over $horizon${if (nearBottom) " and near a local low" else ""}: ${ranked.size}")
            ranked.forEachIndexed { i, (r, h, _) ->
                val where = if (r.asset.onBitpanda) "" else " [${r.asset.tier.label}, ${r.risk?.label?.lowercase() ?: "risk unknown"}, not on Bitpanda]"
                appendLine("${i + 1}. ${r.asset.name} (${r.asset.symbol})$where ${Format.pct(h.probabilityUp)} up, confidence ${Format.pct(h.confidence)}, bottom score ${r.forecast!!.bottom.score} — ${h.signals.firstOrNull()?.description ?: ""}")
            }
            appendLine("_Probabilities, not certainties. Not financial advice._")
        }
        val structured = buildJsonObject {
            putJsonArray("results") {
                ranked.forEach { (r, h, score) ->
                    add(buildJsonObject {
                        put("symbol", r.asset.symbol); put("name", r.asset.name); put("probabilityUp", h.probabilityUp)
                        put("confidence", h.confidence); put("bottomScore", r.forecast!!.bottom.score); put("rankScore", score)
                        put("tier", r.asset.tier.name); r.risk?.let { put("riskScore", it.score) }
                        r.market?.price?.let { put("price", it) }
                    })
                }
            }
        }
        return CallToolResult(listOf(TextContent(text)), structuredContent = structured)
    }

    private suspend fun prices(args: JsonObject?): CallToolResult {
        val symbols = args.str("symbols")?.split(',', ' ', ';')?.map { Text.normalizeSymbol(it) }?.filter { it.isNotEmpty() }.orEmpty()
        require(symbols.isNotEmpty()) { "symbols is required, e.g. BTC,ETH" }
        val snap = snapshot()
        val text = buildString {
            for (s in symbols) {
                val r = snap.rows.firstOrNull { it.asset.symbol == s }
                val m = r?.market
                if (m == null) { appendLine("$s: ${if (r == null) "not in the list (Bitpanda, memecoins, DEX radar)" else "no price data right now"}"); continue }
                val note = if (r.asset.onBitpanda) "" else " [${r.asset.tier.label}, not on Bitpanda]"
                appendLine("${r.asset.name} ($s)$note: ${Text.formatPrice(m.bitpandaPrice ?: m.price, m.currency)} · 1h ${Text.formatPct(m.change1hPct, 2)} · 24h ${Text.formatPct(m.change24hPct, 2)} · 7d ${Text.formatPct(m.change7dPct, 2)} · 30d ${Text.formatPct(m.change30dPct, 1)}")
            }
        }
        return CallToolResult(listOf(TextContent(text)))
    }

    private suspend fun assets(args: JsonObject?): CallToolResult {
        val snap = snapshot()
        val q = args.str("query")?.lowercase()
        val list = snap.rows.map { it.asset }.filter { it.onBitpanda && (q == null || it.symbol.lowercase().contains(q) || it.name.lowercase().contains(q)) }
        val text = "Bitpanda crypto list (${snap.universeSource.name.lowercase().replace('_', ' ')}): ${list.size} assets\n" +
            list.joinToString(", ") { "${it.symbol} (${it.name})" }
        return CallToolResult(listOf(TextContent(text)))
    }

    private suspend fun mood(): CallToolResult {
        val snap = snapshot()
        val ctx = snap.context
        val global = app.sentiment.globalMood()
        val (feedStatus, feedMsg) = app.sentiment.feedState()
        val feed = app.sentiment.feed()
        val text = buildString {
            ctx?.fearGreed?.let { fg ->
                appendLine("Fear & Greed: ${fg.value} (${fg.classification})${fg.change7d?.let { " · 7 days ago ${fg.value - it}" } ?: ""}")
            }
            ctx?.global?.let { g -> appendLine("Total market cap 24h: ${Text.formatPct(g.marketCapChange24hPct, 2)} · BTC dominance ${g.btcDominancePct?.let { "%.1f%%".format(it) } ?: "—"}") }
            ctx?.btcTrendScore?.let { appendLine("Bitcoin trend score: ${"%+.2f".format(it)} (−1 strong down … +1 strong up)") }
            appendLine("Social mood: ${global.label} (${"%+.2f".format(global.composite)}, ${global.totalSamples} items)")
            appendLine("Cloud sentiment hub: ${if (feed != null) "ok, models ${feed.models.joinToString()}, ${feed.sources.joinToString { "${it.name}=${it.status}" }}" else "${feedStatus.name.lowercase()} ${feedMsg ?: ""}"}")
            if (ctx?.trendingCoingeckoIds?.isNotEmpty() == true) appendLine("Trending on CoinGecko: ${ctx.trendingCoingeckoIds.take(10).joinToString()}")
        }
        return CallToolResult(listOf(TextContent(text)))
    }

    private suspend fun telegramSources(args: JsonObject?): CallToolResult {
        val kind = args.str("kind") ?: "all"
        val dir = app.sentiment.telegramDirectory
        val scanned = app.sentiment.channelsToScan(app.config).toSet()
        val entries = dir.entries.filter { kind == "all" || it.kind.equals(kind, true) }
        val feed = app.sentiment.feed()
        val account = feed?.sources?.firstOrNull { it.name == "telegram_account" }
        val text = buildString {
            appendLine("Telegram directory: ${dir.entries.size} entries (${dir.entries.count { it.isChannel }} channels, ${dir.groups.size} groups).")
            appendLine("The phone scans ${scanned.size} channels directly (t.me/s web preview, no login). The cloud hub scans every channel${if (account?.status == "ok") " and your Telegram account (${account.detail ?: "connected"})" else "; your own Telegram account is not connected yet (see tools/sentiment_hub/README.md)"}.")
            entries.forEach { e -> appendLine("- @${e.handle} — ${e.title} [${e.kind}, ${e.category}, ${e.lang}]${if (e.handle in scanned) " · scanned on device" else ""}") }
        }
        return CallToolResult(listOf(TextContent(text)))
    }

    // ---- helpers -----------------------------------------------------------------------------

    private suspend fun safe(block: suspend () -> CallToolResult): CallToolResult = try {
        block()
    } catch (e: CancellationException) {
        throw e
    } catch (e: IllegalArgumentException) {
        CallToolResult(listOf(TextContent(e.message ?: "Invalid arguments")), isError = true)
    } catch (e: Exception) {
        System.err.println("[cryptopulse] tool failed: $e")
        CallToolResult(listOf(TextContent("CryptoPulse could not complete this request: ${e.message ?: e::class.simpleName}")), isError = true)
    }

    private fun requireSymbol(args: JsonObject?): String =
        args.str("symbol")?.let { Text.normalizeSymbol(it) }?.takeIf { it.isNotEmpty() } ?: throw IllegalArgumentException("symbol is required, e.g. BTC")

    private fun unknown(symbol: String) = "$symbol is not in the list (Bitpanda coins, memecoins, trending coins and the DEX radar). Use list_bitpanda_assets or speculative_radar to see what is available."

    private fun tierFilter(raw: String?, default: String): String {
        val t = raw?.trim()?.lowercase()?.ifEmpty { null } ?: return default
        require(t in TIER_FILTERS) { "tier must be one of $TIER_FILTERS" }
        return t
    }

    private fun inTier(r: AssetRow, filter: String): Boolean = when (filter) {
        "all" -> true
        "bitpanda" -> r.asset.onBitpanda
        "meme" -> r.asset.isMeme
        "trending" -> r.asset.tier == Tier.TRENDING || "trending" in r.asset.tags
        "dex" -> r.asset.tier == Tier.DEX
        "speculative" -> !r.asset.onBitpanda
        else -> true
    }

    private suspend fun radar(args: JsonObject?): CallToolResult {
        val snap = snapshot()
        val tier = tierFilter(args.str("tier"), "speculative").let { if (it == "all" || it == "bitpanda") "speculative" else it }
        val maxRisk = args.int("max_risk")
        val hideBlocked = args.bool("hide_blocked") ?: false
        val limit = (args.int("limit") ?: 40).coerceIn(1, 300)
        var rows = snap.rows.filter { !it.asset.onBitpanda && inTier(it, tier) }
            .filter { maxRisk == null || (it.risk?.score ?: 100) <= maxRisk }
            .filter { !hideBlocked || it.upOrNull(it.forecast?.h24) != null }
        rows = when (args.str("sort")) {
            "change_24h" -> rows.sortedByDescending { it.market?.change24hPct ?: Double.NEGATIVE_INFINITY }
            "up_24h" -> rows.sortedByDescending { it.upOrNull(it.forecast?.h24) ?: -1.0 }
            "risk" -> rows.sortedBy { it.risk?.score ?: 101 }
            "liquidity" -> rows.sortedByDescending { it.dex?.liquidityUsd ?: -1.0 }
            "newest" -> rows.sortedByDescending { it.dex?.pairCreatedAtMs ?: Long.MIN_VALUE }
            "market_cap" -> rows.sortedByDescending { it.market?.marketCap ?: -1.0 }
            else -> rows.sortedByDescending { it.dex?.volumeUsd?.h24 ?: it.market?.volume24h ?: -1.0 }
        }.take(limit)
        val text = buildString {
            appendLine("Speculative coins not on Bitpanda ($tier): ${rows.size} shown, prices in ${snap.currency}, updated ${ageText(snap.fetchedAtMs)}.")
            appendLine(Format.SPECULATIVE_NOTICE)
            appendLine()
            append(Format.radarTable(rows, snap.currency, System.currentTimeMillis()))
        }
        val structured = buildJsonObject {
            put("currency", snap.currency)
            put("rows", json.encodeToJsonElement(ListSerializer(AssetRow.serializer()), rows.map { compact(it) }))
        }
        return CallToolResult(listOf(TextContent(text)), structuredContent = structured)
    }

    private fun ageText(ms: Long): String {
        val min = (System.currentTimeMillis() - ms) / 60_000
        return if (min < 1) "just now" else "$min min ago"
    }

    /** Drops the 168-point sparkline and long signal lists from structured overview output to keep it small. */
    private fun compact(r: AssetRow): AssetRow = r.copy(
        market = r.market?.copy(sparkline7d = emptyList()),
        forecast = r.forecast?.let { f -> f.copy(h24 = f.h24.copy(signals = f.h24.signals.take(3)), d7 = f.d7.copy(signals = f.d7.signals.take(3)), sentiment = null) },
        sentiment = r.sentiment?.copy(perSource = emptyList()),
    )

    private fun <T> encode(serializer: KSerializer<T>, value: T): JsonObject = json.encodeToJsonElement(serializer, value).jsonObject

    companion object {
        const val VERSION = "0.2.0"
        val TIER_FILTERS = listOf("all", "bitpanda", "meme", "trending", "dex", "speculative")
        private val json = Json { encodeDefaults = true; explicitNulls = false }

        private fun JsonObject?.str(key: String): String? = this?.get(key)?.let { (it as? JsonPrimitive)?.contentOrNull }?.trim()?.takeIf { it.isNotEmpty() }
        private fun JsonObject?.int(key: String): Int? = this?.get(key)?.let { (it as? JsonPrimitive)?.let { p -> p.intOrNull ?: p.contentOrNull?.toIntOrNull() } }
        private fun JsonObject?.dbl(key: String): Double? = this?.get(key)?.let { (it as? JsonPrimitive)?.let { p -> p.doubleOrNull ?: p.contentOrNull?.toDoubleOrNull() } }
        private fun JsonObject?.bool(key: String): Boolean? = this?.get(key)?.let { (it as? JsonPrimitive)?.let { p -> p.booleanOrNull ?: p.contentOrNull?.toBooleanStrictOrNull() } }

        fun prop(type: String, description: String, enum: List<String>? = null): JsonObject = buildJsonObject {
            put("type", type)
            put("description", description)
            if (enum != null) put("enum", JsonArray(enum.map { JsonPrimitive(it) }))
        }

        fun schema(required: List<String>, vararg props: Pair<String, JsonObject>): ToolSchema =
            ToolSchema(properties = buildJsonObject { props.forEach { (k, v) -> put(k, v) } }, required = required.ifEmpty { null })
    }
}
