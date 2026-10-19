package dev.cryptopulse.core.sources.safety

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.model.TokenSafety
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.bool
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.obj
import dev.cryptopulse.core.util.str
import kotlinx.serialization.json.JsonElement

/**
 * Free contract checks for DEX tokens: GoPlus Security (EVM chains and Solana, keyless) and
 * honeypot.is (EVM buy/sell simulation, keyless). Results are cached for hours: contracts rarely
 * change, and both services rate-limit anonymous callers.
 */
class TokenSafetyApi(
    private val http: Http,
    private val clock: Clock = Clock.SYSTEM,
    private val goplusBase: String = "https://api.gopluslabs.io/api/v1",
    private val honeypotBase: String = "https://api.honeypot.is/v2",
) {
    /** A fresh cached check exists, so [check] would not use any quota. */
    fun isCached(chainId: String, address: String): Boolean = http.isFresh(urlFor(chainId, address) ?: return false, 6 * 3_600_000L, ::isGoPlusAnswer)

    private fun urlFor(chainId: String, address: String): String? = when {
        chainId == "solana" -> "$goplusBase/solana/token_security?contract_addresses=$address"
        EVM_CHAIN_IDS[chainId] != null -> "$goplusBase/token_security/${EVM_CHAIN_IDS[chainId]}?contract_addresses=${address.lowercase()}"
        else -> null
    }

    /** True when GoPlus (and so [check]) covers this DexScreener chain id. */
    fun supports(chainId: String): Boolean = chainId == "solana" || chainId in EVM_CHAIN_IDS

    /**
     * GoPlus contract check of one token. One address per call: GoPlus silently answers only the
     * first address of a comma-separated EVM list. Answers are cached for 6 h, so repeated refreshes
     * are free; [SourceResult.Success.fromCache] tells callers whether the call used any quota. Quota
     * (4029) and partial-data (2) replies come inside a 200 and are never cached: the last good answer
     * is used instead while it is under 3 days old.
     */
    suspend fun check(chainId: String, address: String): SourceResult<TokenSafety?> {
        val solana = chainId == "solana"
        val url = urlFor(chainId, address) ?: return SourceResult.Failure(SourceStatus.EMPTY, "No free safety check for $chainId")
        return when (val r = http.getText(url, JSON, ttlMs = 6 * 3_600_000L, staleOkMs = 3 * 24 * 3_600_000L, retries = 0, accept = ::isGoPlusAnswer)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "GoPlus")
            is HttpResult.Ok -> try {
                when (val code = JsonUtil.parse(r.body)["code"].int) {
                    1 -> {
                        val map = if (solana) parseGoPlusSolana(r.body, clock.nowMs()) else parseGoPlusEvm(r.body, clock.nowMs())
                        // fresh tokens are often not indexed yet: an empty result is "unknown", not "safe"
                        SourceResult.Success(map[if (solana) address else address.lowercase()] ?: map.values.firstOrNull(), r.fromCache, r.ageMs)
                    }
                    4029 -> SourceResult.Failure(SourceStatus.RATE_LIMITED, "GoPlus: free quota used up for now", 429)
                    else -> SourceResult.Failure(SourceStatus.UNAVAILABLE, "GoPlus: code $code")
                }
            } catch (e: Exception) {
                SourceResult.parseError("GoPlus", e)
            }
        }
    }

    /** honeypot.is buy/sell simulation for one EVM token (Ethereum, BSC, Base). */
    suspend fun simulate(chainId: String, address: String): SourceResult<TokenSafety> {
        val evmId = EVM_CHAIN_IDS[chainId]?.takeIf { it in HONEYPOT_CHAINS }
            ?: return SourceResult.Failure(SourceStatus.EMPTY, "honeypot.is does not cover $chainId")
        val url = "$honeypotBase/IsHoneypot?address=$address&chainID=$evmId"
        return when (val r = http.getText(url, JSON, ttlMs = 6 * 3_600_000L, staleOkMs = 3 * 24 * 3_600_000L, retries = 0)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "honeypot.is")
            is HttpResult.Ok -> try {
                SourceResult.Success(parseHoneypotIs(r.body, clock.nowMs()), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("honeypot.is", e)
            }
        }
    }

    companion object {
        private val JSON = mapOf("Accept" to "application/json")

        /** DexScreener chain id -> EVM chain id used by GoPlus and honeypot.is. */
        val EVM_CHAIN_IDS: Map<String, String> = mapOf(
            "ethereum" to "1", "bsc" to "56", "base" to "8453", "arbitrum" to "42161", "polygon" to "137",
            "avalanche" to "43114", "optimism" to "10", "linea" to "59144", "blast" to "81457", "sonic" to "146",
            "zksync" to "324", "scroll" to "534352", "mantle" to "5000", "cronos" to "25", "abstract" to "2741",
        )
        private val HONEYPOT_CHAINS = setOf("1", "56", "8453")
        private val BURN_ADDRESSES = setOf("0x0000000000000000000000000000000000000000", "0x000000000000000000000000000000000000dead")

        /** GoPlus reports errors in the body of a 200: only code 1 is an answer. */
        private fun isGoPlusAnswer(body: String): Boolean = runCatching { JsonUtil.parse(body)["code"].int == 1 }.getOrDefault(false)

        private fun flag(el: JsonElement?): Boolean? = when (el.str) { "1" -> true; "0" -> false; else -> null }

        /** "0.0899" fractions -> percent. */
        private fun pct(el: JsonElement?): Double? = el.dbl?.let { it * 100 }

        fun parseGoPlusEvm(body: String, nowMs: Long): Map<String, TokenSafety> {
            val result = JsonUtil.parse(body)["result"].obj ?: return emptyMap()
            return result.mapNotNull { (address, o) ->
                if (o.obj == null || o.obj!!.isEmpty()) return@mapNotNull null
                // only the zero or dead address is renounced: GoPlus answers "" for "no owner or owner not
                // found" (role-based minting or pausing, for one) and leaves the field out when it does not know
                val owner = o["owner_address"].str?.trim()?.lowercase()
                val renounced = owner?.takeIf { it.isNotEmpty() }?.let { it in BURN_ADDRESSES }
                // owner-only switches do nothing once ownership is renounced (PEPE has a blacklist and a pause
                // switch in its code, but nobody can use them)
                val powers = buildList {
                    if (flag(o["owner_change_balance"]) == true) add("owner can change balances")
                    if (flag(o["can_take_back_ownership"]) == true) add("ownership can be taken back")
                    if (flag(o["hidden_owner"]) == true) add("hidden owner")
                    if (flag(o["selfdestruct"]) == true) add("contract can self-destruct")
                    if ((o["honeypot_with_same_creator"].int ?: 0) > 0) add("creator made honeypots before")
                    if (flag(o["cannot_sell_all"]) == true) add("holders cannot sell everything")
                    if (flag(o["cannot_buy"]) == true) add("buying is blocked")
                    if (renounced != true) {
                        if (flag(o["transfer_pausable"]) == true) add("transfers can be paused")
                        if (flag(o["is_blacklisted"]) == true) add("owner can blacklist wallets")
                        if (flag(o["slippage_modifiable"]) == true) add("owner can change the tax")
                        if (flag(o["personal_slippage_modifiable"]) == true) add("owner can set taxes per wallet")
                        if (flag(o["trading_cooldown"]) == true) add("trading cooldown")
                    }
                }
                val holders = o["holders"].arr.orEmpty()
                val lp = o["lp_holders"].arr.orEmpty()
                // LP tokens held by the token's own renounced contract (most of PEPE's) can never be withdrawn
                val lpLocked = lp.filter { h ->
                    val holder = h["address"].str?.lowercase()
                    h["is_locked"].int == 1 || holder in BURN_ADDRESSES || (renounced == true && holder == address.lowercase())
                }.sumOf { (it["percent"].dbl ?: 0.0) * 100 }
                address.lowercase() to TokenSafety(
                    source = "goplus",
                    honeypot = flag(o["is_honeypot"]),
                    buyTaxPct = pct(o["buy_tax"]),
                    sellTaxPct = pct(o["sell_tax"]),
                    mintable = flag(o["is_mintable"])?.let { it && renounced != true },
                    ownerPowers = powers,
                    ownerRenounced = renounced,
                    openSource = flag(o["is_open_source"]),
                    proxy = flag(o["is_proxy"]),
                    holderCount = o["holder_count"].int,
                    top10HolderPct = holders.take(10).sumOf { (it["percent"].dbl ?: 0.0) * 100 }.takeIf { holders.isNotEmpty() },
                    creatorPct = pct(o["creator_percent"]),
                    lpLockedPct = lpLocked.takeIf { lp.isNotEmpty() }?.coerceAtMost(100.0),
                    cexListed = o["is_in_cex"]["cex_list"].arr.orEmpty().mapNotNull { it.str },
                    trusted = flag(o["trust_list"]),
                    checkedAtMs = nowMs,
                )
            }.toMap()
        }

        fun parseGoPlusSolana(body: String, nowMs: Long): Map<String, TokenSafety> {
            val result = JsonUtil.parse(body)["result"].obj ?: return emptyMap()
            return result.mapNotNull { (address, o) ->
                if (o.obj == null || o.obj!!.isEmpty()) return@mapNotNull null
                fun status(key: String) = flag(o[key]["status"])
                val powers = buildList {
                    if (status("balance_mutable_authority") == true) add("someone can change balances")
                    if (status("closable") == true) add("token can be closed")
                    if (status("transfer_hook_upgradable") == true || o["transfer_hook"].arr.orEmpty().isNotEmpty()) add("transfer hook")
                    if (o["non_transferable"].str == "1") add("token cannot be transferred")
                    if (status("default_account_state_upgradable") == true || o["default_account_state"].str == "2") add("new holders can be frozen by default")
                    if (status("metadata_mutable") == true) add("name and logo can be changed")
                }
                val holders = o["holders"].arr.orEmpty()
                val burn = o["dex"].arr.orEmpty().maxByOrNull { it["tvl"].dbl ?: 0.0 }?.get("burn_percent").dbl
                val fee = o["transfer_fee"]["current_fee_rate"]["fee_rate"].dbl ?: o["transfer_fee"]["fee_rate"].dbl
                address to TokenSafety(
                    source = "goplus",
                    mintable = status("mintable"),
                    freezable = status("freezable"),
                    ownerPowers = powers,
                    transferFeePct = fee?.let { if (it > 1) it / 100 else it * 100 },
                    holderCount = o["holder_count"].int,
                    top10HolderPct = holders.take(10).sumOf { (it["percent"].dbl ?: 0.0) * 100 }.takeIf { holders.isNotEmpty() },
                    lpLockedPct = burn,
                    trusted = o["trusted_token"].int?.let { it == 1 },
                    checkedAtMs = nowMs,
                )
            }.toMap()
        }

        fun parseHoneypotIs(body: String, nowMs: Long): TokenSafety {
            val o = JsonUtil.parse(body)
            val sim = o["simulationResult"]
            val flags = o["summary"]["flags"].arr.orEmpty().mapNotNull { f -> f["flag"].str ?: f["description"].str ?: f.str }
            return TokenSafety(
                source = "honeypot.is",
                honeypot = o["honeypotResult"]["isHoneypot"].bool,
                buyTaxPct = sim["buyTax"].dbl,
                sellTaxPct = sim["sellTax"].dbl,
                ownerPowers = flags,
                openSource = o["contractCode"]["openSource"].bool,
                proxy = o["contractCode"]["isProxy"].bool,
                holderCount = o["token"]["totalHolders"].int,
                checkedAtMs = nowMs,
            )
        }
    }
}
