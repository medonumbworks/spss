package dev.cryptopulse.core.universe

import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.ListingSource
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.Text
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.bool
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.str

/** Loads `bitpanda/bitpanda_snapshot.json` (see tools/build/build_bitpanda_snapshot.py). */
object BundledSnapshot {
    const val RESOURCE = "/bitpanda/bitpanda_snapshot.json"

    fun parse(json: String): List<Asset> =
        JsonUtil.parse(json)["assets"].arr.orEmpty().mapNotNull { a ->
            val symbol = a["symbol"].str?.let { Text.normalizeSymbol(it) }?.takeIf { it.isNotEmpty() } ?: return@mapNotNull null
            Asset(
                symbol = symbol,
                name = a["name"].str ?: symbol,
                coingeckoId = a["coingeckoId"].str?.takeIf { it.isNotBlank() },
                listingSource = ListingSource.BUNDLED_SNAPSHOT,
                snapshotRank = a["rank"].int,
                verified = a["verified"].bool ?: true,
            )
        }.distinctBy { it.symbol }

    fun load(): List<Asset> {
        val text = BundledSnapshot::class.java.getResourceAsStream(RESOURCE)?.use { it.readBytes().toString(Charsets.UTF_8) }
            ?: throw IllegalStateException("Missing bundled resource $RESOURCE")
        return parse(text)
    }
}
