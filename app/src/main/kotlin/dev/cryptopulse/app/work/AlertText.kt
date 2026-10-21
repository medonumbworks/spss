package dev.cryptopulse.app.work

import dev.cryptopulse.core.analysis.journal.ForecastAlert
import kotlin.math.roundToInt

/** Wording of the forecast-alert notification; pure so it is unit-testable without Android. */
object AlertText {
    /** How many coins fit in the collapsed one-line notification text. */
    const val COLLAPSED_LIMIT = 6

    /** "3 coins crossed 50% up-chance (24h)", with the singular for one coin. */
    fun title(count: Int, thresholdPct: Int): String =
        "$count ${if (count == 1) "coin" else "coins"} crossed $thresholdPct% up-chance (24h)"

    /** "BTC 57%, SOL 56%, …": at most [limit] coins, an ellipsis when more crossed. */
    fun summary(alerts: List<ForecastAlert>, limit: Int = COLLAPSED_LIMIT): String {
        // rounded, not truncated: 0.57 * 100 is 56.999… in doubles and must still read "57%"
        val shown = alerts.take(limit).joinToString(", ") { "${it.symbol} ${(it.probabilityUp * 100).roundToInt()}%" }
        return if (alerts.size > limit) "$shown, …" else shown
    }
}
