package dev.cryptopulse.app.ui

import java.util.Locale

/**
 * Trend label of a buzz row: "×2.1 vs 7-day norm" once the hub has a baseline, otherwise the
 * previous window's count. Pure so it is unit-testable without Android.
 */
internal fun buzzTrendLabel(spike: Double?, mentionsPrev: Int?): String =
    if (spike != null) String.format(Locale.ROOT, "×%.1f vs 7-day norm", spike)
    else "prev window: ${mentionsPrev ?: 0}"
