package dev.cryptopulse.core.model

import kotlinx.serialization.Serializable

/**
 * Explicit pump-and-dump warning for one asset, shown to the user as "Pump-and-dump warning" with
 * its [reasons]. No alert only means none of the conservative patterns fired, not that the coin
 * is safe.
 */
@Serializable
data class PumpAlert(val reasons: List<String>)
