package dev.cryptopulse.core.util

/** Injectable time source so forecasts, caches and tests are deterministic. */
fun interface Clock {
    fun nowMs(): Long

    companion object {
        val SYSTEM: Clock = Clock { System.currentTimeMillis() }
        fun fixed(nowMs: Long): Clock = Clock { nowMs }
    }
}
