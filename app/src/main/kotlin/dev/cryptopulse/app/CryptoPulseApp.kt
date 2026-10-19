package dev.cryptopulse.app

import android.app.Application
import android.content.Context
import android.util.Log
import dev.cryptopulse.app.work.AlertScheduler
import dev.cryptopulse.core.CryptoPulse
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.http.FileDiskStore
import dev.cryptopulse.core.http.OkHttpTransport
import dev.cryptopulse.core.util.LogLevel
import dev.cryptopulse.core.util.Logger
import kotlinx.serialization.json.Json
import java.io.File

class CryptoPulseApp : Application() {
    lateinit var cryptoPulse: CryptoPulse
        private set

    lateinit var settings: SettingsStore
        private set

    override fun onCreate() {
        super.onCreate()
        settings = SettingsStore(this)
        cryptoPulse = CryptoPulse.create(
            config = settings.load(),
            transport = OkHttpTransport(userAgent = "CryptoPulse/${BuildConfig.VERSION_NAME} (Android; +https://github.com/medonumbworks/spss)"),
            store = FileDiskStore(File(cacheDir, "cryptopulse")),
            logger = AndroidLogger,
        )
        // hourly "forecast crossed the threshold" notifications; a no-op when the setting is off
        AlertScheduler.sync(this, cryptoPulse.config)
    }
}

object AndroidLogger : Logger {
    override fun log(level: LogLevel, tag: String, message: String, error: Throwable?) {
        val t = "CryptoPulse/$tag"
        when (level) {
            LogLevel.DEBUG -> if (BuildConfig.DEBUG) Log.d(t, message, error)
            LogLevel.INFO -> Log.i(t, message, error)
            LogLevel.WARN -> Log.w(t, message, error)
            LogLevel.ERROR -> Log.e(t, message, error)
        }
    }
}

/** Settings persisted as JSON in SharedPreferences (no extra dependencies). */
class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("cryptopulse_settings", Context.MODE_PRIVATE)
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    fun load(): AppConfig = prefs.getString(KEY, null)?.let { runCatching { json.decodeFromString(AppConfig.serializer(), it) }.getOrNull() } ?: AppConfig()

    fun save(config: AppConfig) {
        prefs.edit().putString(KEY, json.encodeToString(AppConfig.serializer(), config)).apply()
    }

    private companion object {
        const val KEY = "config_v1"
    }
}
