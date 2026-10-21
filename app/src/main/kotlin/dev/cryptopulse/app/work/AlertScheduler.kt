package dev.cryptopulse.app.work

import android.content.Context
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import dev.cryptopulse.core.config.AppConfig
import java.util.concurrent.TimeUnit

/**
 * Keeps the hourly background forecast check in step with the settings. Called from
 * [dev.cryptopulse.app.CryptoPulseApp.onCreate] and after every alert-settings change.
 */
object AlertScheduler {
    const val WORK_NAME = "forecast-alerts"

    fun sync(context: Context, config: AppConfig) {
        val wm = WorkManager.getInstance(context.applicationContext)
        if (!config.alertsEnabled) {
            wm.cancelUniqueWork(WORK_NAME)
            return
        }
        // UPDATE so repeated calls (every app start, threshold changes) never pile up duplicate work;
        // the worker reads the threshold from the live config, so no input needs re-enqueueing.
        val request = PeriodicWorkRequestBuilder<ForecastAlertWorker>(1, TimeUnit.HOURS)
            .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
            .build()
        wm.enqueueUniquePeriodicWork(WORK_NAME, ExistingPeriodicWorkPolicy.UPDATE, request)
    }
}
