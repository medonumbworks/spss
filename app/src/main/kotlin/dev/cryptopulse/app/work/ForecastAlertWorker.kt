package dev.cryptopulse.app.work

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import dev.cryptopulse.app.CryptoPulseApp
import dev.cryptopulse.app.MainActivity
import dev.cryptopulse.app.R
import dev.cryptopulse.core.analysis.journal.ForecastAlert
import kotlinx.coroutines.CancellationException

/**
 * Hourly background check behind "notify me when a Bitpanda coin's predicted 24 h rise crosses
 * the threshold". The crossing detector in :core fires once per crossing, so an unchanged market
 * stays quiet; all fresh crossings of one run share a single notification.
 */
class ForecastAlertWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val app = applicationContext as? CryptoPulseApp ?: return Result.failure()
        val cp = app.cryptoPulse
        if (!cp.config.alertsEnabled) return Result.success() // scheduler races a settings change
        try {
            cp.repository.refresh(force = false) // TTLs keep an hourly check within free API limits
        } catch (e: CancellationException) {
            throw e
        } catch (_: Exception) {
            return Result.retry() // offline or a source hiccup: back off and try again
        }
        val alerts = cp.repository.alertCrossings()
        if (alerts.isNotEmpty()) postNotification(alerts, cp.config.alertThresholdPct)
        return Result.success()
    }

    private fun postNotification(alerts: List<ForecastAlert>, thresholdPct: Int) {
        val nm = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager ?: return
        if (!nm.areNotificationsEnabled()) return // POST_NOTIFICATIONS denied or notifications off
        // creating an existing channel is a no-op, so this stays idempotent across runs
        nm.createNotificationChannel(NotificationChannel(CHANNEL_ID, "Forecast alerts", NotificationManager.IMPORTANCE_DEFAULT))
        val open = PendingIntent.getActivity(
            applicationContext,
            0,
            Intent(applicationContext, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val notification = Notification.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(AlertText.title(alerts.size, thresholdPct))
            .setContentText(AlertText.summary(alerts))
            .setStyle(Notification.BigTextStyle().bigText(AlertText.summary(alerts, limit = alerts.size)))
            .setContentIntent(open)
            .setAutoCancel(true)
            .build()
        nm.notify(NOTIFICATION_ID, notification)
    }

    companion object {
        const val CHANNEL_ID = "forecast_alerts"
        private const val NOTIFICATION_ID = 2001
    }
}
