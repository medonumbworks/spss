package dev.cryptopulse.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import dev.cryptopulse.app.ui.AppRoot
import dev.cryptopulse.app.ui.AppViewModel
import dev.cryptopulse.app.ui.theme.CryptoPulseTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val viewModel: AppViewModel by viewModels()

    // the result needs no handling: the worker checks areNotificationsEnabled() before posting
    private val notificationPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // Android 13+ needs a runtime grant before the forecast alerts can show
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && viewModel.config.value.alertsEnabled &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        setContent {
            CryptoPulseTheme {
                AppRoot(viewModel)
            }
        }
        // Refresh while the app is visible; the repository's TTLs keep this within free API limits.
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                while (true) {
                    viewModel.refreshIfStale()
                    delay(60_000)
                }
            }
        }
    }
}
