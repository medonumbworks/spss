package dev.cryptopulse.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
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
