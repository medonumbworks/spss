package dev.cryptopulse.app.ui

import androidx.activity.compose.BackHandler
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun AppRoot(vm: AppViewModel) {
    val screen by vm.screen.collectAsStateWithLifecycle()
    BackHandler(enabled = screen != Screen.Market) { vm.back() }
    when (val s = screen) {
        Screen.Market -> MarketScreen(vm)
        is Screen.Detail -> DetailScreen(vm, s.symbol)
        Screen.Settings -> SettingsScreen(vm)
    }
}
