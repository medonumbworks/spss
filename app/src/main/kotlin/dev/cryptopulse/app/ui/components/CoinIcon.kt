package dev.cryptopulse.app.ui.components

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.Cache
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

/** Minimal image loader for coin logos: OkHttp disk cache (20 MB) + in-memory LRU, no extra libraries. */
object CoinImages {
    private val memory = LruCache<String, Bitmap>(300)

    @Volatile
    private var client: OkHttpClient? = null

    private fun client(context: Context): OkHttpClient = client ?: synchronized(this) {
        client ?: OkHttpClient.Builder()
            .cache(Cache(File(context.cacheDir, "coin_images"), 20L * 1024 * 1024))
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(15, TimeUnit.SECONDS)
            .build()
            .also { client = it }
    }

    fun cached(url: String): Bitmap? = memory.get(url)

    suspend fun load(context: Context, url: String, sizePx: Int): Bitmap? = withContext(Dispatchers.IO) {
        memory.get(url)?.let { return@withContext it }
        try {
            client(context).newCall(Request.Builder().url(url).build()).execute().use { r ->
                if (!r.isSuccessful) return@withContext null
                val bytes = r.body?.bytes() ?: return@withContext null
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
                var sample = 1
                while (bounds.outWidth / (sample * 2) >= sizePx && bounds.outHeight / (sample * 2) >= sizePx) sample *= 2
                val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, BitmapFactory.Options().apply { inSampleSize = sample })
                bmp?.also { memory.put(url, it) }
            }
        } catch (_: Exception) {
            null
        }
    }
}

/** Coin logo from CoinGecko, falling back to the coloured initials avatar. */
@Composable
fun CoinIcon(symbol: String, imageUrl: String?, size: Dp = 40.dp) {
    if (imageUrl.isNullOrBlank()) {
        CoinAvatar(symbol, size)
        return
    }
    val context = LocalContext.current
    val px = with(androidx.compose.ui.platform.LocalDensity.current) { size.roundToPx() }
    val bitmap by produceState(initialValue = CoinImages.cached(imageUrl), imageUrl) {
        if (value == null) value = CoinImages.load(context.applicationContext, imageUrl, px)
    }
    val b = bitmap
    if (b == null) {
        CoinAvatar(symbol, size)
    } else {
        Image(bitmap = b.asImageBitmap(), contentDescription = symbol, modifier = Modifier.size(size).clip(CircleShape))
    }
}
