package dev.cryptopulse.core.analysis.sentiment

import java.math.BigDecimal
import java.math.RoundingMode
import java.util.Locale

/**
 * The handful of Python 3 `str`/`float` semantics that vaderSentiment depends on, reproduced exactly so
 * the Kotlin port tokenizes, cases and rounds like CPython (3.12+).
 */
internal object PyStr {
    /** Python's `string.punctuation`, which `SentiText._strip_punc_if_word` strips. */
    const val PUNCTUATION: String = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    private val isPunct = BooleanArray(128).also { table -> PUNCTUATION.forEach { table[it.code] = true } }

    /**
     * Java regex character-class body that matches exactly what Python treats as whitespace
     * (`str.isspace()`, `str.split()`, `str.strip()` and `\s` in `re` str patterns).
     */
    const val WHITESPACE_CLASS: String =
        "\\t\\n\\x0B\\f\\r\\x1C-\\x1F \\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000"

    /** Python `str.isspace()` for one character; every Python whitespace character is in the BMP. */
    fun isSpace(c: Char): Boolean = when (c) {
        '\t', '\n', '\u000B', '\u000C', '\r', '\u001C', '\u001D', '\u001E', '\u001F', ' ',
        '\u0085', ' ', ' ', ' ', ' ', ' ', ' ', '　',
        -> true
        else -> c in ' '..' '
    }

    fun isPunctuation(c: Char): Boolean = c.code < 128 && isPunct[c.code]

    /** Python `str.split()` without arguments: split on runs of whitespace, drop empty pieces. */
    fun split(s: String): List<String> {
        val out = ArrayList<String>()
        var i = 0
        val n = s.length
        while (i < n) {
            while (i < n && isSpace(s[i])) i++
            if (i >= n) break
            val start = i
            while (i < n && !isSpace(s[i])) i++
            out.add(s.substring(start, i))
        }
        return out
    }

    /** Python `str.strip()` without arguments. */
    fun strip(s: String): String {
        var start = 0
        var end = s.length
        while (start < end && isSpace(s[start])) start++
        while (end > start && isSpace(s[end - 1])) end--
        return if (start == 0 && end == s.length) s else s.substring(start, end)
    }

    /** Python `" ".join(s.split())`: collapse whitespace runs to one space and trim. */
    fun collapseWhitespace(s: String): String {
        val sb = StringBuilder(s.length)
        var pendingSpace = false
        for (c in s) {
            if (isSpace(c)) {
                pendingSpace = sb.isNotEmpty()
            } else {
                if (pendingSpace) sb.append(' ')
                pendingSpace = false
                sb.append(c)
            }
        }
        return sb.toString()
    }

    /**
     * Python `str.isupper()`: no lowercase or titlecase character and at least one uppercase one.
     * Works on code points, so supplementary characters (e.g. mathematical bold letters) count.
     */
    fun isUpper(s: String): Boolean {
        var cased = false
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            if (Character.isLowerCase(cp) || Character.isTitleCase(cp)) return false
            if (!cased && Character.isUpperCase(cp)) cased = true
            i += Character.charCount(cp)
        }
        return cased
    }

    /** Python `str.lower()` (full Unicode mapping, locale independent). */
    fun lower(s: String): String = s.lowercase(Locale.ROOT)

    /** Python `len(str)`: number of code points. */
    fun len(s: String): Int = s.codePointCount(0, s.length)

    /** Python `round(x, digits)`: exact binary value rounded half-to-even; keeps the sign of zero. */
    fun round(x: Double, digits: Int): Double {
        val rounded = BigDecimal(x).setScale(digits, RoundingMode.HALF_EVEN).toDouble()
        return if (rounded == 0.0) Math.copySign(0.0, x) else rounded
    }

    /**
     * Python's built-in `sum()` over floats. CPython 3.12+ uses Neumaier compensated summation; ints
     * (VADER appends `0`) behave like `0.0` here.
     */
    fun sum(values: DoubleArray): Double {
        var total = 0.0
        var compensation = 0.0
        for (x in values) {
            val t = total + x
            compensation += if (Math.abs(total) >= Math.abs(x)) (total - t) + x else (x - t) + total
            total = t
        }
        if (compensation != 0.0 && compensation.isFinite()) total += compensation
        return total
    }
}
