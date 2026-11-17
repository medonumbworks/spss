package com.medonumbworks.csprngpredictor.core;

import java.util.Calendar;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Date helpers. Every date inside the app is an ISO string {@code yyyy-MM-dd}.
 *
 * <p>Calendar arithmetic uses exact proleptic-Gregorian day numbers (the same result as
 * {@link java.util.GregorianCalendar} for all modern dates, but without time zone / DST
 * pitfalls and much faster when assigning dates to hundreds of thousands of rows).
 */
public final class DateUtil {

    /** Smallest year accepted by {@link #normalize}. */
    public static final int MIN_YEAR = 1900;
    /** Largest year accepted by {@link #normalize}. */
    public static final int MAX_YEAR = 2100;

    /** yyyy-MM-dd / yyyy/MM/dd / yyyy.MM.dd (1- or 2-digit month/day accepted). */
    private static final Pattern YMD =
            Pattern.compile("^([0-9]{4})([-/.])([0-9]{1,2})\\2([0-9]{1,2})$");
    /** a.b.yyyy / a-b-yyyy / a/b/yyyy and 2-digit-year variants. */
    private static final Pattern DMY =
            Pattern.compile("^([0-9]{1,2})([-/.])([0-9]{1,2})\\2([0-9]{4}|[0-9]{2})$");
    /** "16. 10. 2026." style (spaces after the dots, optional trailing dot, 4-digit year). */
    private static final Pattern DMY_DOTS =
            Pattern.compile("^([0-9]{1,2})\\.\\s*([0-9]{1,2})\\.\\s*([0-9]{4})\\.?$");
    /** Start of a time part in a date-time ("T20:00", " 20:00"). */
    private static final Pattern TIME_SUFFIX = Pattern.compile("(?:T|\\s+)[0-9]{1,2}:[0-9]{2}");
    /** Shape of a date with a 4-digit year, valid or not (e.g. "31.02.2026", "2026-13-01"). */
    private static final Pattern DATE_SHAPE = Pattern.compile(
            "^(?:[0-9]{4}([-/.])[0-9]{1,2}\\1[0-9]{1,2}|[0-9]{1,2}([-/.])\\s?[0-9]{1,2}\\2\\s?[0-9]{4}\\.?)"
                    + "(?:(?:T|\\s+)[0-9]{1,2}:[0-9]{2}.*)?$");

    /**
     * Date-like substrings inside free text. The look-arounds stop matches inside longer
     * digit/separator chains such as "1-4-7-9-11-13" (a list of draw numbers, not a date).
     */
    private static final Pattern FIND = Pattern.compile(
            "(?<![0-9])(?<![0-9][-/.])(?:"
                    + "[0-9]{4}([-/.])[0-9]{1,2}\\1[0-9]{1,2}"
                    + "(?:(?:T|[ ]+)[0-9]{1,2}:[0-9]{2}(?:[:][0-9]{2}(?:[.,][0-9]+)?)?(?:Z|[+-][0-9]{2}:?[0-9]{2})?)?"
                    + "|[0-9]{1,2}([-/])[0-9]{1,2}\\2(?:[0-9]{4}|[0-9]{2})"
                    + "|[0-9]{1,2}\\.[ ]?[0-9]{1,2}\\.[ ]?[0-9]{4}\\.?"
                    + "|[0-9]{1,2}\\.[0-9]{1,2}\\.[0-9]{2}"
                    + "|[0-9]{8}"
                    + ")(?![0-9])(?![-/.][0-9])");

    private DateUtil() {
    }

    /** A date found inside a longer text: normalized ISO value plus the matched span. */
    static final class DateMatch {
        final String iso;
        final int start;
        final int end;

        DateMatch(String iso, int start, int end) {
            this.iso = iso;
            this.start = start;
            this.end = end;
        }
    }

    /** Today's local date as ISO. */
    public static String today() {
        Calendar c = Calendar.getInstance();
        return format(c.get(Calendar.YEAR), c.get(Calendar.MONTH) + 1, c.get(Calendar.DAY_OF_MONTH));
    }

    /** Strict {@code yyyy-MM-dd} (zero padded) that is a real calendar date. */
    public static boolean isIso(String s) {
        if (s == null || s.length() != 10 || s.charAt(4) != '-' || s.charAt(7) != '-') {
            return false;
        }
        for (int i = 0; i < 10; i++) {
            if (i != 4 && i != 7 && !isDigit(s.charAt(i))) {
                return false;
            }
        }
        int y = parseDigits(s, 0, 4);
        int m = parseDigits(s, 5, 7);
        int d = parseDigits(s, 8, 10);
        return y >= 1 && isValidDate(y, m, d);
    }

    /**
     * ISO date {@code days} after (negative: before) {@code iso}.
     *
     * @throws IllegalArgumentException when {@code iso} is not a valid ISO date
     */
    public static String addDays(String iso, int days) {
        long epoch = toEpochDay(requireIso(iso));
        return fromEpochDay(epoch + days);
    }

    /**
     * {@code isoB - isoA} in days.
     *
     * @throws IllegalArgumentException when either argument is not a valid ISO date
     */
    public static int daysBetween(String isoA, String isoB) {
        long a = toEpochDay(requireIso(isoA));
        long b = toEpochDay(requireIso(isoB));
        return (int) (b - a);
    }

    /** Negative, zero or positive (-1/0/1); ISO strings compare lexicographically. null sorts first. */
    public static int compare(String isoA, String isoB) {
        if (isoA == null) {
            return isoB == null ? 0 : -1;
        }
        if (isoB == null) {
            return 1;
        }
        int c = isoA.compareTo(isoB);
        return c < 0 ? -1 : (c > 0 ? 1 : 0);
    }

    /**
     * Converts one date token to ISO, or returns null.
     *
     * <p>Accepted (trimmed, optional surrounding quotes): yyyy-MM-dd, yyyy/MM/dd, yyyy.MM.dd,
     * yyyyMMdd, a.b.yyyy, a-b-yyyy, a/b/yyyy, the same with 2-digit years (20yy),
     * "16. 10. 2026." and ISO date-times (the time part is dropped). For a/b/year: if a &gt; 12
     * the token is day-first; else if b &gt; 12 it is month-first; else {@code dayFirst} decides.
     * The result must be a real calendar date with a year in 1900..2100.
     */
    public static String normalize(String token, boolean dayFirst) {
        if (token == null) {
            return null;
        }
        String s = stripQuotes(token.trim());
        int len = s.length();
        if (len < 6 || len > 48 || !isDigit(s.charAt(0))) {
            return null;
        }
        boolean allDigits = true;
        int digits = 0;
        for (int i = 0; i < len; i++) {
            if (isDigit(s.charAt(i))) {
                digits++;
            } else {
                allDigits = false;
            }
        }
        if (digits < 4) {
            return null;
        }
        if (allDigits) {
            if (len != 8) {
                return null;
            }
            return build(parseDigits(s, 0, 4), parseDigits(s, 4, 6), parseDigits(s, 6, 8));
        }
        if (len == 10 && digits == 8 && s.charAt(4) == '-' && s.charAt(7) == '-') {
            // fast path for the most common form, plain ISO (no regex)
            return build(parseDigits(s, 0, 4), parseDigits(s, 5, 7), parseDigits(s, 8, 10));
        }
        if (s.indexOf(':') > 0) {
            Matcher tm = TIME_SUFFIX.matcher(s);
            if (tm.find()) {
                if (tm.start() == 0) {
                    return null;
                }
                s = s.substring(0, tm.start()).trim();
            }
        }
        Matcher m = YMD.matcher(s);
        if (m.matches()) {
            return build(Integer.parseInt(m.group(1)), Integer.parseInt(m.group(3)),
                    Integer.parseInt(m.group(4)));
        }
        m = DMY.matcher(s);
        if (m.matches()) {
            return ambiguous(Integer.parseInt(m.group(1)), Integer.parseInt(m.group(3)),
                    year(m.group(4)), dayFirst);
        }
        m = DMY_DOTS.matcher(s);
        if (m.matches()) {
            return ambiguous(Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)),
                    year(m.group(3)), dayFirst);
        }
        return null;
    }

    /** First date-like substring of {@code text}, normalized to ISO; null if there is none. */
    public static String findDate(String text, boolean dayFirst) {
        DateMatch m = findDateMatch(text, dayFirst);
        return m == null ? null : m.iso;
    }

    /** "dd.MM.yyyy" for the UI; non-ISO input is returned unchanged ("" for null). */
    public static String display(String iso) {
        if (iso == null) {
            return "";
        }
        if (!isIso(iso)) {
            return iso;
        }
        return iso.substring(8, 10) + "." + iso.substring(5, 7) + "." + iso.substring(0, 4);
    }

    // ------------------------------------------------------------------ package helpers

    /**
     * True when the token has the shape of a date with a 4-digit year, whether or not it is
     * a real date ({@link #normalize} decides that). Used to keep the digits of broken dates
     * such as "31.02.2026" out of the draw numbers.
     */
    static boolean looksLikeDate(String token) {
        if (token == null) {
            return false;
        }
        String s = stripQuotes(token.trim());
        int len = s.length();
        return len >= 8 && len <= 48 && isDigit(s.charAt(0)) && DATE_SHAPE.matcher(s).matches();
    }

    /** Like {@link #findDate} but also returns where the date was found. */
    static DateMatch findDateMatch(String text, boolean dayFirst) {
        if (text == null || !hasDigits(text, 4)) {
            return null;
        }
        Matcher m = FIND.matcher(text);
        int from = 0;
        int len = text.length();
        while (from < len && m.find(from)) {
            String iso = normalize(m.group(), dayFirst);
            if (iso != null) {
                return new DateMatch(iso, m.start(), m.end());
            }
            from = m.start() + 1;
        }
        return null;
    }

    /** {@code text} with every recognisable date replaced by a space (so its digits never count as numbers). */
    static String removeDates(String text, boolean dayFirst) {
        if (text == null || !hasDigits(text, 4)) {
            return text;
        }
        StringBuilder sb = null;
        int copied = 0;
        Matcher m = FIND.matcher(text);
        int from = 0;
        int len = text.length();
        while (from < len && m.find(from)) {
            if (normalize(m.group(), dayFirst) != null) {
                if (sb == null) {
                    sb = new StringBuilder(len);
                }
                sb.append(text, copied, m.start()).append(' ');
                copied = m.end();
                from = m.end();
            } else {
                from = m.start() + 1;
            }
        }
        if (sb == null) {
            return text;
        }
        sb.append(text, copied, len);
        return sb.toString();
    }

    static boolean isValidDate(int y, int m, int d) {
        return m >= 1 && m <= 12 && d >= 1 && d <= daysInMonth(y, m);
    }

    static int daysInMonth(int y, int m) {
        switch (m) {
            case 2:
                return isLeap(y) ? 29 : 28;
            case 4:
            case 6:
            case 9:
            case 11:
                return 30;
            default:
                return 31;
        }
    }

    static boolean isLeap(int y) {
        return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    }

    static String format(int y, int m, int d) {
        StringBuilder sb = new StringBuilder(10);
        if (y < 1000) {
            sb.append(y < 10 ? "000" : (y < 100 ? "00" : "0"));
        }
        sb.append(y).append('-');
        if (m < 10) {
            sb.append('0');
        }
        sb.append(m).append('-');
        if (d < 10) {
            sb.append('0');
        }
        sb.append(d);
        return sb.toString();
    }

    // ------------------------------------------------------------------ private helpers

    private static String requireIso(String iso) {
        String s = iso == null ? null : iso.trim();
        if (s == null || !isIso(s)) {
            throw new IllegalArgumentException("Not an ISO date (yyyy-MM-dd): " + iso);
        }
        return s;
    }

    private static String ambiguous(int a, int b, int y, boolean dayFirst) {
        int day;
        int month;
        if (a > 12) {
            day = a;
            month = b;
        } else if (b > 12) {
            month = a;
            day = b;
        } else if (dayFirst) {
            day = a;
            month = b;
        } else {
            month = a;
            day = b;
        }
        return build(y, month, day);
    }

    private static int year(String digits) {
        int v = Integer.parseInt(digits);
        return digits.length() == 2 ? 2000 + v : v;
    }

    private static String build(int y, int m, int d) {
        if (y < MIN_YEAR || y > MAX_YEAR || !isValidDate(y, m, d)) {
            return null;
        }
        return format(y, m, d);
    }

    /** Removes surrounding whitespace and quote characters (also unbalanced ones from naive CSV splits). */
    private static String stripQuotes(String s) {
        int a = 0;
        int b = s.length();
        while (true) {
            while (a < b && Character.isWhitespace(s.charAt(a))) {
                a++;
            }
            while (b > a && Character.isWhitespace(s.charAt(b - 1))) {
                b--;
            }
            boolean changed = false;
            if (a < b && isQuote(s.charAt(a))) {
                a++;
                changed = true;
            }
            if (b > a && isQuote(s.charAt(b - 1))) {
                b--;
                changed = true;
            }
            if (!changed) {
                break;
            }
        }
        return s.substring(a, b);
    }

    private static boolean isQuote(char c) {
        return c == '"' || c == '\'';
    }

    private static boolean hasDigits(String s, int min) {
        int n = 0;
        for (int i = 0, len = s.length(); i < len; i++) {
            if (isDigit(s.charAt(i)) && ++n >= min) {
                return true;
            }
        }
        return false;
    }

    private static boolean isDigit(char c) {
        return c >= '0' && c <= '9';
    }

    private static int parseDigits(String s, int from, int to) {
        int v = 0;
        for (int i = from; i < to; i++) {
            v = v * 10 + (s.charAt(i) - '0');
        }
        return v;
    }

    /** Days since 1970-01-01 (proleptic Gregorian; H. Hinnant's days_from_civil). */
    private static long toEpochDay(String iso) {
        long y = parseDigits(iso, 0, 4);
        int m = parseDigits(iso, 5, 7);
        int d = parseDigits(iso, 8, 10);
        y -= m <= 2 ? 1 : 0;
        long era = (y >= 0 ? y : y - 399) / 400;
        long yoe = y - era * 400;
        long doy = (153L * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
        long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        return era * 146097 + doe - 719468;
    }

    private static String fromEpochDay(long epochDay) {
        long z = epochDay + 719468;
        long era = (z >= 0 ? z : z - 146096) / 146097;
        long doe = z - era * 146097;
        long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        long y = yoe + era * 400;
        long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        long mp = (5 * doy + 2) / 153;
        long d = doy - (153 * mp + 2) / 5 + 1;
        long m = mp + (mp < 10 ? 3 : -9);
        if (m <= 2) {
            y++;
        }
        return format((int) y, (int) m, (int) d);
    }
}
