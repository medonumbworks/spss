package com.medonumbworks.csprngpredictor.core;

import java.util.Locale;

/**
 * Tunable parameters of the learner. Plain mutable bean; call {@link #clamp()} after editing.
 * Persisted as "key=value" lines.
 */
public final class ModelParams {

    /** Hedge learning rate (0..5). */
    public double eta = 0.5;
    /** Fixed-share mixing toward uniform weights (0..0.5). */
    public double share = 0.02;
    /** Per-number correction rate (0..1). */
    public double biasRate = 0.1;
    /** Per-number correction decay (0..1). */
    public double biasDecay = 0.05;
    /** "recent10" window (2..200). */
    public int shortWindow = 10;
    /** "recent30" window (2..1000). */
    public int longWindow = 30;
    /** "decay" half-life in draws (1..500). */
    public double halfLife = 15;
    /** Ambiguous dates such as 05/06/2026 are read day-first when true. */
    public boolean dayFirst = true;

    public ModelParams copy() {
        ModelParams p = new ModelParams();
        p.eta = eta;
        p.share = share;
        p.biasRate = biasRate;
        p.biasDecay = biasDecay;
        p.shortWindow = shortWindow;
        p.longWindow = longWindow;
        p.halfLife = halfLife;
        p.dayFirst = dayFirst;
        return p;
    }

    /** "key=value" lines ("\n"), numbers in Locale.US / round-trip exact form. */
    public String serialize() {
        StringBuilder sb = new StringBuilder(160);
        sb.append("eta=").append(formatNumber(eta)).append('\n');
        sb.append("share=").append(formatNumber(share)).append('\n');
        sb.append("biasRate=").append(formatNumber(biasRate)).append('\n');
        sb.append("biasDecay=").append(formatNumber(biasDecay)).append('\n');
        sb.append("shortWindow=").append(shortWindow).append('\n');
        sb.append("longWindow=").append(longWindow).append('\n');
        sb.append("halfLife=").append(formatNumber(halfLife)).append('\n');
        sb.append("dayFirst=").append(dayFirst).append('\n');
        return sb.toString();
    }

    /**
     * Parses {@link #serialize()} output. Unknown, missing or invalid keys keep their
     * defaults; the result is clamped to the valid ranges. Never throws.
     */
    public static ModelParams parse(String s) {
        ModelParams p = new ModelParams();
        if (s == null) {
            return p;
        }
        String[] lines = s.split("[\\r\\n;]+");
        for (String raw : lines) {
            String line = raw.trim();
            int eq = line.indexOf('=');
            if (eq <= 0 || line.startsWith("#")) {
                continue;
            }
            String key = line.substring(0, eq).trim().toLowerCase(Locale.US);
            String value = line.substring(eq + 1).trim();
            if (key.equals("eta")) {
                p.eta = parseDouble(value, p.eta);
            } else if (key.equals("share")) {
                p.share = parseDouble(value, p.share);
            } else if (key.equals("biasrate")) {
                p.biasRate = parseDouble(value, p.biasRate);
            } else if (key.equals("biasdecay")) {
                p.biasDecay = parseDouble(value, p.biasDecay);
            } else if (key.equals("shortwindow")) {
                p.shortWindow = parseInt(value, p.shortWindow);
            } else if (key.equals("longwindow")) {
                p.longWindow = parseInt(value, p.longWindow);
            } else if (key.equals("halflife")) {
                p.halfLife = parseDouble(value, p.halfLife);
            } else if (key.equals("dayfirst")) {
                String v = value.toLowerCase(Locale.US);
                if (v.equals("true") || v.equals("1") || v.equals("yes")) {
                    p.dayFirst = true;
                } else if (v.equals("false") || v.equals("0") || v.equals("no")) {
                    p.dayFirst = false;
                }
            }
        }
        p.clamp();
        return p;
    }

    /** Forces every field into its documented range (NaN -&gt; default). */
    public void clamp() {
        ModelParams d = new ModelParams();
        eta = clamp(eta, 0, 5, d.eta);
        share = clamp(share, 0, 0.5, d.share);
        biasRate = clamp(biasRate, 0, 1, d.biasRate);
        biasDecay = clamp(biasDecay, 0, 1, d.biasDecay);
        shortWindow = Math.max(2, Math.min(200, shortWindow));
        longWindow = Math.max(2, Math.min(1000, longWindow));
        halfLife = clamp(halfLife, 1, 500, d.halfLife);
    }

    private static double clamp(double v, double lo, double hi, double fallback) {
        if (Double.isNaN(v)) {
            return fallback;
        }
        return (v < lo ? lo : (v > hi ? hi : v)) + 0.0;   // + 0.0 turns -0.0 into 0.0
    }

    private static double parseDouble(String v, double fallback) {
        try {
            double d = Double.parseDouble(v.replace(',', '.'));
            return Double.isNaN(d) || Double.isInfinite(d) ? fallback : d;
        } catch (RuntimeException e) {
            return fallback;
        }
    }

    private static int parseInt(String v, int fallback) {
        try {
            return Integer.parseInt(v);
        } catch (RuntimeException e) {
            double d = parseDouble(v, Double.NaN);
            if (Double.isNaN(d)) {
                return fallback;
            }
            return (int) Math.max(Integer.MIN_VALUE, Math.min(Integer.MAX_VALUE, Math.round(d)));
        }
    }

    /** Locale-independent, round-trip exact; integral values without ".0" (e.g. "15", "0.5"). */
    static String formatNumber(double v) {
        if (v == Math.rint(v) && !Double.isInfinite(v) && Math.abs(v) < 1e15) {
            return Long.toString((long) v);
        }
        return Double.toString(v);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ModelParams)) {
            return false;
        }
        ModelParams p = (ModelParams) o;
        return Double.compare(eta, p.eta) == 0
                && Double.compare(share, p.share) == 0
                && Double.compare(biasRate, p.biasRate) == 0
                && Double.compare(biasDecay, p.biasDecay) == 0
                && shortWindow == p.shortWindow
                && longWindow == p.longWindow
                && Double.compare(halfLife, p.halfLife) == 0
                && dayFirst == p.dayFirst;
    }

    @Override
    public int hashCode() {
        int h = hash(eta);
        h = 31 * h + hash(share);
        h = 31 * h + hash(biasRate);
        h = 31 * h + hash(biasDecay);
        h = 31 * h + shortWindow;
        h = 31 * h + longWindow;
        h = 31 * h + hash(halfLife);
        h = 31 * h + (dayFirst ? 1 : 0);
        return h;
    }

    private static int hash(double v) {
        long bits = Double.doubleToLongBits(v);
        return (int) (bits ^ (bits >>> 32));
    }

    @Override
    public String toString() {
        return "ModelParams{eta=" + formatNumber(eta)
                + ", share=" + formatNumber(share)
                + ", biasRate=" + formatNumber(biasRate)
                + ", biasDecay=" + formatNumber(biasDecay)
                + ", shortWindow=" + shortWindow
                + ", longWindow=" + longWindow
                + ", halfLife=" + formatNumber(halfLife)
                + ", dayFirst=" + dayFirst + "}";
    }
}
