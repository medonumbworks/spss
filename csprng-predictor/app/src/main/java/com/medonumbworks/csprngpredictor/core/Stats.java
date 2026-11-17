package com.medonumbworks.csprngpredictor.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Accuracy statistics of a series of 11-number predictions, compared with random guessing.
 *
 * <p>For a random 11-of-22 guess the hit count is hypergeometric: mean 5.5, variance
 * 11 * 0.5 * 0.5 * 11/21 ~ 1.4405 (SD ~ 1.2).
 */
public final class Stats {

    /** C(22, 11). */
    private static final long TOTAL = 705432L;
    /** C(11, k) for k = 0..11. */
    private static final long[] C11 = {1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1};

    /** Evaluated predictions. */
    public final int n;
    /** 0 when n == 0. */
    public final double meanHits;
    /** Sample SD of hits (0 when n &lt; 2). */
    public final double sd;
    /** meanHits - 5.5 (0 when n == 0). */
    public final double edge;
    /** (meanHits - 5.5) / (1.2 / sqrt(n)); 0 when n == 0. */
    public final double z;
    /** Length 12, index = hits. */
    public final int[] histogram;
    /** Best / worst hit count; -1 when n == 0. */
    public final int best;
    public final int worst;
    /** Mean over the last min(10, n) predictions (0 when n == 0). */
    public final double lastTenMean;

    private Stats(int[] hits) {
        int count = hits.length;
        int[] hist = new int[Game.PICK + 1];
        long sum = 0;
        int b = -1;
        int w = -1;
        for (int h : hits) {
            hist[h]++;
            sum += h;
            if (b < 0 || h > b) {
                b = h;
            }
            if (w < 0 || h < w) {
                w = h;
            }
        }
        this.n = count;
        this.histogram = hist;
        this.best = b;
        this.worst = w;
        if (count == 0) {
            meanHits = 0.0;
            sd = 0.0;
            edge = 0.0;
            z = 0.0;
            lastTenMean = 0.0;
            return;
        }
        double mean = sum / (double) count;
        double ss = 0.0;
        for (int h : hits) {
            double d = h - mean;
            ss += d * d;
        }
        meanHits = mean;
        sd = count < 2 ? 0.0 : Math.sqrt(ss / (count - 1));
        edge = mean - Game.EXPECTED_HITS;
        z = (mean - Game.EXPECTED_HITS) / (1.2 / Math.sqrt(count));
        int m = Math.min(10, count);
        long last = 0;
        for (int i = count - m; i < count; i++) {
            last += hits[i];
        }
        lastTenMean = last / (double) m;
    }

    /** Statistics over replay records (in order). */
    public static Stats of(List<PredictionRecord> records) {
        if (records == null) {
            return new Stats(new int[0]);
        }
        int[] h = new int[records.size()];
        int c = 0;
        for (PredictionRecord r : records) {
            if (r != null) {
                h[c++] = clampHits(r.hits);
            }
        }
        return new Stats(trim(h, c));
    }

    /** Statistics over plain hit counts (values outside 0..11, e.g. -1 = pending, are ignored). */
    public static Stats ofHits(int[] hits) {
        if (hits == null) {
            return new Stats(new int[0]);
        }
        int[] h = new int[hits.length];
        int c = 0;
        for (int v : hits) {
            if (v >= 0 && v <= Game.PICK) {
                h[c++] = v;
            }
        }
        return new Stats(trim(h, c));
    }

    /** Statistics over plain hit counts (null and values outside 0..11 are ignored). */
    public static Stats ofHits(List<Integer> hits) {
        if (hits == null) {
            return new Stats(new int[0]);
        }
        int[] h = new int[hits.size()];
        int c = 0;
        for (Integer v : hits) {
            if (v != null && v.intValue() >= 0 && v.intValue() <= Game.PICK) {
                h[c++] = v.intValue();
            }
        }
        return new Stats(trim(h, c));
    }

    /** Statistics of nothing (n == 0). */
    public static Stats empty() {
        return new Stats(new int[0]);
    }

    /** P(hits == k) for a random 11-of-22 pick, k in 0..11 (0 otherwise). */
    public static double hypergeometricPmf(int k) {
        if (k < 0 || k > Game.PICK) {
            return 0.0;
        }
        return (C11[k] * C11[Game.PICK - k]) / (double) TOTAL;
    }

    /** P(hits &gt;= k) for a random pick; 1 for k &lt;= 0, 0 for k &gt; 11. */
    public static double hypergeometricTail(int k) {
        if (k <= 0) {
            return 1.0;
        }
        if (k > Game.PICK) {
            return 0.0;
        }
        long s = 0;
        for (int j = k; j <= Game.PICK; j++) {
            s += C11[j] * C11[Game.PICK - j];
        }
        return s / (double) TOTAL;
    }

    /** 5.5. */
    public static double expectedHits() {
        return Game.EXPECTED_HITS;
    }

    /** 11 * 0.5 * 0.5 * 11 / 21 ~ 1.4405. */
    public static double variance() {
        return 11.0 * 0.5 * 0.5 * 11.0 / 21.0;
    }

    /** One-sided probability of a mean at least this high under pure chance (normal approximation). */
    public double chanceOfLuck() {
        if (n == 0) {
            return 1.0;
        }
        return 1.0 - normalCdf(z);
    }

    /** Share of predictions with exactly {@code k} hits (0 when n == 0). */
    public double fraction(int k) {
        if (n == 0 || k < 0 || k > Game.PICK) {
            return 0.0;
        }
        return histogram[k] / (double) n;
    }

    /** Multi-line, plain-text summary for the UI (Locale.US number formatting). */
    public String summaryText() {
        if (n == 0) {
            return "No predictions evaluated yet.\n"
                    + "Add at least two draws: every draw after the first is predicted from the"
                    + " draws before it and then scored.";
        }
        StringBuilder sb = new StringBuilder(512);
        sb.append(String.format(Locale.US, "Evaluated predictions: %d\n", n));
        sb.append(String.format(Locale.US, "Mean hits: %.2f / 11 (random guessing: %.2f)\n",
                meanHits, Game.EXPECTED_HITS));
        sb.append(String.format(Locale.US, "Edge over random: %+.2f hits per draw\n", edge));
        sb.append(String.format(Locale.US, "Standard deviation: %.2f (random: %.2f)\n",
                sd, Math.sqrt(variance())));
        sb.append(String.format(Locale.US, "z-score: %+.2f\n", z));
        sb.append(String.format(Locale.US,
                "Chance of doing at least this well by pure luck: %.1f%%\n", 100.0 * chanceOfLuck()));
        sb.append(String.format(Locale.US, "Best: %d hits, worst: %d hits\n", best, worst));
        sb.append(String.format(Locale.US, "Last %d predictions: mean %.2f hits\n",
                Math.min(10, n), lastTenMean));
        sb.append(verdict());
        return sb.toString();
    }

    /** One-sentence interpretation of the z-score. */
    public String verdict() {
        if (n == 0) {
            return "No predictions evaluated yet.";
        }
        if (n < 30) {
            return "Too few predictions for a reliable verdict (at least 30 are needed).";
        }
        if (z >= 3.0) {
            return "Clearly better than random (z >= 3): the source may not be a true CSPRNG,"
                    + " or the data contains a pattern.";
        }
        if (z <= -3.0) {
            return "Clearly worse than random (z <= -3): unusual - check the data for mistakes.";
        }
        return "Consistent with a truly random source: no exploitable pattern detected.";
    }

    /** Standard normal CDF (Abramowitz-Stegun 7.1.26 erf, |error| &lt; 1.5e-7); deterministic. */
    static double normalCdf(double x) {
        double t = x / Math.sqrt(2.0);
        double a = Math.abs(t);
        double k = 1.0 / (1.0 + 0.3275911 * a);
        double poly = k * (0.254829592 + k * (-0.284496736 + k * (1.421413741
                + k * (-1.453152027 + k * 1.061405429))));
        double erf = 1.0 - poly * StrictMath.exp(-a * a);
        if (t < 0) {
            erf = -erf;
        }
        return 0.5 * (1.0 + erf);
    }

    private static int clampHits(int h) {
        return h < 0 ? 0 : (h > Game.PICK ? Game.PICK : h);
    }

    private static int[] trim(int[] a, int n) {
        if (n == a.length) {
            return a;
        }
        int[] out = new int[n];
        System.arraycopy(a, 0, out, 0, n);
        return out;
    }

    /** Hit counts of the records, in order (helper for UI charts). */
    public static List<Integer> hitsOf(List<PredictionRecord> records) {
        List<Integer> out = new ArrayList<Integer>();
        if (records != null) {
            for (PredictionRecord r : records) {
                if (r != null) {
                    out.add(Integer.valueOf(r.hits));
                }
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return String.format(Locale.US, "Stats{n=%d, mean=%.3f, sd=%.3f, z=%.2f}", n, meanHits, sd, z);
    }
}
