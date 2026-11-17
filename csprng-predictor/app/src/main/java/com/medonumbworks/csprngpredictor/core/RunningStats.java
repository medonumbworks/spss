package com.medonumbworks.csprngpredictor.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Mutable, incrementally updated statistics over a growing chronological list of draws.
 *
 * <p>{@link Replay} pushes one draw per step and hands a cheap {@link History} view to the
 * strategies, so a full walk-forward replay costs O(N * 22 * 11) instead of O(N^2).
 * Every statistic is available for the current size (fast, from the running state) and for
 * any earlier prefix size (slow path, recomputed from the stored draws) so that a History
 * view stays correct even if it is used after more draws were pushed.
 *
 * <p>Decayed counts are always computed with the same recurrence (v = v * f + x per draw)
 * on both paths, so results are bit-for-bit identical whichever path produced them.
 */
final class RunningStats {

    private static final int N = Game.MAX_NUMBER;
    private static final int MAX_TRACKERS = 8;

    private final ArrayList<Draw> draws = new ArrayList<Draw>();
    private final int[] counts = new int[N];
    private final int[][] trans = new int[N][N];
    private final int[] containsCounts = new int[N];
    private final int[][] pairs = new int[N][N];
    private final int[] lastSeen = new int[N];
    private long overlapSum;

    private final ArrayList<WindowTracker> windows = new ArrayList<WindowTracker>();
    private final ArrayList<DecayTracker> decays = new ArrayList<DecayTracker>();

    private static final class WindowTracker {
        final int window;
        final int[] counts;

        WindowTracker(int window, int[] counts) {
            this.window = window;
            this.counts = counts;
        }
    }

    private static final class DecayTracker {
        final double halfLife;
        final double factor;
        final double[] values;

        DecayTracker(double halfLife, double factor, double[] values) {
            this.halfLife = halfLife;
            this.factor = factor;
            this.values = values;
        }
    }

    RunningStats() {
        for (int i = 0; i < N; i++) {
            lastSeen[i] = -1;
        }
    }

    RunningStats(List<Draw> chronological) {
        this();
        if (chronological != null) {
            for (Draw d : chronological) {
                if (d != null) {
                    push(d);
                }
            }
        }
    }

    int size() {
        return draws.size();
    }

    Draw draw(int i) {
        return draws.get(i);
    }

    /** A read-only view of the current state. */
    History view() {
        return new History(this, draws.size());
    }

    /** Appends the next draw (chronological order) and updates every statistic in O(22 * 11). */
    void push(Draw d) {
        int t = draws.size();
        int[] nums = d.numbers;
        if (t >= 1) {
            int[] prev = draws.get(t - 1).numbers;
            for (int a : prev) {
                containsCounts[a - 1]++;
                int[] row = trans[a - 1];
                for (int b : nums) {
                    row[b - 1]++;
                }
            }
            overlapSum += Long.bitCount(Draw.mask(prev) & Draw.mask(nums));
        }
        for (int a : nums) {
            counts[a - 1]++;
            lastSeen[a - 1] = t;
            int[] row = pairs[a - 1];
            for (int b : nums) {
                row[b - 1]++;                 // diagonal = how often the number itself appeared
            }
        }
        draws.add(d);
        int size = t + 1;
        for (WindowTracker w : windows) {
            for (int a : nums) {
                w.counts[a - 1]++;
            }
            if (size > w.window) {
                for (int a : draws.get(size - 1 - w.window).numbers) {
                    w.counts[a - 1]--;
                }
            }
        }
        for (DecayTracker dt : decays) {
            step(dt.values, dt.factor, nums);
        }
    }

    // ------------------------------------------------------------------ statistics for prefix [0, n)

    int[] counts(int n) {
        if (n == draws.size()) {
            return counts.clone();
        }
        int[] c = new int[N];
        for (int t = 0; t < n; t++) {
            for (int a : draws.get(t).numbers) {
                c[a - 1]++;
            }
        }
        return c;
    }

    int[] countsInLast(int n, int window) {
        if (window <= 0 || n == 0) {
            return new int[N];
        }
        if (window >= n) {
            return counts(n);
        }
        if (n == draws.size()) {
            for (WindowTracker w : windows) {
                if (w.window == window) {
                    return w.counts.clone();
                }
            }
            int[] c = windowCounts(n, window);
            if (windows.size() < MAX_TRACKERS) {
                windows.add(new WindowTracker(window, c.clone()));
            }
            return c;
        }
        return windowCounts(n, window);
    }

    private int[] windowCounts(int n, int window) {
        int[] c = new int[N];
        for (int t = Math.max(0, n - window); t < n; t++) {
            for (int a : draws.get(t).numbers) {
                c[a - 1]++;
            }
        }
        return c;
    }

    double[] decayedCounts(int n, double halfLife) {
        double hl = halfLife;
        if (!(hl > 1e-6)) {
            hl = 1e-6;                                    // also catches NaN
        }
        double f = Double.isInfinite(hl) ? 1.0 : StrictMath.pow(0.5, 1.0 / hl);
        if (n == draws.size()) {
            for (DecayTracker dt : decays) {
                if (dt.halfLife == hl) {
                    return dt.values.clone();
                }
            }
            double[] v = decayFromScratch(n, f);
            if (decays.size() < MAX_TRACKERS) {
                decays.add(new DecayTracker(hl, f, v.clone()));
            }
            return v;
        }
        return decayFromScratch(n, f);
    }

    private double[] decayFromScratch(int n, double f) {
        double[] v = new double[N];
        for (int t = 0; t < n; t++) {
            step(v, f, draws.get(t).numbers);
        }
        return v;
    }

    private static void step(double[] v, double f, int[] nums) {
        for (int i = 0; i < N; i++) {
            v[i] *= f;
        }
        for (int a : nums) {
            v[a - 1] += 1.0;
        }
    }

    int[] drawsSinceSeen(int n) {
        int[] s = new int[N];
        if (n == draws.size()) {
            for (int i = 0; i < N; i++) {
                s[i] = lastSeen[i] < 0 ? n : (n - 1 - lastSeen[i]);
            }
            return s;
        }
        int found = 0;
        for (int i = 0; i < N; i++) {
            s[i] = -1;
        }
        for (int t = n - 1; t >= 0 && found < N; t--) {
            for (int a : draws.get(t).numbers) {
                if (s[a - 1] < 0) {
                    s[a - 1] = n - 1 - t;
                    found++;
                }
            }
        }
        for (int i = 0; i < N; i++) {
            if (s[i] < 0) {
                s[i] = n;
            }
        }
        return s;
    }

    int[][] transitionCounts(int n) {
        if (n == draws.size()) {
            return copy(trans);
        }
        int[][] m = new int[N][N];
        for (int t = 1; t < n; t++) {
            int[] next = draws.get(t).numbers;
            for (int a : draws.get(t - 1).numbers) {
                for (int b : next) {
                    m[a - 1][b - 1]++;
                }
            }
        }
        return m;
    }

    int[] containsCounts(int n) {
        if (n == draws.size()) {
            return containsCounts.clone();
        }
        int[] c = new int[N];
        for (int t = 0; t + 1 < n; t++) {
            for (int a : draws.get(t).numbers) {
                c[a - 1]++;
            }
        }
        return c;
    }

    int[][] pairCounts(int n) {
        if (n == draws.size()) {
            return copy(pairs);
        }
        int[][] m = new int[N][N];
        for (int t = 0; t < n; t++) {
            int[] nums = draws.get(t).numbers;
            for (int a : nums) {
                for (int b : nums) {
                    m[a - 1][b - 1]++;
                }
            }
        }
        return m;
    }

    double repeatRate(int n) {
        if (n < 2) {
            return 0.5;
        }
        long sum;
        if (n == draws.size()) {
            sum = overlapSum;
        } else {
            sum = 0;
            for (int t = 1; t < n; t++) {
                sum += Long.bitCount(Draw.mask(draws.get(t - 1).numbers) & Draw.mask(draws.get(t).numbers));
            }
        }
        return sum / (Game.PICK * (double) (n - 1));
    }

    private static int[][] copy(int[][] m) {
        int[][] c = new int[m.length][];
        for (int i = 0; i < m.length; i++) {
            c[i] = m[i].clone();
        }
        return c;
    }
}
