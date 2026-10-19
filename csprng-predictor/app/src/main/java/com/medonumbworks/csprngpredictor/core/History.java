package com.medonumbworks.csprngpredictor.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Immutable read-only view of {@code draws[0..size())}. Statistics are computed lazily and
 * cached. Views created by {@link Replay} share a {@link RunningStats} with the replay loop,
 * so obtaining a statistic for the current step is O(22 * 22) at most.
 *
 * <p>The public getters return copies; the package-private {@code ...Ref()} getters used by
 * the built-in strategies return the cached arrays directly (they must not be modified).
 * Not thread-safe.
 */
public final class History {

    private final RunningStats rs;
    private final int n;

    private int[] counts;
    private int[] since;
    private int[][] trans;
    private int[] containsC;
    private int[][] pairs;
    private double repeat = Double.NaN;
    private final ArrayList<int[]> windowCounts = new ArrayList<int[]>(2);
    private final ArrayList<Integer> windowKeys = new ArrayList<Integer>(2);
    private final ArrayList<double[]> decayValues = new ArrayList<double[]>(1);
    private final ArrayList<Double> decayKeys = new ArrayList<Double>(1);

    /** History over the given chronological draws (may be empty or null; null entries are skipped). */
    public History(List<Draw> drawsChronological) {
        this.rs = new RunningStats(drawsChronological);
        this.n = rs.size();
    }

    History(RunningStats rs, int n) {
        this.rs = rs;
        this.n = n;
    }

    public int size() {
        return n;
    }

    public Draw get(int i) {
        if (i < 0 || i >= n) {
            throw new IndexOutOfBoundsException("index " + i + ", size " + n);
        }
        return rs.draw(i);
    }

    /** The most recent draw, null when empty. */
    public Draw last() {
        return n == 0 ? null : rs.draw(n - 1);
    }

    /** All-time counts per number (length 22). */
    public int[] counts() {
        return countsRef().clone();
    }

    /** Counts per number over the last {@code window} draws (all draws if window &gt;= size). */
    public int[] countsInLast(int window) {
        return countsInLastRef(window).clone();
    }

    /** Sum over draws of 0.5^(age / halfLife), age 0 = last draw. */
    public double[] decayedCounts(double halfLifeDraws) {
        return decayedRef(halfLifeDraws).clone();
    }

    /** Draws since each number last appeared (0 = in the last draw); size() if never seen. */
    public int[] drawsSinceSeen() {
        return sinceRef().clone();
    }

    /** T[i][j] = times number j+1 appeared in the draw following a draw containing i+1. */
    public int[][] transitionCounts() {
        return copy(transRef());
    }

    /** C[i] = times number i+1 appeared in a draw that has a successor. */
    public int[] containsCounts() {
        return containsRef().clone();
    }

    /** Mean over t &gt;= 1 of |draw[t] intersect draw[t-1]| / 11; 0.5 when fewer than 2 draws. */
    public double repeatRate() {
        if (Double.isNaN(repeat)) {
            repeat = rs.repeatRate(n);
        }
        return repeat;
    }

    /**
     * P[i][j] = number of draws containing both i+1 and j+1 (symmetric). The diagonal
     * P[i][i] equals counts()[i].
     */
    public int[][] pairCounts() {
        return copy(pairsRef());
    }

    /** The draws of this view, chronological (a copy). */
    public List<Draw> draws() {
        ArrayList<Draw> out = new ArrayList<Draw>(n);
        for (int i = 0; i < n; i++) {
            out.add(rs.draw(i));
        }
        return out;
    }

    // ------------------------------------------------------------------ cached, shared arrays

    int[] countsRef() {
        if (counts == null) {
            counts = rs.counts(n);
        }
        return counts;
    }

    int[] countsInLastRef(int window) {
        for (int i = 0; i < windowKeys.size(); i++) {
            if (windowKeys.get(i).intValue() == window) {
                return windowCounts.get(i);
            }
        }
        int[] c = rs.countsInLast(n, window);
        windowKeys.add(Integer.valueOf(window));
        windowCounts.add(c);
        return c;
    }

    double[] decayedRef(double halfLife) {
        for (int i = 0; i < decayKeys.size(); i++) {
            if (decayKeys.get(i).doubleValue() == halfLife) {
                return decayValues.get(i);
            }
        }
        double[] v = rs.decayedCounts(n, halfLife);
        decayKeys.add(Double.valueOf(halfLife));
        decayValues.add(v);
        return v;
    }

    int[] sinceRef() {
        if (since == null) {
            since = rs.drawsSinceSeen(n);
        }
        return since;
    }

    int[][] transRef() {
        if (trans == null) {
            trans = rs.transitionCounts(n);
        }
        return trans;
    }

    int[] containsRef() {
        if (containsC == null) {
            containsC = rs.containsCounts(n);
        }
        return containsC;
    }

    int[][] pairsRef() {
        if (pairs == null) {
            pairs = rs.pairCounts(n);
        }
        return pairs;
    }

    private static int[][] copy(int[][] m) {
        int[][] c = new int[m.length][];
        for (int i = 0; i < m.length; i++) {
            c[i] = m[i].clone();
        }
        return c;
    }
}
