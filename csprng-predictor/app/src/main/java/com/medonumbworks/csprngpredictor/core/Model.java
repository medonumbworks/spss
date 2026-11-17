package com.medonumbworks.csprngpredictor.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The online learner: a Hedge (exponential weights) mixture of experts with fixed share,
 * followed by a per-number multiplicative bias correction.
 *
 * <p>Fully deterministic (no randomness; {@link StrictMath} for exp). Not thread-safe.
 */
public final class Model {

    private static final int N = Game.MAX_NUMBER;
    private static final int PICK = Game.PICK;

    private final ModelParams params;
    private final List<Strategy> strategies;
    private final String[] ids;
    private final double[] weights;
    private final double[] biases = new double[N];
    private final int[] hitsTotal;
    private int observed;

    /**
     * @param params     null = defaults; a clamped copy is kept
     * @param strategies null or empty = {@link Strategies#defaults(ModelParams)}
     */
    public Model(ModelParams params, List<Strategy> strategies) {
        ModelParams p = params == null ? new ModelParams() : params.copy();
        p.clamp();
        this.params = p;
        List<Strategy> s = new ArrayList<Strategy>();
        if (strategies != null) {
            for (Strategy x : strategies) {
                if (x != null) {
                    s.add(x);
                }
            }
        }
        if (s.isEmpty()) {
            s.addAll(Strategies.defaults(p));
        }
        this.strategies = Collections.unmodifiableList(s);
        int k = s.size();
        this.ids = new String[k];
        for (int i = 0; i < k; i++) {
            String id = s.get(i).id();
            ids[i] = id == null ? "expert" + (i + 1) : id;
        }
        this.weights = new double[k];
        for (int i = 0; i < k; i++) {
            weights[i] = 1.0 / k;
        }
        this.hitsTotal = new int[k];
    }

    /** Predicts the draw following {@code h}. Pure: does not change the model. */
    public Prediction predict(History h) {
        History hist = h == null ? new History(null) : h;
        int k = strategies.size();

        // 1. each expert -> non-negative scores + 1e-9, normalised to sum 11; own top-11
        double[][] p = new double[k][];
        int[][] picks = new int[k][];
        for (int e = 0; e < k; e++) {
            p[e] = normaliseExpert(strategies.get(e).scores(hist));
            picks[e] = topPicks(p[e]);
        }

        // 2. weighted mixture with weights normalised to sum 1
        double[] w = normalisedWeights();
        double[] combined = new double[N];
        for (int e = 0; e < k; e++) {
            double we = w[e];
            double[] pe = p[e];
            for (int i = 0; i < N; i++) {
                combined[i] += we * pe[i];
            }
        }

        // 3. per-number bias, renormalised to sum 11
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            combined[i] *= StrictMath.exp(biases[i]);
            sum += combined[i];
        }
        if (sum > 0.0 && sum < Double.POSITIVE_INFINITY) {
            for (int i = 0; i < N; i++) {
                combined[i] = PICK * combined[i] / sum;
            }
        } else {
            for (int i = 0; i < N; i++) {
                combined[i] = PICK / (double) N;
            }
        }

        // 4. top 11 (ties: lower number first)
        int[] numbers = topPicks(combined);
        Draw last = hist.last();
        String basedOn = last == null ? "" : last.date;
        String target = last == null ? DateUtil.today() : DateUtil.addDays(last.date, 1);
        return new Prediction(basedOn, hist.size(), target, numbers, combined, ids, w, picks);
    }

    /**
     * Learns from the real draw that followed prediction {@code p} (which must come from
     * this model): Hedge update with loss 1 - hits/11, fixed share, per-number bias update.
     */
    public void observe(Prediction p, Draw actual) {
        if (p == null || actual == null) {
            throw new IllegalArgumentException("prediction and actual draw must not be null");
        }
        int k = weights.length;
        if (p.expertPicks.length != k) {
            throw new IllegalArgumentException("prediction has " + p.expertPicks.length
                    + " experts, this model has " + k);
        }
        // 1-2. expert hits, Hedge, normalise, fixed share
        double eta = params.eta;
        double sum = 0.0;
        for (int e = 0; e < k; e++) {
            int hits = actual.hitsAgainst(p.expertPicks[e]);
            hitsTotal[e] += hits;
            double loss = 1.0 - hits / (double) PICK;
            weights[e] *= StrictMath.exp(-eta * loss);
            sum += weights[e];
        }
        if (sum > 0.0 && sum < Double.POSITIVE_INFINITY) {
            for (int e = 0; e < k; e++) {
                weights[e] /= sum;
            }
        } else {
            for (int e = 0; e < k; e++) {
                weights[e] = 1.0 / k;
            }
        }
        double share = params.share;
        for (int e = 0; e < k; e++) {
            weights[e] = (1.0 - share) * weights[e] + share / k;
        }

        // 3. per-number bias toward what was actually drawn
        long drawn = Draw.mask(actual.numbers);
        double keep = 1.0 - params.biasDecay;
        double rate = params.biasRate;
        for (int i = 0; i < N; i++) {
            double target = (drawn & (1L << (i + 1))) != 0L ? 1.0 : 0.0;
            // scores sum to PICK (11), so scores[i] is already the inclusion
            // probability of number i+1; compare the 0/1 target on that scale
            // (dividing by PICK again would make every bias drift upward).
            double expected = i < p.scores.length ? p.scores[i] : 0.5;
            if (Double.isNaN(expected) || Double.isInfinite(expected)) {
                expected = 0.5;
            }
            expected = expected < 0.0 ? 0.0 : (expected > 1.0 ? 1.0 : expected);
            double b = keep * biases[i] + rate * (target - expected);
            biases[i] = b < -1.0 ? -1.0 : (b > 1.0 ? 1.0 : b);
        }

        // 4. bookkeeping
        observed++;
    }

    /** Expert weights (sum 1), strategy order; a copy. */
    public double[] weights() {
        return weights.clone();
    }

    /** Per-number log-multipliers in [-1, 1]; a copy. */
    public double[] biases() {
        return biases.clone();
    }

    /** Number of observe() calls. */
    public int observed() {
        return observed;
    }

    /** Cumulative hits per expert; a copy. */
    public int[] expertHitsTotal() {
        return hitsTotal.clone();
    }

    /** Mean hits per expert, 0 when nothing was observed. */
    public double[] expertMeanHits() {
        double[] m = new double[hitsTotal.length];
        if (observed > 0) {
            for (int e = 0; e < m.length; e++) {
                m[e] = hitsTotal[e] / (double) observed;
            }
        }
        return m;
    }

    /** Unmodifiable, in weight order. */
    public List<Strategy> strategies() {
        return strategies;
    }

    /** Expert ids in strategy order; a copy. */
    public String[] expertIds() {
        return ids.clone();
    }

    /** A copy of the (clamped) parameters. */
    public ModelParams params() {
        return params.copy();
    }

    // ------------------------------------------------------------------ helpers

    private double[] normalisedWeights() {
        int k = weights.length;
        double sum = 0.0;
        for (double v : weights) {
            sum += v;
        }
        double[] w = new double[k];
        if (sum > 0.0 && sum < Double.POSITIVE_INFINITY) {
            for (int e = 0; e < k; e++) {
                w[e] = weights[e] / sum;
            }
        } else {
            for (int e = 0; e < k; e++) {
                w[e] = 1.0 / k;
            }
        }
        return w;
    }

    /** Clamp negatives / NaN to 0, add 1e-9, normalise to sum 11 (overflow-safe). */
    static double[] normaliseExpert(double[] s) {
        double[] v = new double[N];
        double sum = 0.0;
        double max = 0.0;
        for (int i = 0; i < N; i++) {
            double x = s == null ? 1.0 : (i < s.length ? s[i] : 0.0);
            if (!(x > 0.0)) {
                x = 0.0;                                  // negatives and NaN
            } else if (x > Double.MAX_VALUE) {
                x = Double.MAX_VALUE;                     // +Infinity
            }
            v[i] = x + 1e-9;
            sum += v[i];
            if (v[i] > max) {
                max = v[i];
            }
        }
        if (!(sum < Double.POSITIVE_INFINITY)) {
            sum = 0.0;                                    // rescale huge scores before summing
            for (int i = 0; i < N; i++) {
                v[i] = v[i] / max;
                sum += v[i];
            }
        }
        for (int i = 0; i < N; i++) {
            v[i] = PICK * v[i] / sum;
        }
        return v;
    }

    /** The 11 highest-scoring numbers (ties: lower number first), sorted ascending. */
    static int[] topPicks(double[] score) {
        int[] idx = new int[N];
        for (int i = 0; i < N; i++) {
            idx[i] = i;
        }
        for (int i = 1; i < N; i++) {                     // insertion sort, stable and tiny
            int x = idx[i];
            int j = i - 1;
            while (j >= 0 && better(score, x, idx[j])) {
                idx[j + 1] = idx[j];
                j--;
            }
            idx[j + 1] = x;
        }
        boolean[] chosen = new boolean[N];
        for (int r = 0; r < PICK; r++) {
            chosen[idx[r]] = true;
        }
        int[] out = new int[PICK];
        int c = 0;
        for (int i = 0; i < N; i++) {
            if (chosen[i]) {
                out[c++] = i + 1;
            }
        }
        return out;
    }

    private static boolean better(double[] s, int a, int b) {
        return s[a] > s[b] || (s[a] == s[b] && a < b);
    }
}
