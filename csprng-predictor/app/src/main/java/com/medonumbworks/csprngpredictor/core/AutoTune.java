package com.medonumbworks.csprngpredictor.core;

import java.util.List;

/**
 * Small grid search over the learner parameters, scored by walk-forward mean hits.
 *
 * <p>Note: choosing the best of 27 configurations on the same data is itself a mild form of
 * overfitting, so the winner's backtest score is optimistic; on truly random draws every
 * configuration stays near 5.5.
 */
public final class AutoTune {

    /** Grid values (outer to inner loop: eta, halfLife, biasRate). */
    public static final double[] ETA_GRID = {0.1, 0.3, 1.0};
    public static final double[] HALF_LIFE_GRID = {5, 15, 40};
    public static final double[] BIAS_RATE_GRID = {0, 0.05, 0.2};

    /** Fewer draws than this: the base parameters are returned unchanged. */
    public static final int MIN_DRAWS = 15;

    private AutoTune() {
    }

    /** A parameter set with its backtest statistics. */
    public static final class Candidate {
        public final ModelParams params;
        public final Stats stats;

        public Candidate(ModelParams params, Stats stats) {
            this.params = params;
            this.stats = stats;
        }

        @Override
        public String toString() {
            return "Candidate{" + params + ", " + stats + "}";
        }
    }

    /**
     * Grid eta {0.1, 0.3, 1.0} x halfLife {5, 15, 40} x biasRate {0, 0.05, 0.2}, other fields
     * from {@code base}. Highest meanHits over the last {@code evaluateLastN} records wins
     * (ties: fewer changed fields vs base, then first in grid order). With fewer than 15
     * draws, {@code base} is returned unchanged with its own stats.
     */
    public static Candidate tune(List<Draw> chronological, ModelParams base, int evaluateLastN) {
        ModelParams b = base == null ? new ModelParams() : base.copy();
        b.clamp();
        int size = chronological == null ? 0 : chronological.size();
        if (size < MIN_DRAWS) {
            return new Candidate(b, Replay.run(chronological, b, evaluateLastN).stats);
        }
        ModelParams bestParams = null;
        Stats bestStats = null;
        long bestTotal = Long.MIN_VALUE;
        int bestChanges = Integer.MAX_VALUE;
        for (double eta : ETA_GRID) {
            for (double halfLife : HALF_LIFE_GRID) {
                for (double biasRate : BIAS_RATE_GRID) {
                    ModelParams p = b.copy();
                    p.eta = eta;
                    p.halfLife = halfLife;
                    p.biasRate = biasRate;
                    p.clamp();
                    Stats s = Replay.run(chronological, p, evaluateLastN).stats;
                    // every candidate is evaluated on the same records, so total hits
                    // compare exactly like mean hits (no floating point ties issues)
                    long total = Math.round(s.meanHits * s.n);
                    int changes = (eta != b.eta ? 1 : 0) + (halfLife != b.halfLife ? 1 : 0)
                            + (biasRate != b.biasRate ? 1 : 0);
                    if (bestParams == null || total > bestTotal
                            || (total == bestTotal && changes < bestChanges)) {
                        bestParams = p;
                        bestStats = s;
                        bestTotal = total;
                        bestChanges = changes;
                    }
                }
            }
        }
        return new Candidate(bestParams, bestStats);
    }
}
