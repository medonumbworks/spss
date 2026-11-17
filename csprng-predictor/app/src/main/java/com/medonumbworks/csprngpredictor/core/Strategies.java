package com.medonumbworks.csprngpredictor.core;

import java.util.ArrayList;
import java.util.List;

/**
 * The nine default experts. {@link #defaults(ModelParams)} returns them in the fixed order
 * uniform, freq, recent10, recent30, decay, gap, transition, repeat, pairs.
 */
public final class Strategies {

    public static final String UNIFORM = "uniform";
    public static final String FREQ = "freq";
    public static final String RECENT10 = "recent10";
    public static final String RECENT30 = "recent30";
    public static final String DECAY = "decay";
    public static final String GAP = "gap";
    public static final String TRANSITION = "transition";
    public static final String REPEAT = "repeat";
    public static final String PAIRS = "pairs";

    private static final String[] IDS = {
            UNIFORM, FREQ, RECENT10, RECENT30, DECAY, GAP, TRANSITION, REPEAT, PAIRS
    };

    private static final int N = Game.MAX_NUMBER;

    private Strategies() {
    }

    /** The default experts configured from {@code p} (null = defaults), in the documented order. */
    public static List<Strategy> defaults(ModelParams p) {
        ModelParams q = p == null ? new ModelParams() : p.copy();
        q.clamp();
        List<Strategy> list = new ArrayList<Strategy>(IDS.length);
        for (String id : IDS) {
            list.add(byId(id, q));
        }
        return list;
    }

    /** Ids of the default experts in order. */
    public static String[] defaultIds() {
        return IDS.clone();
    }

    /** A default expert by id, or null for an unknown id. */
    public static Strategy byId(String id, ModelParams p) {
        if (id == null) {
            return null;
        }
        ModelParams q = p == null ? new ModelParams() : p.copy();
        q.clamp();
        if (UNIFORM.equals(id)) {
            return new Uniform();
        } else if (FREQ.equals(id)) {
            return new Frequency();
        } else if (RECENT10.equals(id)) {
            return new Recent(RECENT10, q.shortWindow);
        } else if (RECENT30.equals(id)) {
            return new Recent(RECENT30, q.longWindow);
        } else if (DECAY.equals(id)) {
            return new Decay(q.halfLife);
        } else if (GAP.equals(id)) {
            return new Gap();
        } else if (TRANSITION.equals(id)) {
            return new Transition();
        } else if (REPEAT.equals(id)) {
            return new Repeat();
        } else if (PAIRS.equals(id)) {
            return new Pairs();
        }
        return null;
    }

    private static double[] filled(double v) {
        double[] s = new double[N];
        for (int i = 0; i < N; i++) {
            s[i] = v;
        }
        return s;
    }

    // ================================================================== experts

    /** Control expert: every number scores 1. */
    public static final class Uniform implements Strategy {
        @Override
        public String id() {
            return UNIFORM;
        }

        @Override
        public String name() {
            return "Uniform (control)";
        }

        @Override
        public String description() {
            return "Scores every number the same; if this expert ends up with the most weight, the draws look random.";
        }

        @Override
        public double[] scores(History h) {
            return filled(1.0);
        }
    }

    /** Laplace-smoothed all-time frequency: count + 1. */
    public static final class Frequency implements Strategy {
        @Override
        public String id() {
            return FREQ;
        }

        @Override
        public String name() {
            return "All-time frequency";
        }

        @Override
        public String description() {
            return "Favours the numbers that were drawn most often over the whole history.";
        }

        @Override
        public double[] scores(History h) {
            int[] c = h.countsRef();
            double[] s = new double[N];
            for (int i = 0; i < N; i++) {
                s[i] = c[i] + 1.0;
            }
            return s;
        }
    }

    /** "Hot" numbers: counts in the last {@code window} draws + 0.5. */
    public static final class Recent implements Strategy {
        private final String id;
        private final int window;

        public Recent(String id, int window) {
            this.id = id;
            this.window = Math.max(1, window);
        }

        public int window() {
            return window;
        }

        @Override
        public String id() {
            return id;
        }

        @Override
        public String name() {
            return "Hot (last " + window + " draws)";
        }

        @Override
        public String description() {
            return "Favours the numbers drawn most often in the last " + window + " draws.";
        }

        @Override
        public double[] scores(History h) {
            int[] c = h.countsInLastRef(window);
            double[] s = new double[N];
            for (int i = 0; i < N; i++) {
                s[i] = c[i] + 0.5;
            }
            return s;
        }
    }

    /** Exponentially decayed frequency + 0.5. */
    public static final class Decay implements Strategy {
        private final double halfLife;

        public Decay(double halfLife) {
            this.halfLife = halfLife;
        }

        @Override
        public String id() {
            return DECAY;
        }

        @Override
        public String name() {
            return "Decayed frequency";
        }

        @Override
        public String description() {
            return "Like frequency, but a draw's influence halves every "
                    + ModelParams.formatNumber(halfLife) + " draws.";
        }

        @Override
        public double[] scores(History h) {
            double[] v = h.decayedRef(halfLife);
            double[] s = new double[N];
            for (int i = 0; i < N; i++) {
                s[i] = v[i] + 0.5;
            }
            return s;
        }
    }

    /** Overdue numbers: draws since last seen + 1 (never seen: size + 1). */
    public static final class Gap implements Strategy {
        @Override
        public String id() {
            return GAP;
        }

        @Override
        public String name() {
            return "Overdue";
        }

        @Override
        public String description() {
            return "Favours the numbers that have not been drawn for the longest time.";
        }

        @Override
        public double[] scores(History h) {
            int[] g = h.sinceRef();
            double[] s = new double[N];
            for (int i = 0; i < N; i++) {
                s[i] = g[i] + 1.0;
            }
            return s;
        }
    }

    /** First-order Markov: mean over i in the last draw of (T[i][j] + 1) / (C[i] + 2). */
    public static final class Transition implements Strategy {
        @Override
        public String id() {
            return TRANSITION;
        }

        @Override
        public String name() {
            return "Transition (Markov)";
        }

        @Override
        public String description() {
            return "Favours the numbers that historically followed the numbers of the last draw.";
        }

        @Override
        public double[] scores(History h) {
            Draw last = h.last();
            if (last == null) {
                return filled(1.0);
            }
            int[][] t = h.transRef();
            int[] c = h.containsRef();
            double[] s = new double[N];
            for (int a : last.numbers) {
                int i = a - 1;
                double denom = c[i] + 2.0;
                int[] row = t[i];
                for (int j = 0; j < N; j++) {
                    s[j] += (row[j] + 1.0) / denom;
                }
            }
            double k = last.numbers.length;
            for (int j = 0; j < N; j++) {
                s[j] /= k;
            }
            return s;
        }
    }

    /** Last-draw numbers get repeatRate(), the others 1 - repeatRate(). */
    public static final class Repeat implements Strategy {
        @Override
        public String id() {
            return REPEAT;
        }

        @Override
        public String name() {
            return "Repeat";
        }

        @Override
        public String description() {
            return "Expects numbers of the last draw to repeat at the historically observed repeat rate.";
        }

        @Override
        public double[] scores(History h) {
            Draw last = h.last();
            if (last == null) {
                return filled(1.0);
            }
            double r = h.repeatRate();
            double[] s = filled(1.0 - r);
            for (int a : last.numbers) {
                s[a - 1] = r;
            }
            return s;
        }
    }

    /**
     * Co-occurrence affinity with the last draw: sum over i in the last draw of
     * (P[i][j] + 1) / (counts[i] + 2). The self term i == j is left out (a number always
     * "co-occurs" with itself) and the sum for numbers of the last draw is rescaled by 11/10
     * so they compete with the other numbers on equal terms (no built-in repeat bias).
     */
    public static final class Pairs implements Strategy {
        @Override
        public String id() {
            return PAIRS;
        }

        @Override
        public String name() {
            return "Pairs";
        }

        @Override
        public String description() {
            return "Favours the numbers that often appeared together with the numbers of the last draw.";
        }

        @Override
        public double[] scores(History h) {
            Draw last = h.last();
            if (last == null) {
                return filled(1.0);
            }
            int[][] p = h.pairsRef();
            int[] c = h.countsRef();
            double[] s = new double[N];
            boolean[] inLast = new boolean[N];
            for (int a : last.numbers) {
                int i = a - 1;
                inLast[i] = true;
                double denom = c[i] + 2.0;
                int[] row = p[i];
                for (int j = 0; j < N; j++) {
                    if (j != i) {
                        s[j] += (row[j] + 1.0) / denom;
                    }
                }
            }
            int k = last.numbers.length;
            double scale = k > 1 ? k / (double) (k - 1) : 1.0;
            for (int j = 0; j < N; j++) {
                if (inLast[j]) {
                    s[j] *= scale;
                }
            }
            return s;
        }
    }
}
