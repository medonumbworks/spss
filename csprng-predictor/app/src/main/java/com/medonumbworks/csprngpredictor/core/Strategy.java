package com.medonumbworks.csprngpredictor.core;

/**
 * One prediction "expert". Implementations must be deterministic and side-effect free.
 */
public interface Strategy {

    /**
     * Stable identifier used in persisted data: "uniform", "freq", "recent10", "recent30",
     * "decay", "gap", "transition", "repeat", "pairs" for the defaults.
     */
    String id();

    /** UI label. */
    String name();

    /** One sentence for the UI. */
    String description();

    /**
     * Scores for the next draw given the history: length 22, index i -&gt; number i+1,
     * non-negative and not all zero. Higher = more likely.
     */
    double[] scores(History h);
}
