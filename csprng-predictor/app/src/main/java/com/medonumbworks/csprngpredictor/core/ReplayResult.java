package com.medonumbworks.csprngpredictor.core;

import java.util.Collections;
import java.util.List;

/**
 * Outcome of a walk-forward replay over the full history.
 */
public final class ReplayResult {

    /** State after observing all draws. */
    public final Model model;
    /** One per draw index t &gt;= 1: predicted from draws[0..t), scored on draws[t]. Unmodifiable. */
    public final List<PredictionRecord> records;
    /** Prediction for the next (unseen) draw. */
    public final Prediction next;
    /** Statistics over the evaluated records (the last N if requested). */
    public final Stats stats;
    /** The (clamped) parameters used. */
    public final ModelParams params;

    public ReplayResult(Model model, List<PredictionRecord> records, Prediction next, Stats stats,
                        ModelParams params) {
        this.model = model;
        this.records = records == null
                ? Collections.<PredictionRecord>emptyList()
                : Collections.unmodifiableList(records);
        this.next = next;
        this.stats = stats == null ? Stats.of(this.records) : stats;
        this.params = params == null ? new ModelParams() : params;
    }
}
