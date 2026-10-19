package com.medonumbworks.csprngpredictor.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Walk-forward replay: the model predicts every draw from the draws before it, is scored on
 * the real draw and then learns from it. This is an honest out-of-sample backtest and also
 * the way the model is (re)trained from scratch whenever the data changes.
 *
 * <p>Cost: O(N * 22 * 11) thanks to the incremental {@link RunningStats}.
 */
public final class Replay {

    private static final Comparator<Draw> BY_DATE = new Comparator<Draw>() {
        @Override
        public int compare(Draw a, Draw b) {
            return a.date.compareTo(b.date);
        }
    };

    private Replay() {
    }

    /** Replay with statistics over all records. */
    public static ReplayResult run(List<Draw> chronological, ModelParams params) {
        return run(chronological, params, 0);
    }

    /**
     * Replay with statistics over the last {@code evaluateLastN} records only
     * ({@code evaluateLastN <= 0}: all records). The model always learns from every draw.
     */
    public static ReplayResult run(List<Draw> chronological, ModelParams params, int evaluateLastN) {
        List<Draw> draws = new ArrayList<Draw>();
        if (chronological != null) {
            for (Draw d : chronological) {
                if (d != null) {
                    draws.add(d);
                }
            }
        }
        Collections.sort(draws, BY_DATE);               // stable: no-op for chronological input

        ModelParams p = params == null ? new ModelParams() : params.copy();
        p.clamp();
        Model model = new Model(p, Strategies.defaults(p));
        RunningStats rs = new RunningStats();
        int n = draws.size();
        List<PredictionRecord> records = new ArrayList<PredictionRecord>(Math.max(0, n - 1));
        for (int t = 0; t < n; t++) {
            Draw actual = draws.get(t);
            if (t >= 1) {
                Prediction pred = model.predict(rs.view());
                records.add(new PredictionRecord(pred, actual));
                model.observe(pred, actual);
            }
            rs.push(actual);
        }
        Prediction next = model.predict(rs.view());

        List<PredictionRecord> evaluated = records;
        if (evaluateLastN > 0 && evaluateLastN < records.size()) {
            evaluated = records.subList(records.size() - evaluateLastN, records.size());
        }
        Stats stats = Stats.of(evaluated);
        return new ReplayResult(model, records, next, stats, p);
    }
}
