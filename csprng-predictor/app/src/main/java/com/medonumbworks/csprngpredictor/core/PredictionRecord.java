package com.medonumbworks.csprngpredictor.core;

/**
 * One walk-forward replay step: the prediction made from the earlier draws and the draw
 * that actually followed.
 */
public final class PredictionRecord {

    public final Prediction prediction;
    /** Never null. */
    public final Draw actual;
    /** actual.hitsAgainst(prediction.numbers). */
    public final int hits;
    /** Hits of each expert's own top-11, in strategy order. */
    public final int[] expertHits;

    /** Scores {@code prediction} (and each of its experts) against {@code actual}. */
    public PredictionRecord(Prediction prediction, Draw actual) {
        if (prediction == null || actual == null) {
            throw new IllegalArgumentException("prediction and actual must not be null");
        }
        this.prediction = prediction;
        this.actual = actual;
        this.hits = actual.hitsAgainst(prediction.numbers);
        int[] eh = new int[prediction.expertPicks.length];
        for (int k = 0; k < eh.length; k++) {
            eh[k] = actual.hitsAgainst(prediction.expertPicks[k]);
        }
        this.expertHits = eh;
    }

    @Override
    public String toString() {
        return actual.date + ": " + hits + " / 11 hits";
    }
}
