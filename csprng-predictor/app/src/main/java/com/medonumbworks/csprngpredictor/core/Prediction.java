package com.medonumbworks.csprngpredictor.core;

import java.util.Arrays;

/**
 * One prediction of the next draw, with the per-number scores and the experts behind it.
 * All arrays are defensive copies made at construction.
 */
public final class Prediction {

    /** Date of the latest draw used ("" if none). */
    public final String basedOnDate;
    /** Number of draws used. */
    public final int basedOnCount;
    /** basedOnDate + 1 day, or today if there were no draws. */
    public final String targetDate;
    /** The 11 picks, sorted ascending. */
    public final int[] numbers;
    /** Length 22: combined score per number, normalised to sum 11 (an expected count, 0.5 each when uniform). */
    public final double[] scores;
    /** Expert ids in strategy order. */
    public final String[] expertIds;
    /** Expert weights, normalised to sum 1, same order. */
    public final double[] expertWeights;
    /** Each expert's own top-11 (sorted ascending), same order. */
    public final int[][] expertPicks;

    public Prediction(String basedOnDate, int basedOnCount, String targetDate, int[] numbers,
                      double[] scores, String[] expertIds, double[] expertWeights, int[][] expertPicks) {
        this.basedOnDate = basedOnDate == null ? "" : basedOnDate;
        this.basedOnCount = basedOnCount;
        this.targetDate = targetDate == null ? "" : targetDate;
        int[] nums = numbers == null ? new int[0] : numbers.clone();
        Arrays.sort(nums);
        this.numbers = nums;
        this.scores = scores == null ? new double[Game.MAX_NUMBER] : scores.clone();
        this.expertIds = expertIds == null ? new String[0] : expertIds.clone();
        this.expertWeights = expertWeights == null ? new double[0] : expertWeights.clone();
        int[][] picks = expertPicks == null ? new int[0][] : new int[expertPicks.length][];
        for (int k = 0; k < picks.length; k++) {
            int[] p = expertPicks[k] == null ? new int[0] : expertPicks[k].clone();
            Arrays.sort(p);
            picks[k] = p;
        }
        this.expertPicks = picks;
    }

    /** Combined score (expected count 0..1) of a number 1..22; 0 for invalid numbers. */
    public double scoreOf(int number) {
        if (!Game.isValidNumber(number) || number > scores.length) {
            return 0.0;
        }
        return scores[number - 1];
    }

    /**
     * 1 = highest score; ties are ranked by the lower number first (the same rule that picks
     * the numbers), so ranks 1..11 are exactly the predicted numbers. -1 for invalid numbers.
     */
    public int rankOf(int number) {
        if (!Game.isValidNumber(number) || number > scores.length) {
            return -1;
        }
        int i = number - 1;
        double s = scores[i];
        int rank = 1;
        for (int j = 0; j < scores.length; j++) {
            if (scores[j] > s || (scores[j] == s && j < i)) {
                rank++;
            }
        }
        return rank;
    }

    public String numbersString(String sep) {
        return Draw.join(numbers, sep);
    }

    /** Index of an expert id, or -1. */
    public int expertIndex(String id) {
        for (int k = 0; k < expertIds.length; k++) {
            if (expertIds[k] != null && expertIds[k].equals(id)) {
                return k;
            }
        }
        return -1;
    }

    /** True when {@code number} is one of the 11 picks. */
    public boolean contains(int number) {
        for (int n : numbers) {
            if (n == number) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return "Prediction{for " + targetDate + " based on " + basedOnCount + " draws (latest "
                + basedOnDate + "): " + numbersString(" ") + "}";
    }
}
