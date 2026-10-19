package com.medonumbworks.csprngpredictor.core;

import java.util.Arrays;

/**
 * A prediction that was actually shown to the user, kept so it can be checked against the
 * real draw later (forward test, as opposed to the replay backtest).
 */
public final class IssuedPrediction {

    /** ISO date the prediction was shown. */
    public final String issuedAt;
    /** Latest draw date it was based on ("" if none). */
    public final String basedOnDate;
    public final int basedOnCount;
    public final String targetDate;
    /** The 11 predicted numbers, sorted. */
    public final int[] numbers;
    /** null while pending. */
    public String actualDate;
    /** null while pending. */
    public int[] actual;
    /** -1 while pending. */
    public int hits = -1;

    /**
     * @throws IllegalArgumentException when {@code numbers} is not 11 distinct values 1..22
     */
    public IssuedPrediction(String issuedAt, String basedOnDate, int basedOnCount, String targetDate,
                            int[] numbers) {
        if (!Game.isValidDraw(numbers)) {
            throw new IllegalArgumentException("A prediction needs 11 distinct numbers 1..22: "
                    + Arrays.toString(numbers));
        }
        this.issuedAt = issuedAt == null ? "" : issuedAt;
        this.basedOnDate = basedOnDate == null ? "" : basedOnDate;
        this.basedOnCount = basedOnCount;
        this.targetDate = targetDate == null ? "" : targetDate;
        int[] n = numbers.clone();
        Arrays.sort(n);
        this.numbers = n;
    }

    /** The entry for showing {@code p} on {@code issuedAtIso} (null = today). */
    public static IssuedPrediction of(Prediction p, String issuedAtIso) {
        return new IssuedPrediction(issuedAtIso == null ? DateUtil.today() : issuedAtIso,
                p.basedOnDate, p.basedOnCount, p.targetDate, p.numbers);
    }

    public boolean isPending() {
        return actual == null;
    }

    /** Fills actualDate / actual / hits from the real draw. */
    public void settle(Draw d) {
        if (d == null) {
            return;
        }
        actualDate = d.date;
        actual = d.numbers.clone();
        hits = d.hitsAgainst(numbers);
    }

    /** Predicted numbers that were drawn (empty while pending). */
    public int[] hitNumbers() {
        if (actual == null) {
            return new int[0];
        }
        long m = Draw.mask(actual);
        int[] tmp = new int[numbers.length];
        int c = 0;
        for (int x : numbers) {
            if (x >= 1 && x <= Game.MAX_NUMBER && (m & (1L << x)) != 0L) {
                tmp[c++] = x;
            }
        }
        return Arrays.copyOf(tmp, c);
    }

    @Override
    public String toString() {
        return "IssuedPrediction{issued " + issuedAt + ", for " + targetDate + ": "
                + Draw.join(numbers, " ") + (isPending() ? ", pending" : ", " + hits + " hits") + "}";
    }
}
