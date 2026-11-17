package com.medonumbworks.csprngpredictor.core;

/**
 * Constants and validation rules of the daily "pick 11 numbers out of 1..22" draw.
 */
public final class Game {

    /** Numbers are 1..MAX_NUMBER. */
    public static final int MAX_NUMBER = 22;

    /** How many numbers are drawn (and predicted) per draw. */
    public static final int PICK = 11;

    /** Expected hits of a random 11-number guess: 11 * 11 / 22. */
    public static final double EXPECTED_HITS = 5.5;

    private Game() {
    }

    /** True for 1..22. */
    public static boolean isValidNumber(int n) {
        return n >= 1 && n <= MAX_NUMBER;
    }

    /** True when the array has exactly 11 distinct numbers, each 1..22 (any order). */
    public static boolean isValidDraw(int[] nums) {
        if (nums == null || nums.length != PICK) {
            return false;
        }
        boolean[] seen = new boolean[MAX_NUMBER + 1];
        for (int n : nums) {
            if (!isValidNumber(n) || seen[n]) {
                return false;
            }
            seen[n] = true;
        }
        return true;
    }
}
