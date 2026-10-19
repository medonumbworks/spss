package com.medonumbworks.csprngpredictor.core;

import java.util.Arrays;

/**
 * One published draw: an ISO date and 11 distinct numbers 1..22 (stored sorted ascending).
 */
public final class Draw implements Comparable<Draw> {

    /** ISO yyyy-MM-dd, never null. */
    public final String date;

    /** Defensive copy, sorted ascending, distinct, 1..22, length 11. */
    public final int[] numbers;

    /**
     * @throws IllegalArgumentException when the date is not a real ISO date or the numbers
     *                                  are not 11 distinct values in 1..22
     */
    public Draw(String isoDate, int[] numbers) {
        String d = isoDate == null ? null : isoDate.trim();
        if (d == null || !DateUtil.isIso(d)) {
            throw new IllegalArgumentException("Invalid ISO date (expected yyyy-MM-dd): " + isoDate);
        }
        if (!Game.isValidDraw(numbers)) {
            throw new IllegalArgumentException(
                    "A draw needs exactly 11 distinct numbers from 1 to 22: " + Arrays.toString(numbers));
        }
        int[] copy = numbers.clone();
        Arrays.sort(copy);
        this.date = d;
        this.numbers = copy;
    }

    /** True when {@code n} is one of the drawn numbers. */
    public boolean contains(int n) {
        for (int x : numbers) {
            if (x == n) {
                return true;
            }
        }
        return false;
    }

    /** The numbers in ascending order joined by {@code separator}, e.g. "1 4 7 ...". */
    public String numbersString(String separator) {
        return join(numbers, separator);
    }

    /** |numbers intersect predicted|; duplicates and invalid values in {@code predicted} are ignored. */
    public int hitsAgainst(int[] predicted) {
        if (predicted == null) {
            return 0;
        }
        long mine = mask(numbers);
        long seen = 0L;
        int hits = 0;
        for (int p : predicted) {
            if (p >= 1 && p <= Game.MAX_NUMBER) {
                long bit = 1L << p;
                if ((mine & bit) != 0L && (seen & bit) == 0L) {
                    hits++;
                    seen |= bit;
                }
            }
        }
        return hits;
    }

    /** Bit {@code n} set for each valid number n in the array. */
    static long mask(int[] nums) {
        long m = 0L;
        if (nums != null) {
            for (int x : nums) {
                if (x >= 1 && x <= Game.MAX_NUMBER) {
                    m |= 1L << x;
                }
            }
        }
        return m;
    }

    static String join(int[] nums, String separator) {
        if (nums == null) {
            return "";
        }
        String sep = separator == null ? "" : separator;
        StringBuilder sb = new StringBuilder(nums.length * 4);
        for (int i = 0; i < nums.length; i++) {
            if (i > 0) {
                sb.append(sep);
            }
            sb.append(nums[i]);
        }
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Draw)) {
            return false;
        }
        Draw other = (Draw) o;
        return date.equals(other.date) && Arrays.equals(numbers, other.numbers);
    }

    @Override
    public int hashCode() {
        return 31 * date.hashCode() + Arrays.hashCode(numbers);
    }

    /** By date, then numbers lexicographically. */
    @Override
    public int compareTo(Draw o) {
        int c = date.compareTo(o.date);
        if (c != 0) {
            return c;
        }
        int n = Math.min(numbers.length, o.numbers.length);
        for (int i = 0; i < n; i++) {
            if (numbers[i] != o.numbers[i]) {
                return numbers[i] < o.numbers[i] ? -1 : 1;
            }
        }
        return numbers.length - o.numbers.length;
    }

    @Override
    public String toString() {
        return date + ": " + numbersString(" ");
    }
}
