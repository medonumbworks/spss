package com.medonumbworks.csprngpredictor.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Log of the predictions shown to the user, settled against real draws as they arrive.
 * Thread-safe (all methods synchronized).
 */
public final class IssuedLog {

    public static final String HEADER =
            "issuedAt,basedOnDate,basedOnCount,targetDate,numbers,actualDate,actual,hits";

    private final ArrayList<IssuedPrediction> entries = new ArrayList<IssuedPrediction>();

    public IssuedLog() {
    }

    /** Unmodifiable snapshot in issue order. */
    public synchronized List<IssuedPrediction> all() {
        return Collections.unmodifiableList(new ArrayList<IssuedPrediction>(entries));
    }

    public synchronized int size() {
        return entries.size();
    }

    /** Existing pending entry with the same basis, or null. */
    public synchronized IssuedPrediction pendingFor(String basedOnDate, int basedOnCount) {
        String date = basedOnDate == null ? "" : basedOnDate;
        for (IssuedPrediction p : entries) {
            if (p.isPending() && p.basedOnCount == basedOnCount && p.basedOnDate.equals(date)) {
                return p;
            }
        }
        return null;
    }

    /** Appends (no dedupe); returns {@code p}. Null is ignored. */
    public synchronized IssuedPrediction add(IssuedPrediction p) {
        if (p != null) {
            entries.add(p);
        }
        return p;
    }

    /**
     * For each pending entry, the real draw is the first draw after its basedOnDate (the
     * earliest draw when basedOnDate is ""). Returns the number of entries settled.
     */
    public synchronized int settle(DrawSet draws) {
        if (draws == null) {
            return 0;
        }
        int settled = 0;
        for (IssuedPrediction p : entries) {
            if (!p.isPending()) {
                continue;
            }
            Draw d = p.basedOnDate.length() == 0 ? draws.earliest() : draws.firstAfter(p.basedOnDate);
            if (d != null) {
                p.settle(d);
                settled++;
            }
        }
        return settled;
    }

    /** The most recent (in issue order) settled entry, or null. */
    public synchronized IssuedPrediction lastSettled() {
        for (int i = entries.size() - 1; i >= 0; i--) {
            if (!entries.get(i).isPending()) {
                return entries.get(i);
            }
        }
        return null;
    }

    /** Number of entries still waiting for their draw. */
    public synchronized int pendingCount() {
        int c = 0;
        for (IssuedPrediction p : entries) {
            if (p.isPending()) {
                c++;
            }
        }
        return c;
    }

    /** CSV with {@link #HEADER}; numbers space-separated in one cell; empty cells while pending. */
    public synchronized String toCsv() {
        StringBuilder sb = new StringBuilder(HEADER.length() + 1 + entries.size() * 100);
        sb.append(HEADER).append('\n');
        for (IssuedPrediction p : entries) {
            sb.append(clean(p.issuedAt)).append(',')
                    .append(clean(p.basedOnDate)).append(',')
                    .append(p.basedOnCount).append(',')
                    .append(clean(p.targetDate)).append(',')
                    .append(Draw.join(p.numbers, " ")).append(',');
            if (!p.isPending()) {
                sb.append(clean(p.actualDate)).append(',')
                        .append(Draw.join(p.actual, " ")).append(',')
                        .append(p.hits);
            } else {
                sb.append(",,");
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    /** Tolerant reader for {@link #toCsv()} output: bad lines are skipped, never throws. */
    public static IssuedLog fromCsv(String csv) {
        IssuedLog log = new IssuedLog();
        if (csv == null) {
            return log;
        }
        String[] lines = csv.split("\r\n|\r|\n");
        for (String raw : lines) {
            String line = raw.replace("\uFEFF", "").trim();
            if (line.length() == 0 || line.startsWith("#")
                    || line.regionMatches(true, 0, "issuedAt", 0, 8)) {
                continue;
            }
            try {
                IssuedPrediction p = parseLine(line);
                if (p != null) {
                    log.entries.add(p);
                }
            } catch (RuntimeException ignored) {
                // tolerant: skip the line
            }
        }
        return log;
    }

    private static IssuedPrediction parseLine(String line) {
        String[] f = line.split(",", -1);
        if (f.length < 5) {
            return null;
        }
        for (int i = 0; i < f.length; i++) {
            f[i] = unquote(f[i]);                         // tolerate a spreadsheet re-save
        }
        int[] nums = DrawParser.extractNumbers(f[4]);
        if (!Game.isValidDraw(nums)) {
            return null;
        }
        int count = 0;
        try {
            count = Integer.parseInt(f[2].trim());
        } catch (NumberFormatException ignored) {
            // keep 0
        }
        IssuedPrediction p = new IssuedPrediction(f[0].trim(), f[1].trim(), count, f[3].trim(), nums);
        if (f.length >= 7) {
            int[] actual = DrawParser.extractNumbers(f[6]);
            if (Game.isValidDraw(actual)) {
                int[] sorted = actual.clone();
                java.util.Arrays.sort(sorted);
                p.actualDate = f[5].trim();
                p.actual = sorted;
                // recomputed rather than trusted, so a hand-edited file stays consistent
                p.hits = Long.bitCount(Draw.mask(sorted) & Draw.mask(p.numbers));
            }
        }
        return p;
    }

    /** Stats over the settled entries (issue order). */
    public synchronized Stats stats() {
        List<Integer> hits = new ArrayList<Integer>();
        for (IssuedPrediction p : entries) {
            if (!p.isPending()) {
                hits.add(Integer.valueOf(p.hits));
            }
        }
        return Stats.ofHits(hits);
    }

    public synchronized void clear() {
        entries.clear();
    }

    private static String unquote(String s) {
        String t = s.trim();
        if (t.length() >= 2 && t.charAt(0) == '"' && t.charAt(t.length() - 1) == '"') {
            t = t.substring(1, t.length() - 1).trim();
        }
        return t;
    }

    private static String clean(String s) {
        if (s == null) {
            return "";
        }
        return s.replace(',', ' ').replace('\n', ' ').replace('\r', ' ');
    }
}
