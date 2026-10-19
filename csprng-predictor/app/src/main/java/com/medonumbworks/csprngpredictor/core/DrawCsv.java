package com.medonumbworks.csprngpredictor.core;

import java.util.ArrayList;
import java.util.List;

/**
 * The app's own storage format for draws: {@code date,n1,...,n11} with a header line.
 */
public final class DrawCsv {

    public static final String HEADER = "date,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11";

    private DrawCsv() {
    }

    /** Header + one line per draw, "\n" line endings. */
    public static String write(List<Draw> draws) {
        StringBuilder sb = new StringBuilder(64 + (draws == null ? 0 : draws.size() * 42));
        sb.append(HEADER).append('\n');
        if (draws != null) {
            for (Draw d : draws) {
                if (d == null) {
                    continue;
                }
                sb.append(d.date);
                for (int n : d.numbers) {
                    sb.append(',').append(n);
                }
                sb.append('\n');
            }
        }
        return sb.toString();
    }

    /**
     * Strict reader for the app's own file: header, blank and '#' lines are ignored; any line
     * that is not exactly {@code isoDate,11 numbers} is skipped. Never throws.
     */
    public static List<Draw> read(String csv) {
        return read(csv, null);
    }

    /** Like {@link #read(String)}; descriptions of skipped lines are appended to {@code problems} (may be null). */
    public static List<Draw> read(String csv, List<String> problems) {
        List<Draw> out = new ArrayList<Draw>();
        if (csv == null) {
            return out;
        }
        int len = csv.length();
        int start = 0;
        int lineNo = 0;
        for (int i = 0; i <= len; i++) {
            char ch = i < len ? csv.charAt(i) : '\n';
            if (ch != '\n' && ch != '\r') {
                continue;
            }
            lineNo++;
            String line = csv.substring(start, i).replace("\uFEFF", "").trim();
            if (ch == '\r' && i + 1 < len && csv.charAt(i + 1) == '\n') {
                i++;
            }
            start = i + 1;
            if (line.length() == 0 || line.charAt(0) == '#'
                    || line.regionMatches(true, 0, "date", 0, 4)) {
                continue;
            }
            Draw d = parseLine(line);
            if (d != null) {
                out.add(d);
            } else if (problems != null) {
                problems.add("line " + lineNo + ": not a valid draw: " + abbreviate(line));
            }
        }
        return out;
    }

    private static Draw parseLine(String line) {
        String[] parts = line.split(",", -1);
        if (parts.length < 1 + Game.PICK) {
            return null;
        }
        for (int i = 1 + Game.PICK; i < parts.length; i++) {
            if (parts[i].trim().length() > 0) {
                return null;                              // extra non-empty columns: not our format
            }
        }
        String date = parts[0].trim();
        if (!DateUtil.isIso(date)) {
            return null;
        }
        int[] nums = new int[Game.PICK];
        for (int k = 0; k < Game.PICK; k++) {
            String p = parts[k + 1].trim();
            if (p.length() == 0 || p.length() > 3) {
                return null;
            }
            int v = 0;
            for (int j = 0; j < p.length(); j++) {
                char c = p.charAt(j);
                if (c < '0' || c > '9') {
                    return null;
                }
                v = v * 10 + (c - '0');
            }
            nums[k] = v;
        }
        if (!Game.isValidDraw(nums)) {
            return null;
        }
        return new Draw(date, nums);
    }

    private static String abbreviate(String s) {
        return s.length() > 60 ? s.substring(0, 60) + "..." : s;
    }
}
