package com.medonumbworks.csprngpredictor.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Outcome of parsing an imported CSV / TXT / XML file.
 */
public final class ParseResult {

    /** At most this many individual warnings are kept; the rest is summarised as "... and N more". */
    public static final int MAX_WARNINGS = 20;

    /** Rows / elements with a recognised date (file order). */
    public final List<Draw> dated = new ArrayList<Draw>();

    /** Rows / elements with 11 valid numbers but no date (file order); sorted, distinct. */
    public final List<int[]> undated = new ArrayList<int[]>();

    /** Human readable problems, max ~20 kept + "... and N more". */
    public final List<String> warnings = new ArrayList<String>();

    /** Non-blank lines (CSV) or candidate elements (XML) that had no usable draw. */
    public int skippedLines;

    /** Non-blank lines (CSV) or candidate elements (XML) examined. */
    public int totalLines;

    private int droppedWarnings;

    /** True when neither dated nor undated draws were found. */
    public boolean isEmpty() {
        return dated.isEmpty() && undated.isEmpty();
    }

    /** dated + undated. */
    public int drawCount() {
        return dated.size() + undated.size();
    }

    /** Number of warnings that were reported, including the ones not kept in {@link #warnings}. */
    public int warningCount() {
        return (warnings.size() - (droppedWarnings > 0 ? 1 : 0)) + droppedWarnings;
    }

    /** Adds a warning, keeping at most {@link #MAX_WARNINGS} plus one "... and N more" line. */
    public void addWarning(String warning) {
        if (warning == null) {
            return;
        }
        if (droppedWarnings == 0 && warnings.size() < MAX_WARNINGS) {
            warnings.add(warning);
            return;
        }
        droppedWarnings++;
        String summary = "... and " + droppedWarnings + " more";
        if (droppedWarnings == 1 || warnings.isEmpty()) {
            warnings.add(summary);
        } else {
            warnings.set(warnings.size() - 1, summary);
        }
    }

    /** Puts a general note at the top of the warnings list. */
    void addWarningFirst(String warning) {
        if (warning != null) {
            warnings.add(0, warning);
        }
    }

    @Override
    public String toString() {
        return "ParseResult{dated=" + dated.size() + ", undated=" + undated.size()
                + ", skipped=" + skippedLines + ", total=" + totalLines
                + ", warnings=" + warningCount() + "}";
    }
}
