package com.medonumbworks.csprngpredictor.core;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;

/**
 * The user's draw history: sorted by date ascending (same date: insertion order) and free of
 * duplicates (equal date and numbers). Thread-safe (all methods synchronized).
 */
public final class DrawSet {

    private static final Comparator<Draw> BY_DATE = new Comparator<Draw>() {
        @Override
        public int compare(Draw a, Draw b) {
            return a.date.compareTo(b.date);
        }
    };

    private final ArrayList<Draw> list = new ArrayList<Draw>();
    private final HashSet<Draw> set = new HashSet<Draw>();

    public DrawSet() {
    }

    public DrawSet(Collection<Draw> initial) {
        addAll(initial);
    }

    /** False if an equal draw already exists (or {@code d} is null). */
    public synchronized boolean add(Draw d) {
        if (d == null || set.contains(d)) {
            return false;
        }
        set.add(d);
        list.add(upperBound(d.date), d);
        return true;
    }

    /** Adds all new draws; returns how many were actually added. */
    public synchronized int addAll(Collection<Draw> ds) {
        if (ds == null || ds.isEmpty()) {
            return 0;
        }
        ArrayList<Draw> fresh = new ArrayList<Draw>();
        for (Draw d : ds) {
            if (d != null && set.add(d)) {
                fresh.add(d);
            }
        }
        if (fresh.size() <= 16) {
            for (Draw d : fresh) {
                list.add(upperBound(d.date), d);
            }
        } else {
            // Append and stable-sort by date: keeps insertion order within a date, O(n log n).
            list.addAll(fresh);
            Collections.sort(list, BY_DATE);
        }
        return fresh.size();
    }

    public synchronized boolean remove(Draw d) {
        if (d == null || !set.remove(d)) {
            return false;
        }
        list.remove(d);
        return true;
    }

    public synchronized void clear() {
        list.clear();
        set.clear();
    }

    public synchronized int size() {
        return list.size();
    }

    public synchronized boolean isEmpty() {
        return list.isEmpty();
    }

    /** Unmodifiable snapshot, chronological (oldest first). */
    public synchronized List<Draw> all() {
        return Collections.unmodifiableList(new ArrayList<Draw>(list));
    }

    /** Snapshot, newest first. */
    public synchronized List<Draw> newestFirst() {
        ArrayList<Draw> out = new ArrayList<Draw>(list.size());
        for (int i = list.size() - 1; i >= 0; i--) {
            out.add(list.get(i));
        }
        return out;
    }

    /** The chronologically last draw, or null if empty. */
    public synchronized Draw latest() {
        return list.isEmpty() ? null : list.get(list.size() - 1);
    }

    /** The chronologically first draw, or null if empty. */
    public synchronized Draw earliest() {
        return list.isEmpty() ? null : list.get(0);
    }

    /** True when an equal draw exists. */
    public synchronized boolean contains(Draw d) {
        return d != null && set.contains(d);
    }

    /** All draws on that date, in insertion order. */
    public synchronized List<Draw> onDate(String iso) {
        ArrayList<Draw> out = new ArrayList<Draw>();
        if (iso == null) {
            return out;
        }
        for (int i = lowerBound(iso); i < list.size() && list.get(i).date.equals(iso); i++) {
            out.add(list.get(i));
        }
        return out;
    }

    /** Earliest draw with date &gt; iso (any draw for null / ""), or null. */
    public synchronized Draw firstAfter(String iso) {
        int i = upperBound(iso == null ? "" : iso);
        return i < list.size() ? list.get(i) : null;
    }

    /** First index whose date is &gt;= iso. */
    private int lowerBound(String iso) {
        int lo = 0;
        int hi = list.size();
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (list.get(mid).date.compareTo(iso) < 0) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /** First index whose date is &gt; iso. */
    private int upperBound(String iso) {
        int lo = 0;
        int hi = list.size();
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (list.get(mid).date.compareTo(iso) <= 0) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
}
