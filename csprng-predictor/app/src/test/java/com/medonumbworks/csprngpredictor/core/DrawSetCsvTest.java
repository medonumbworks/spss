package com.medonumbworks.csprngpredictor.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.junit.Test;

/**
 * Game / Draw value rules, the DrawCsv storage format (SPEC 1.5) and DrawSet ordering,
 * deduplication and queries.
 */
public class DrawSetCsvTest {

    private static final int[] A = {1, 4, 7, 9, 11, 13, 15, 17, 19, 21, 22};
    private static final int[] B = {2, 5, 6, 8, 9, 12, 13, 17, 19, 20, 22};
    private static final int[] C = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    private static Draw d(String date, int[] nums) {
        return new Draw(date, nums);
    }

    private static int[] randomDraw(Random rnd) {
        int[] pool = new int[22];
        for (int i = 0; i < 22; i++) {
            pool[i] = i + 1;
        }
        for (int i = 21; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = pool[i];
            pool[i] = pool[j];
            pool[j] = t;
        }
        return Arrays.copyOf(pool, 11);
    }

    // ================================================================== Game / Draw

    @Test
    public void gameConstants() {
        assertEquals(22, Game.MAX_NUMBER);
        assertEquals(11, Game.PICK);
        assertEquals(5.5, Game.EXPECTED_HITS, 0.0);
        assertTrue(Game.isValidNumber(1));
        assertTrue(Game.isValidNumber(22));
        assertFalse(Game.isValidNumber(0));
        assertFalse(Game.isValidNumber(23));
        assertFalse(Game.isValidNumber(-1));
    }

    @Test
    public void gameIsValidDraw() {
        assertTrue(Game.isValidDraw(A));
        assertTrue(Game.isValidDraw(new int[] {22, 1, 19, 4, 17, 7, 15, 9, 13, 11, 21}));
        assertFalse(Game.isValidDraw(null));
        assertFalse(Game.isValidDraw(new int[0]));
        assertFalse(Game.isValidDraw(Arrays.copyOf(A, 10)));
        assertFalse(Game.isValidDraw(new int[] {1, 1, 7, 9, 11, 13, 15, 17, 19, 21, 22}));
        assertFalse(Game.isValidDraw(new int[] {0, 4, 7, 9, 11, 13, 15, 17, 19, 21, 22}));
        assertFalse(Game.isValidDraw(new int[] {1, 4, 7, 9, 11, 13, 15, 17, 19, 21, 23}));
        assertFalse(Game.isValidDraw(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
    }

    @Test
    public void drawSortsAndCopiesNumbers() {
        int[] in = {22, 1, 19, 4, 17, 7, 15, 9, 13, 11, 21};
        Draw x = d("2026-10-16", in);
        assertArrayEquals(A, x.numbers);
        in[0] = 5;
        assertArrayEquals("defensive copy", A, x.numbers);
        assertEquals("2026-10-16", x.date);
    }

    @Test
    public void drawRejectsInvalidInput() {
        String[] badDates = {null, "", "16.10.2026", "2026-02-30", "2026-1-5"};
        for (String bd : badDates) {
            try {
                d(bd, A);
                fail("accepted date " + bd);
            } catch (IllegalArgumentException expected) {
                // ok
            }
        }
        int[][] badNums = {null, new int[0], Arrays.copyOf(A, 10), {1, 1, 7, 9, 11, 13, 15, 17, 19, 21, 22},
                {0, 4, 7, 9, 11, 13, 15, 17, 19, 21, 22}, {1, 4, 7, 9, 11, 13, 15, 17, 19, 21, 23}};
        for (int[] bn : badNums) {
            try {
                d("2026-10-16", bn);
                fail("accepted numbers " + Arrays.toString(bn));
            } catch (IllegalArgumentException expected) {
                // ok
            }
        }
    }

    @Test
    public void drawQueries() {
        Draw x = d("2026-10-16", A);
        assertTrue(x.contains(1));
        assertTrue(x.contains(22));
        assertFalse(x.contains(2));
        assertFalse(x.contains(0));
        assertEquals("1 4 7 9 11 13 15 17 19 21 22", x.numbersString(" "));
        assertEquals("1,4,7,9,11,13,15,17,19,21,22", x.numbersString(","));
        assertEquals("2026-10-16: 1 4 7 9 11 13 15 17 19 21 22", x.toString());
    }

    @Test
    public void drawHitsAgainst() {
        Draw x = d("2026-10-16", A);
        assertEquals(11, x.hitsAgainst(A));
        assertEquals(5, x.hitsAgainst(B));          // 9, 13, 17, 19, 22
        assertEquals(1, x.hitsAgainst(new int[] {1, 1, 1}));   // duplicates count once
        assertEquals(2, x.hitsAgainst(new int[] {0, 1, 23, 22, -4}));
        assertEquals(0, x.hitsAgainst(new int[] {2, 3, 5, 6, 8, 10, 12, 14, 16, 18, 20}));
        assertEquals(0, x.hitsAgainst(new int[0]));
        assertEquals(0, x.hitsAgainst(null));
    }

    @Test
    public void drawEqualsHashCodeCompareTo() {
        Draw a1 = d("2026-10-16", A);
        Draw a2 = d("2026-10-16", A.clone());
        Draw a3 = d("2026-10-17", A);
        Draw b1 = d("2026-10-16", B);
        assertEquals(a1, a2);
        assertEquals(a1.hashCode(), a2.hashCode());
        assertNotEquals(a1, a3);
        assertNotEquals(a1, b1);
        assertEquals(0, a1.compareTo(a2));
        assertTrue(a1.compareTo(a3) < 0);
        assertTrue(a3.compareTo(a1) > 0);
        // same date: numbers lexicographically (A starts 1,4 ; B starts 2,5)
        assertTrue(a1.compareTo(b1) < 0);
        assertTrue(b1.compareTo(a1) > 0);
        Draw c1 = d("2026-10-16", C);
        assertTrue(c1.compareTo(a1) < 0);         // 1,2 < 1,4
    }

    // ================================================================== DrawCsv

    @Test
    public void writeFormat() {
        List<Draw> list = new ArrayList<Draw>();
        list.add(d("2026-10-15", B));
        list.add(d("2026-10-16", A));
        String csv = DrawCsv.write(list);
        assertEquals(DrawCsv.HEADER + "\n"
                + "2026-10-15,2,5,6,8,9,12,13,17,19,20,22\n"
                + "2026-10-16,1,4,7,9,11,13,15,17,19,21,22\n", csv);
        assertFalse(csv.contains("\r"));
    }

    @Test
    public void headerConstant() {
        assertEquals("date,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11", DrawCsv.HEADER);
    }

    @Test
    public void writeEmpty() {
        assertEquals(DrawCsv.HEADER + "\n", DrawCsv.write(new ArrayList<Draw>()));
    }

    @Test
    public void roundTrip() {
        Random rnd = new Random(5);
        List<Draw> list = new ArrayList<Draw>();
        String day = "2024-02-25";
        for (int i = 0; i < 300; i++) {
            list.add(d(day, randomDraw(rnd)));
            if (i % 3 != 0) {
                day = DateUtil.addDays(day, 1);       // some days hold several draws
            }
        }
        List<Draw> back = DrawCsv.read(DrawCsv.write(list));
        assertEquals(list, back);
    }

    @Test
    public void roundTripThroughDrawSet() {
        DrawSet set = new DrawSet();
        set.add(d("2026-10-16", A));
        set.add(d("2026-10-14", B));
        set.add(d("2026-10-15", C));
        DrawSet back = new DrawSet(DrawCsv.read(DrawCsv.write(set.all())));
        assertEquals(set.all(), back.all());
    }

    @Test
    public void readIsTolerantOfHeaderBlankLinesCrlfAndBom() {
        String csv = "﻿" + DrawCsv.HEADER + "\r\n"
                + "\r\n"
                + "2026-10-15,2,5,6,8,9,12,13,17,19,20,22\r\n"
                + "   \n"
                + "# a comment\n"
                + " 2026-10-16 , 1 , 4 , 7 , 9 , 11 , 13 , 15 , 17 , 19 , 21 , 22 \n"
                + "2026-10-17,1,2,3,4,5,6,7,8,9,10,11,";
        List<Draw> out = DrawCsv.read(csv);
        assertEquals(3, out.size());
        assertEquals(d("2026-10-15", B), out.get(0));
        assertEquals(d("2026-10-16", A), out.get(1));
        assertEquals(d("2026-10-17", C), out.get(2));
    }

    @Test
    public void readSkipsInvalidLines() {
        String csv = DrawCsv.HEADER + "\n"
                + "2026-10-15,2,5,6,8,9,12,13,17,19,20,22\n"
                + "2026-02-30,2,5,6,8,9,12,13,17,19,20,22\n"      // invalid date
                + "2026-10-16,1,4,7,9,11,13,15,17,19,21\n"        // 10 numbers
                + "2026-10-16,1,4,7,9,11,13,15,17,19,21,21\n"     // duplicate
                + "2026-10-16,1,4,7,9,11,13,15,17,19,21,23\n"     // out of range
                + "2026-10-16,1,4,7,9,11,13,15,17,19,21,22,5\n"   // extra column
                + "2026-10-16;1;4;7;9;11;13;15;17;19;21;22\n"     // not the own format
                + "16.10.2026,1,4,7,9,11,13,15,17,19,21,22\n"     // not ISO
                + "garbage\n"
                + "2026-10-16,1,4,7,9,11,13,15,17,19,21,x\n"
                + "2026-10-17,1,2,3,4,5,6,7,8,9,10,11\n";
        List<String> problems = new ArrayList<String>();
        List<Draw> out = DrawCsv.read(csv, problems);
        assertEquals(2, out.size());
        assertEquals(d("2026-10-15", B), out.get(0));
        assertEquals(d("2026-10-17", C), out.get(1));
        assertEquals(9, problems.size());
        assertTrue(problems.get(0), problems.get(0).startsWith("line 3"));
    }

    @Test
    public void readNeverThrows() {
        assertTrue(DrawCsv.read(null).isEmpty());
        assertTrue(DrawCsv.read("").isEmpty());
        assertTrue(DrawCsv.read(DrawCsv.HEADER).isEmpty());
        assertTrue(DrawCsv.read(",,,,,,,,,,,,,,,").isEmpty());
        Random rnd = new Random(3);
        String alphabet = "0123456789,-\n\r date";
        for (int round = 0; round < 50; round++) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 500; i++) {
                sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
            }
            for (Draw x : DrawCsv.read(sb.toString())) {
                assertTrue(Game.isValidDraw(x.numbers));
            }
        }
    }

    // ================================================================== DrawSet basics

    @Test
    public void emptySet() {
        DrawSet s = new DrawSet();
        assertTrue(s.isEmpty());
        assertEquals(0, s.size());
        assertNull(s.latest());
        assertTrue(s.all().isEmpty());
        assertTrue(s.newestFirst().isEmpty());
        assertTrue(s.onDate("2026-10-16").isEmpty());
        assertNull(s.firstAfter("2026-10-16"));
    }

    @Test
    public void addDedupes() {
        DrawSet s = new DrawSet();
        assertTrue(s.add(d("2026-10-16", A)));
        assertFalse(s.add(d("2026-10-16", A.clone())));
        assertEquals(1, s.size());
        assertTrue("same date, other numbers is another draw", s.add(d("2026-10-16", B)));
        assertTrue("same numbers, other date is another draw", s.add(d("2026-10-17", A)));
        assertEquals(3, s.size());
        assertFalse(s.add(null));
    }

    @Test
    public void addAllCountsOnlyNewDraws() {
        DrawSet s = new DrawSet();
        s.add(d("2026-10-16", A));
        List<Draw> more = new ArrayList<Draw>();
        more.add(d("2026-10-16", A));
        more.add(d("2026-10-17", B));
        more.add(d("2026-10-17", B));
        more.add(d("2026-10-15", C));
        assertEquals(2, s.addAll(more));
        assertEquals(3, s.size());
        assertEquals(0, s.addAll(more));
        assertEquals(0, s.addAll(new ArrayList<Draw>()));
    }

    @Test
    public void constructorDedupesAndSorts() {
        List<Draw> init = new ArrayList<Draw>();
        init.add(d("2026-10-17", A));
        init.add(d("2026-10-15", B));
        init.add(d("2026-10-17", A));
        init.add(d("2026-10-16", C));
        DrawSet s = new DrawSet(init);
        assertEquals(3, s.size());
        assertEquals("2026-10-15", s.all().get(0).date);
        assertEquals("2026-10-16", s.all().get(1).date);
        assertEquals("2026-10-17", s.all().get(2).date);
    }

    @Test
    public void chronologicalOrderRegardlessOfInsertion() {
        DrawSet s = new DrawSet();
        s.add(d("2026-10-18", A));
        s.add(d("2026-10-14", B));
        s.add(d("2026-10-16", C));
        s.add(d("2025-12-31", A));
        s.add(d("2027-01-01", B));
        List<Draw> all = s.all();
        for (int i = 1; i < all.size(); i++) {
            assertTrue(all.get(i - 1).date.compareTo(all.get(i).date) <= 0);
        }
        assertEquals("2025-12-31", all.get(0).date);
        assertEquals("2027-01-01", s.latest().date);
    }

    @Test
    public void sameDateKeepsInsertionOrder() {
        DrawSet s = new DrawSet();
        Draw x = d("2026-10-16", B);
        Draw y = d("2026-10-16", A);
        Draw z = d("2026-10-16", C);
        s.add(d("2026-10-17", A));
        s.add(x);
        s.add(d("2026-10-15", A));
        s.add(y);
        s.add(z);
        assertEquals(Arrays.asList(x, y, z), s.onDate("2026-10-16"));
        List<Draw> all = s.all();
        assertSame(x, all.get(1));
        assertSame(y, all.get(2));
        assertSame(z, all.get(3));
        assertSame("latest of the same day = last inserted", z, new DrawSet(Arrays.asList(x, y, z)).latest());
    }

    @Test
    public void sameDateKeepsInsertionOrderWithBulkAdd() {
        // > 16 new draws take the bulk (sort) path of addAll
        Random rnd = new Random(11);
        List<Draw> bulk = new ArrayList<Draw>();
        for (int i = 0; i < 60; i++) {
            String date = DateUtil.addDays("2026-01-01", (i * 7) % 5);
            bulk.add(d(date, randomDraw(rnd)));
        }
        DrawSet s = new DrawSet();
        s.add(d("2026-01-03", A));
        int added = s.addAll(bulk);
        List<Draw> expected = new ArrayList<Draw>();
        expected.add(d("2026-01-03", A));
        for (Draw x : bulk) {
            if (!expected.contains(x)) {
                expected.add(x);
            }
        }
        assertEquals(expected.size() - 1, added);
        // expected order: stable sort by date of (existing, then bulk in order)
        final List<Draw> sorted = new ArrayList<Draw>(expected);
        Collections.sort(sorted, new java.util.Comparator<Draw>() {
            @Override
            public int compare(Draw p, Draw q) {
                return p.date.compareTo(q.date);
            }
        });
        assertEquals(sorted, s.all());
    }

    @Test
    public void allIsUnmodifiableSnapshot() {
        DrawSet s = new DrawSet();
        s.add(d("2026-10-16", A));
        List<Draw> snap = s.all();
        try {
            snap.add(d("2026-10-17", A));
            fail("all() must be unmodifiable");
        } catch (UnsupportedOperationException expected) {
            // ok
        }
        s.add(d("2026-10-17", B));
        assertEquals(1, snap.size());
        assertEquals(2, s.all().size());
    }

    @Test
    public void newestFirstIsReversed() {
        DrawSet s = new DrawSet();
        Draw x = d("2026-10-14", A);
        Draw y = d("2026-10-16", B);
        Draw z = d("2026-10-15", C);
        s.add(x);
        s.add(y);
        s.add(z);
        assertEquals(Arrays.asList(y, z, x), s.newestFirst());
        List<Draw> rev = new ArrayList<Draw>(s.all());
        Collections.reverse(rev);
        assertEquals(rev, s.newestFirst());
    }

    @Test
    public void latestIsTheNewest() {
        DrawSet s = new DrawSet();
        s.add(d("2026-10-16", A));
        s.add(d("2026-10-14", B));
        assertEquals(d("2026-10-16", A), s.latest());
        s.add(d("2026-10-20", C));
        assertEquals(d("2026-10-20", C), s.latest());
    }

    @Test
    public void removeAndClear() {
        DrawSet s = new DrawSet();
        s.add(d("2026-10-16", A));
        s.add(d("2026-10-17", B));
        assertTrue(s.remove(d("2026-10-16", A.clone())));
        assertFalse(s.remove(d("2026-10-16", A)));
        assertFalse(s.remove(d("2026-10-18", A)));
        assertFalse(s.remove(null));
        assertEquals(1, s.size());
        assertTrue("re-adding a removed draw works", s.add(d("2026-10-16", A)));
        s.clear();
        assertTrue(s.isEmpty());
        assertNull(s.latest());
        assertTrue(s.add(d("2026-10-16", A)));
    }

    // ================================================================== onDate / firstAfter

    @Test
    public void onDate() {
        DrawSet s = new DrawSet();
        s.add(d("2026-10-15", A));
        s.add(d("2026-10-16", B));
        s.add(d("2026-10-16", C));
        s.add(d("2026-10-17", A));
        assertEquals(Arrays.asList(d("2026-10-16", B), d("2026-10-16", C)), s.onDate("2026-10-16"));
        assertEquals(1, s.onDate("2026-10-15").size());
        assertTrue(s.onDate("2026-10-18").isEmpty());
        assertTrue(s.onDate("2026-10-14").isEmpty());
    }

    @Test
    public void firstAfterIsStrictlyAfter() {
        DrawSet s = new DrawSet();
        Draw x = d("2026-10-14", A);
        Draw y1 = d("2026-10-16", B);
        Draw y2 = d("2026-10-16", C);
        Draw z = d("2026-10-20", A);
        s.add(z);
        s.add(y1);
        s.add(x);
        s.add(y2);
        assertEquals(x, s.firstAfter("2026-10-01"));
        assertEquals(x, s.firstAfter("2026-10-13"));
        assertEquals(y1, s.firstAfter("2026-10-14"));       // earliest of the next date, first inserted
        assertEquals(y1, s.firstAfter("2026-10-15"));
        assertEquals(z, s.firstAfter("2026-10-16"));        // never a draw on the same date
        assertEquals(z, s.firstAfter("2026-10-19"));
        assertNull(s.firstAfter("2026-10-20"));
        assertNull(s.firstAfter("2027-01-01"));
    }

    @Test
    public void firstAfterAcrossYears() {
        DrawSet s = new DrawSet();
        s.add(d("2025-12-31", A));
        s.add(d("2026-01-01", B));
        assertEquals(d("2026-01-01", B), s.firstAfter("2025-12-31"));
        assertEquals(d("2025-12-31", A), s.firstAfter("2025-01-01"));
    }

    @Test
    public void largeSetStaysSortedAndDeduplicated() {
        Random rnd = new Random(77);
        DrawSet s = new DrawSet();
        List<Draw> added = new ArrayList<Draw>();
        for (int i = 0; i < 2000; i++) {
            String date = DateUtil.addDays("2020-01-01", rnd.nextInt(400));
            Draw x = d(date, randomDraw(rnd));
            boolean fresh = !added.contains(x);
            assertEquals(fresh, s.add(x));
            if (fresh) {
                added.add(x);
            }
            if (i % 10 == 0) {
                assertFalse(s.add(added.get(rnd.nextInt(added.size()))));
            }
        }
        assertEquals(added.size(), s.size());
        List<Draw> all = s.all();
        for (int i = 1; i < all.size(); i++) {
            assertTrue(all.get(i - 1).date.compareTo(all.get(i).date) <= 0);
        }
    }
}
