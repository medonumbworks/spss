package com.medonumbworks.csprngpredictor.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Test;

/**
 * IssuedPrediction / IssuedLog (SPEC 1.12): CSV round trip including pending entries,
 * settling against the first draw after the basis, lastSettled, pendingFor and stats.
 */
public class IssuedLogTest {

    private static final int[] A = {1, 4, 7, 9, 11, 13, 15, 17, 19, 21, 22};
    private static final int[] B = {2, 5, 6, 8, 9, 12, 13, 17, 19, 20, 22};
    private static final int[] C = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    private static final int[] D = {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

    private static IssuedPrediction issued(String issuedAt, String basedOn, int count, String target, int[] nums) {
        return new IssuedPrediction(issuedAt, basedOn, count, target, nums);
    }

    private static void assertSameEntry(IssuedPrediction e, IssuedPrediction a) {
        assertEquals(e.issuedAt, a.issuedAt);
        assertEquals(e.basedOnDate, a.basedOnDate);
        assertEquals(e.basedOnCount, a.basedOnCount);
        assertEquals(e.targetDate, a.targetDate);
        assertArrayEquals(e.numbers, a.numbers);
        assertEquals(e.isPending(), a.isPending());
        assertEquals(e.actualDate, a.actualDate);
        assertArrayEquals(e.actual, a.actual);
        assertEquals(e.hits, a.hits);
    }

    private static IssuedLog sampleLog() {
        IssuedLog log = new IssuedLog();
        IssuedPrediction a = issued("2026-10-14", "2026-10-13", 10, "2026-10-14", A);
        a.settle(new Draw("2026-10-14", B));
        IssuedPrediction b = issued("2026-10-15", "2026-10-14", 11, "2026-10-15", C);
        b.settle(new Draw("2026-10-15", C));
        IssuedPrediction c = issued("2026-10-16", "2026-10-15", 12, "2026-10-16", D);
        log.add(a);
        log.add(b);
        log.add(c);
        return log;
    }

    // ================================================================== IssuedPrediction

    @Test
    public void newEntryIsPending() {
        IssuedPrediction p = issued("2026-10-16", "2026-10-15", 3, "2026-10-16", A);
        assertTrue(p.isPending());
        assertNull(p.actual);
        assertNull(p.actualDate);
        assertEquals(-1, p.hits);
        assertArrayEquals(A, p.numbers);
    }

    @Test
    public void numbersAreSortedCopies() {
        int[] in = {22, 1, 19, 4, 17, 7, 15, 9, 13, 11, 21};
        IssuedPrediction p = issued("2026-10-16", "", 0, "2026-10-16", in);
        assertArrayEquals(A, p.numbers);
        in[0] = 3;
        assertArrayEquals(A, p.numbers);
    }

    @Test(expected = IllegalArgumentException.class)
    public void invalidNumbersAreRejected() {
        issued("2026-10-16", "", 0, "2026-10-16", new int[] {1, 2, 3});
    }

    @Test
    public void settleFillsActualAndHits() {
        IssuedPrediction p = issued("2026-10-16", "2026-10-15", 3, "2026-10-16", A);
        p.settle(new Draw("2026-10-16", B));
        assertFalse(p.isPending());
        assertEquals("2026-10-16", p.actualDate);
        assertArrayEquals(B, p.actual);
        assertEquals(5, p.hits);
        assertArrayEquals(new int[] {9, 13, 17, 19, 22}, p.hitNumbers());
    }

    @Test
    public void ofPrediction() {
        List<Draw> ds = new ArrayList<Draw>();
        ds.add(new Draw("2026-10-15", A));
        ds.add(new Draw("2026-10-16", B));
        Prediction next = Replay.run(ds, new ModelParams()).next;
        IssuedPrediction p = IssuedPrediction.of(next, "2026-10-16");
        assertEquals("2026-10-16", p.issuedAt);
        assertEquals("2026-10-16", p.basedOnDate);
        assertEquals(2, p.basedOnCount);
        assertEquals("2026-10-17", p.targetDate);
        assertArrayEquals(next.numbers, p.numbers);
        assertTrue(p.isPending());
        assertEquals(DateUtil.today(), IssuedPrediction.of(next, null).issuedAt);
    }

    // ================================================================== CSV

    @Test
    public void csvHeader() {
        String csv = new IssuedLog().toCsv();
        assertEquals("issuedAt,basedOnDate,basedOnCount,targetDate,numbers,actualDate,actual,hits",
                csv.split("\n")[0]);
        assertTrue(IssuedLog.fromCsv(csv).all().isEmpty());
    }

    @Test
    public void csvLayout() {
        String[] lines = sampleLog().toCsv().split("\n");
        assertEquals(4, lines.length);
        assertEquals("2026-10-14,2026-10-13,10,2026-10-14,1 4 7 9 11 13 15 17 19 21 22,"
                + "2026-10-14,2 5 6 8 9 12 13 17 19 20 22,5", lines[1]);
        // pending: empty actualDate, actual and hits cells
        String[] cells = lines[3].split(",", -1);
        assertEquals(8, cells.length);
        assertEquals("12 13 14 15 16 17 18 19 20 21 22", cells[4]);
        assertEquals("", cells[5]);
        assertEquals("", cells[6]);
        assertEquals("", cells[7]);
    }

    @Test
    public void csvRoundTripIncludingPendingEntries() {
        IssuedLog log = sampleLog();
        IssuedLog back = IssuedLog.fromCsv(log.toCsv());
        assertEquals(3, back.all().size());
        for (int i = 0; i < 3; i++) {
            assertSameEntry(log.all().get(i), back.all().get(i));
        }
        assertTrue(back.all().get(2).isPending());
        assertEquals(-1, back.all().get(2).hits);
        assertEquals(log.toCsv(), back.toCsv());
    }

    @Test
    public void csvRoundTripWithEmptyBasis() {
        IssuedLog log = new IssuedLog();
        log.add(issued("2026-10-16", "", 0, "2026-10-16", A));
        IssuedLog back = IssuedLog.fromCsv(log.toCsv());
        assertEquals(1, back.all().size());
        assertEquals("", back.all().get(0).basedOnDate);
        assertEquals(0, back.all().get(0).basedOnCount);
        assertTrue(back.all().get(0).isPending());
    }

    @Test
    public void csvRoundTripOfManyEntries() {
        Random rnd = new Random(4);
        IssuedLog log = new IssuedLog();
        String day = "2025-01-01";
        for (int i = 0; i < 200; i++) {
            int[] nums = i % 2 == 0 ? A : D;
            IssuedPrediction p = issued(day, DateUtil.addDays(day, -1), i, day, nums);
            if (rnd.nextBoolean()) {
                p.settle(new Draw(day, i % 3 == 0 ? B : C));
            }
            log.add(p);
            day = DateUtil.addDays(day, 1);
        }
        IssuedLog back = IssuedLog.fromCsv(log.toCsv());
        assertEquals(200, back.all().size());
        for (int i = 0; i < 200; i++) {
            assertSameEntry(log.all().get(i), back.all().get(i));
        }
    }

    @Test
    public void fromCsvIsTolerant() {
        assertTrue(IssuedLog.fromCsv(null).all().isEmpty());
        assertTrue(IssuedLog.fromCsv("").all().isEmpty());
        assertTrue(IssuedLog.fromCsv("garbage\n,,,\n1,2,3").all().isEmpty());
        String good = sampleLog().toCsv();
        String messy = "﻿" + good.replace("\n", "\r\n")
                + "\r\n"
                + "2026-10-17,2026-10-16,13,2026-10-17,1 2 3,,,\r\n"                // too few numbers
                + "2026-10-17,2026-10-16,x,2026-10-17\r\n"                         // too few cells
                + "# comment\r\n";
        IssuedLog back = IssuedLog.fromCsv(messy);
        assertEquals(3, back.all().size());
        assertEquals(5, back.all().get(0).hits);
        assertTrue(back.all().get(2).isPending());
    }

    @Test
    public void fromCsvAcceptsASpreadsheetResave() {
        String csv = "\"issuedAt\",\"basedOnDate\",\"basedOnCount\",\"targetDate\",\"numbers\",\"actualDate\",\"actual\",\"hits\"\n"
                + "\"2026-10-14\",\"2026-10-13\",\"10\",\"2026-10-14\",\"1 4 7 9 11 13 15 17 19 21 22\","
                + "\"2026-10-14\",\"2 5 6 8 9 12 13 17 19 20 22\",\"5\"\n"
                + "\"2026-10-16\",\"2026-10-15\",\"12\",\"2026-10-16\",\"12 13 14 15 16 17 18 19 20 21 22\",\"\",\"\",\"\"\n";
        IssuedLog log = IssuedLog.fromCsv(csv);
        assertEquals(2, log.all().size());
        assertSameEntry(sampleLog().all().get(0), log.all().get(0));
        assertSameEntry(sampleLog().all().get(2), log.all().get(1));
    }

    @Test
    public void fromCsvRecomputesHits() {
        // a hand-edited hits cell is not trusted
        String csv = IssuedLog.HEADER + "\n"
                + "2026-10-14,2026-10-13,10,2026-10-14,1 4 7 9 11 13 15 17 19 21 22,"
                + "2026-10-14,2 5 6 8 9 12 13 17 19 20 22,11\n";
        assertEquals(5, IssuedLog.fromCsv(csv).all().get(0).hits);
    }

    @Test
    public void fromCsvNeverThrowsOnRandomInput() {
        Random rnd = new Random(8);
        String alphabet = "0123456789 ,-\n\"";
        for (int round = 0; round < 100; round++) {
            StringBuilder sb = new StringBuilder(IssuedLog.HEADER + "\n");
            for (int i = 0; i < 300; i++) {
                sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
            }
            IssuedLog log = IssuedLog.fromCsv(sb.toString());
            assertNotNull(log.toCsv());
            assertNotNull(log.stats());
        }
    }

    // ================================================================== settle

    private static DrawSet draws() {
        DrawSet s = new DrawSet();
        s.add(new Draw("2026-10-14", B));
        s.add(new Draw("2026-10-16", C));
        s.add(new Draw("2026-10-18", D));
        return s;
    }

    @Test
    public void settlePicksTheFirstDrawAfterBasedOnDate() {
        IssuedLog log = new IssuedLog();
        IssuedPrediction p15 = log.add(issued("2026-10-15", "2026-10-15", 2, "2026-10-16", A));
        IssuedPrediction p16 = log.add(issued("2026-10-16", "2026-10-16", 3, "2026-10-17", A));
        IssuedPrediction p18 = log.add(issued("2026-10-18", "2026-10-18", 4, "2026-10-19", A));
        IssuedPrediction p13 = log.add(issued("2026-10-13", "2026-10-13", 1, "2026-10-14", A));
        assertEquals(3, log.settle(draws()));
        assertEquals("2026-10-16", p15.actualDate);
        assertArrayEquals(C, p15.actual);
        assertEquals(new Draw("2026-10-16", C).hitsAgainst(A), p15.hits);
        assertEquals("never a draw on the basis date itself", "2026-10-18", p16.actualDate);
        assertArrayEquals(D, p16.actual);
        assertEquals(new Draw("2026-10-18", D).hitsAgainst(A), p16.hits);
        assertTrue("no draw after the latest one yet", p18.isPending());
        assertEquals("2026-10-14", p13.actualDate);
        assertEquals(5, p13.hits);
    }

    @Test
    public void settleWithEmptyBasisUsesTheEarliestDraw() {
        IssuedLog log = new IssuedLog();
        IssuedPrediction p = log.add(issued("2026-10-10", "", 0, "2026-10-10", A));
        assertEquals(1, log.settle(draws()));
        assertEquals("2026-10-14", p.actualDate);
        assertArrayEquals(B, p.actual);
    }

    @Test
    public void settledEntriesAreNotSettledAgain() {
        IssuedLog log = new IssuedLog();
        IssuedPrediction p = log.add(issued("2026-10-15", "2026-10-15", 2, "2026-10-16", A));
        DrawSet ds = draws();
        assertEquals(1, log.settle(ds));
        ds.add(new Draw("2026-10-15", D));
        ds.add(new Draw("2026-10-16", A));
        assertEquals(0, log.settle(ds));
        assertEquals("2026-10-16", p.actualDate);
        assertArrayEquals(C, p.actual);
    }

    @Test
    public void settleLaterWhenTheDrawArrives() {
        IssuedLog log = new IssuedLog();
        IssuedPrediction p = log.add(issued("2026-10-18", "2026-10-18", 3, "2026-10-19", A));
        DrawSet ds = draws();
        assertEquals(0, log.settle(ds));
        assertTrue(p.isPending());
        assertEquals(0, log.settle(new DrawSet()));
        ds.add(new Draw("2026-10-19", A));
        assertEquals(1, log.settle(ds));
        assertEquals(11, p.hits);
        assertEquals(0, log.settle(null));
    }

    // ================================================================== queries

    @Test
    public void lastSettled() {
        IssuedLog log = new IssuedLog();
        assertNull(log.lastSettled());
        IssuedPrediction a = log.add(issued("2026-10-13", "2026-10-13", 1, "2026-10-14", A));
        IssuedPrediction b = log.add(issued("2026-10-15", "2026-10-15", 2, "2026-10-16", B));
        log.add(issued("2026-10-18", "2026-10-18", 3, "2026-10-19", C));
        assertNull(log.lastSettled());
        log.settle(draws());
        assertTrue(a != null && !a.isPending());
        assertSame("most recent settled in issue order (the third is still pending)", b, log.lastSettled());
    }

    @Test
    public void pendingFor() {
        IssuedLog log = new IssuedLog();
        IssuedPrediction p = log.add(issued("2026-10-15", "2026-10-15", 2, "2026-10-16", A));
        assertSame(p, log.pendingFor("2026-10-15", 2));
        assertNull("different count", log.pendingFor("2026-10-15", 3));
        assertNull("different date", log.pendingFor("2026-10-14", 2));
        log.settle(draws());
        assertNull("settled entries are not pending", log.pendingFor("2026-10-15", 2));
        IssuedPrediction e = log.add(issued("2026-10-10", "", 0, "2026-10-10", A));
        assertSame(e, log.pendingFor("", 0));
    }

    @Test
    public void addDoesNotDedupe() {
        IssuedLog log = new IssuedLog();
        IssuedPrediction p = issued("2026-10-15", "2026-10-15", 2, "2026-10-16", A);
        assertSame(p, log.add(p));
        log.add(issued("2026-10-15", "2026-10-15", 2, "2026-10-16", A));
        assertEquals(2, log.all().size());
        assertSame(p, log.all().get(0));
    }

    @Test
    public void allIsInIssueOrder() {
        IssuedLog log = new IssuedLog();
        IssuedPrediction x = log.add(issued("2026-10-18", "2026-10-17", 3, "2026-10-18", A));
        IssuedPrediction y = log.add(issued("2026-10-10", "2026-10-09", 1, "2026-10-10", B));
        assertSame(x, log.all().get(0));
        assertSame(y, log.all().get(1));
    }

    @Test
    public void statsOverSettledEntries() {
        IssuedLog log = sampleLog();          // hits 5 and 11, one pending
        Stats s = log.stats();
        assertEquals(2, s.n);
        assertEquals(8.0, s.meanHits, 1e-12);
        assertEquals(11, s.best);
        assertEquals(5, s.worst);
        assertEquals(1, s.histogram[5]);
        assertEquals(1, s.histogram[11]);
        assertEquals(0, new IssuedLog().stats().n);
        assertEquals(s.meanHits, IssuedLog.fromCsv(log.toCsv()).stats().meanHits, 0.0);
    }

    @Test
    public void statsIgnorePending() {
        IssuedLog log = new IssuedLog();
        log.add(issued("2026-10-15", "2026-10-15", 2, "2026-10-16", A));
        log.add(issued("2026-10-16", "2026-10-16", 3, "2026-10-17", B));
        assertEquals(0, log.stats().n);
        log.settle(draws());
        Stats s = log.stats();
        assertEquals(2, s.n);
        double expected = (new Draw("2026-10-16", C).hitsAgainst(A) + new Draw("2026-10-18", D).hitsAgainst(B)) / 2.0;
        assertEquals(expected, s.meanHits, 1e-12);
    }

    @Test
    public void clear() {
        IssuedLog log = sampleLog();
        log.clear();
        assertTrue(log.all().isEmpty());
        assertNull(log.lastSettled());
        assertEquals(0, log.stats().n);
        assertEquals(IssuedLog.HEADER + "\n", log.toCsv());
    }
}
