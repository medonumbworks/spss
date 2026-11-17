package com.medonumbworks.csprngpredictor.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.junit.Test;

/**
 * The nine default experts (SPEC 1.6) and the History statistics they use, checked against a
 * straightforward brute-force reference implementation and a small hand-built history.
 */
public class StrategyTest {

    private static final double EPS = 1e-12;
    private static final String[] IDS = {"uniform", "freq", "recent10", "recent30", "decay", "gap",
            "transition", "repeat", "pairs"};

    // ------------------------------------------------------------------ fixtures

    private static int[] range(int from, int to) {
        int[] r = new int[to - from + 1];
        for (int i = 0; i < r.length; i++) {
            r[i] = from + i;
        }
        return r;
    }

    /** D0 = 1..11, D1 = 2..12, D2 = 3..13 on consecutive days. */
    private static List<Draw> handBuilt() {
        List<Draw> list = new ArrayList<Draw>();
        list.add(new Draw("2026-10-14", range(1, 11)));
        list.add(new Draw("2026-10-15", range(2, 12)));
        list.add(new Draw("2026-10-16", range(3, 13)));
        return list;
    }

    private static int[] randomDraw(Random rnd) {
        int[] pool = range(1, 22);
        for (int i = 21; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = pool[i];
            pool[i] = pool[j];
            pool[j] = t;
        }
        int[] d = Arrays.copyOf(pool, 11);
        Arrays.sort(d);
        return d;
    }

    private static List<Draw> randomDraws(int n, long seed) {
        Random rnd = new Random(seed);
        List<Draw> list = new ArrayList<Draw>();
        String day = "2020-01-01";
        for (int i = 0; i < n; i++) {
            list.add(new Draw(day, randomDraw(rnd)));
            day = DateUtil.addDays(day, 1);
        }
        return list;
    }

    private static History history(List<Draw> draws) {
        return new History(draws);
    }

    private static Strategy strategy(String id) {
        for (Strategy s : Strategies.defaults(new ModelParams())) {
            if (s.id().equals(id)) {
                return s;
            }
        }
        throw new AssertionError("no strategy " + id);
    }

    private static void assertValidScores(String what, double[] s) {
        assertNotNull(what, s);
        assertEquals(what, 22, s.length);
        double sum = 0;
        for (int i = 0; i < 22; i++) {
            assertFalse(what + " NaN at " + i, Double.isNaN(s[i]));
            assertFalse(what + " infinite at " + i, Double.isInfinite(s[i]));
            assertTrue(what + " negative at " + i + ": " + s[i], s[i] >= 0.0);
            sum += s[i];
        }
        assertTrue(what + " all zero", sum > 0.0);
    }

    private static void assertAll(String what, double expected, double[] s) {
        for (int i = 0; i < s.length; i++) {
            assertEquals(what + " [" + i + "]", expected, s[i], EPS);
        }
    }

    // ------------------------------------------------------------------ brute-force reference

    private static boolean in(Draw d, int number) {
        for (int x : d.numbers) {
            if (x == number) {
                return true;
            }
        }
        return false;
    }

    private static int[] refCounts(List<Draw> ds, int window) {
        int[] c = new int[22];
        for (int t = Math.max(0, ds.size() - window); t < ds.size(); t++) {
            for (int x : ds.get(t).numbers) {
                c[x - 1]++;
            }
        }
        return c;
    }

    private static double[] refDecay(List<Draw> ds, double halfLife) {
        double[] v = new double[22];
        int n = ds.size();
        for (int t = 0; t < n; t++) {
            int age = n - 1 - t;
            for (int x : ds.get(t).numbers) {
                v[x - 1] += Math.pow(0.5, age / halfLife);
            }
        }
        return v;
    }

    private static int[] refSince(List<Draw> ds) {
        int n = ds.size();
        int[] s = new int[22];
        for (int i = 0; i < 22; i++) {
            s[i] = n;
            for (int t = n - 1; t >= 0; t--) {
                if (in(ds.get(t), i + 1)) {
                    s[i] = n - 1 - t;
                    break;
                }
            }
        }
        return s;
    }

    private static int[][] refTransition(List<Draw> ds) {
        int[][] m = new int[22][22];
        for (int t = 1; t < ds.size(); t++) {
            for (int a : ds.get(t - 1).numbers) {
                for (int b : ds.get(t).numbers) {
                    m[a - 1][b - 1]++;
                }
            }
        }
        return m;
    }

    private static int[] refContains(List<Draw> ds) {
        int[] c = new int[22];
        for (int t = 0; t + 1 < ds.size(); t++) {
            for (int a : ds.get(t).numbers) {
                c[a - 1]++;
            }
        }
        return c;
    }

    private static double refRepeat(List<Draw> ds) {
        if (ds.size() < 2) {
            return 0.5;
        }
        double sum = 0;
        for (int t = 1; t < ds.size(); t++) {
            int common = 0;
            for (int x : ds.get(t).numbers) {
                if (in(ds.get(t - 1), x)) {
                    common++;
                }
            }
            sum += common / 11.0;
        }
        return sum / (ds.size() - 1);
    }

    private static int[][] refPairs(List<Draw> ds) {
        int[][] m = new int[22][22];
        for (Draw d : ds) {
            for (int a : d.numbers) {
                for (int b : d.numbers) {
                    if (a != b) {
                        m[a - 1][b - 1]++;
                    }
                }
            }
        }
        return m;
    }

    private static double[] refTransitionScores(List<Draw> ds) {
        double[] s = new double[22];
        if (ds.isEmpty()) {
            Arrays.fill(s, 1.0);
            return s;
        }
        int[][] t = refTransition(ds);
        int[] c = refContains(ds);
        Draw last = ds.get(ds.size() - 1);
        for (int j = 0; j < 22; j++) {
            double sum = 0;
            for (int a : last.numbers) {
                sum += (t[a - 1][j] + 1.0) / (c[a - 1] + 2.0);
            }
            s[j] = sum / last.numbers.length;
        }
        return s;
    }

    private static void assertHistoryMatchesReference(String what, History h, List<Draw> ds) {
        assertEquals(what, ds.size(), h.size());
        if (ds.isEmpty()) {
            assertNull(what, h.last());
        } else {
            assertEquals(what, ds.get(ds.size() - 1), h.last());
            assertEquals(what, ds.get(0), h.get(0));
        }
        assertArrayEquals(what + " counts", refCounts(ds, Integer.MAX_VALUE), h.counts());
        assertArrayEquals(what + " last10", refCounts(ds, 10), h.countsInLast(10));
        assertArrayEquals(what + " last3", refCounts(ds, 3), h.countsInLast(3));
        assertArrayEquals(what + " last1000", refCounts(ds, 1000), h.countsInLast(1000));
        assertArrayEquals(what + " decay", refDecay(ds, 15), h.decayedCounts(15), 1e-9);
        assertArrayEquals(what + " decay4", refDecay(ds, 4), h.decayedCounts(4), 1e-9);
        assertArrayEquals(what + " since", refSince(ds), h.drawsSinceSeen());
        int[][] t = h.transitionCounts();
        int[][] rt = refTransition(ds);
        for (int i = 0; i < 22; i++) {
            assertArrayEquals(what + " transition row " + i, rt[i], t[i]);
        }
        assertArrayEquals(what + " contains", refContains(ds), h.containsCounts());
        assertEquals(what + " repeat", refRepeat(ds), h.repeatRate(), 1e-12);
        int[][] p = h.pairCounts();
        int[][] rp = refPairs(ds);
        for (int i = 0; i < 22; i++) {
            for (int j = 0; j < 22; j++) {
                if (i != j) {
                    assertEquals(what + " pairs " + i + "," + j, rp[i][j], p[i][j]);
                }
            }
        }
    }

    // ================================================================== Strategies.defaults

    @Test
    public void defaultsAreTheNineExpertsInOrder() {
        List<Strategy> list = Strategies.defaults(new ModelParams());
        assertEquals(9, list.size());
        Set<String> seen = new HashSet<String>();
        for (int i = 0; i < IDS.length; i++) {
            Strategy s = list.get(i);
            assertEquals(IDS[i], s.id());
            assertNotNull(s.name());
            assertFalse(s.name().trim().isEmpty());
            assertNotNull(s.description());
            assertFalse(s.description().trim().isEmpty());
            assertTrue(seen.add(s.id()));
        }
    }

    @Test
    public void defaultsWithNullParams() {
        List<Strategy> list = Strategies.defaults(null);
        assertEquals(9, list.size());
        assertEquals("uniform", list.get(0).id());
        assertEquals("pairs", list.get(8).id());
    }

    // ================================================================== generic properties

    @Test
    public void everyStrategyIsValidOnEmptyHistory() {
        History h = history(new ArrayList<Draw>());
        for (Strategy s : Strategies.defaults(new ModelParams())) {
            assertValidScores(s.id() + " empty", s.scores(h));
        }
    }

    @Test
    public void everyStrategyIsValidOnOneDraw() {
        List<Draw> one = new ArrayList<Draw>();
        one.add(new Draw("2026-10-16", range(1, 11)));
        History h = history(one);
        for (Strategy s : Strategies.defaults(new ModelParams())) {
            assertValidScores(s.id() + " one", s.scores(h));
        }
    }

    @Test
    public void everyStrategyIsValidOnManyDraws() {
        History h = history(randomDraws(500, 42));
        for (Strategy s : Strategies.defaults(new ModelParams())) {
            assertValidScores(s.id() + " many", s.scores(h));
        }
    }

    @Test
    public void everyStrategyIsValidOnDegenerateHistories() {
        // identical draws (repeat rate 1) and alternating complementary draws (repeat rate 0)
        List<Draw> same = new ArrayList<Draw>();
        List<Draw> alternating = new ArrayList<Draw>();
        String day = "2026-01-01";
        for (int i = 0; i < 40; i++) {
            same.add(new Draw(day, range(1, 11)));
            alternating.add(new Draw(day, i % 2 == 0 ? range(1, 11) : range(12, 22)));
            day = DateUtil.addDays(day, 1);
        }
        for (Strategy s : Strategies.defaults(new ModelParams())) {
            assertValidScores(s.id() + " same", s.scores(history(same)));
            assertValidScores(s.id() + " alternating", s.scores(history(alternating)));
        }
    }

    @Test
    public void everyStrategyIsValidWithExtremeParams() {
        ModelParams lo = new ModelParams();
        lo.shortWindow = 2;
        lo.longWindow = 2;
        lo.halfLife = 1;
        ModelParams hi = new ModelParams();
        hi.shortWindow = 200;
        hi.longWindow = 1000;
        hi.halfLife = 500;
        History h = history(randomDraws(300, 9));
        for (Strategy s : Strategies.defaults(lo)) {
            assertValidScores(s.id() + " lo", s.scores(h));
        }
        for (Strategy s : Strategies.defaults(hi)) {
            assertValidScores(s.id() + " hi", s.scores(h));
        }
    }

    @Test
    public void strategiesAreDeterministicAndDoNotMutateTheHistory() {
        List<Draw> draws = randomDraws(120, 3);
        History h = history(draws);
        int[] countsBefore = h.counts();
        for (Strategy s : Strategies.defaults(new ModelParams())) {
            double[] a = s.scores(h);
            double[] b = s.scores(h);
            double[] c = s.scores(history(draws));
            assertArrayEquals(s.id(), a, b, 0.0);
            assertArrayEquals(s.id(), a, c, 0.0);
            a[0] = -99;                                  // the result is the caller's copy
            assertArrayEquals(s.id(), b, s.scores(h), 0.0);
        }
        assertArrayEquals(countsBefore, h.counts());
    }

    // ================================================================== each expert's formula

    @Test
    public void uniformIsAllOnes() {
        assertAll("empty", 1.0, strategy("uniform").scores(history(new ArrayList<Draw>())));
        assertAll("many", 1.0, strategy("uniform").scores(history(randomDraws(50, 1))));
    }

    @Test
    public void freqIsLaplaceSmoothedCount() {
        List<Draw> ds = randomDraws(77, 5);
        double[] s = strategy("freq").scores(history(ds));
        int[] c = refCounts(ds, Integer.MAX_VALUE);
        for (int i = 0; i < 22; i++) {
            assertEquals(c[i] + 1.0, s[i], EPS);
        }
        assertAll("empty", 1.0, strategy("freq").scores(history(new ArrayList<Draw>())));
    }

    @Test
    public void recentUsesTheConfiguredWindows() {
        List<Draw> ds = randomDraws(77, 6);
        ModelParams p = new ModelParams();
        p.shortWindow = 7;
        p.longWindow = 25;
        List<Strategy> list = Strategies.defaults(p);
        double[] r10 = list.get(2).scores(history(ds));
        double[] r30 = list.get(3).scores(history(ds));
        int[] c7 = refCounts(ds, 7);
        int[] c25 = refCounts(ds, 25);
        for (int i = 0; i < 22; i++) {
            assertEquals(c7[i] + 0.5, r10[i], EPS);
            assertEquals(c25[i] + 0.5, r30[i], EPS);
        }
    }

    @Test
    public void recentWithDefaultWindowsOnHandBuiltHistory() {
        double[] s = strategy("recent10").scores(history(handBuilt()));
        // window 10 >= 3 draws: all-time counts + 0.5
        assertEquals(1.5, s[0], EPS);      // 1: once
        assertEquals(2.5, s[1], EPS);      // 2: twice
        assertEquals(3.5, s[2], EPS);      // 3..11: three times
        assertEquals(3.5, s[10], EPS);
        assertEquals(2.5, s[11], EPS);     // 12: twice
        assertEquals(1.5, s[12], EPS);     // 13: once
        assertEquals(0.5, s[13], EPS);     // 14..22: never
        assertEquals(0.5, s[21], EPS);
    }

    @Test
    public void decayIsHalfLifeWeightedCountPlusHalf() {
        List<Draw> ds = handBuilt();
        ModelParams p = new ModelParams();
        p.halfLife = 1;
        double[] s = Strategies.defaults(p).get(4).scores(history(ds));
        // ages: D2 = 0 (weight 1), D1 = 1 (0.5), D0 = 2 (0.25)
        assertEquals(0.25 + 0.5, s[0], EPS);                 // 1 in D0
        assertEquals(0.25 + 0.5 + 0.5, s[1], EPS);           // 2 in D0, D1
        assertEquals(0.25 + 0.5 + 1.0 + 0.5, s[2], EPS);     // 3 in all
        assertEquals(0.5 + 1.0 + 0.5, s[11], EPS);           // 12 in D1, D2
        assertEquals(1.0 + 0.5, s[12], EPS);                 // 13 in D2
        assertEquals(0.5, s[13], EPS);                       // 14 never
        List<Draw> rnd = randomDraws(200, 8);
        double[] d15 = strategy("decay").scores(history(rnd));
        double[] ref = refDecay(rnd, 15);
        for (int i = 0; i < 22; i++) {
            assertEquals(ref[i] + 0.5, d15[i], 1e-9);
        }
    }

    @Test
    public void gapOnHandBuiltHistory() {
        double[] s = strategy("gap").scores(history(handBuilt()));
        assertEquals("1 last seen 2 draws ago", 3.0, s[0], EPS);
        assertEquals("2 last seen 1 draw ago", 2.0, s[1], EPS);
        for (int n = 3; n <= 13; n++) {
            assertEquals(n + " in the last draw", 1.0, s[n - 1], EPS);
        }
        for (int n = 14; n <= 22; n++) {
            assertEquals(n + " never seen -> size()+1", 4.0, s[n - 1], EPS);
        }
    }

    @Test
    public void gapOnEmptyHistoryIsUniform() {
        assertAll("gap empty", 1.0, strategy("gap").scores(history(new ArrayList<Draw>())));
    }

    @Test
    public void gapMatchesReferenceOnRandomData() {
        List<Draw> ds = randomDraws(150, 13);
        double[] s = strategy("gap").scores(history(ds));
        int[] since = refSince(ds);
        for (int i = 0; i < 22; i++) {
            assertEquals(since[i] + 1.0, s[i], EPS);
        }
    }

    @Test
    public void transitionOnHandBuiltHistory() {
        double[] s = strategy("transition").scores(history(handBuilt()));
        // last draw 3..13; C[3..11] = 2, C[12] = 1, C[13] = 0
        // j not in D1 and not in D2 (1, 14..22): (9 * 1/4 + 1/3 + 1/2) / 11
        double none = (9 * 0.25 + 1.0 / 3 + 0.5) / 11;
        // j in D1 only (2): (9 * 2/4 + 1/3 + 1/2) / 11
        double d1only = (9 * 0.5 + 1.0 / 3 + 0.5) / 11;
        // j in D1 and D2 (3..12): (9 * 3/4 + 2/3 + 1/2) / 11
        double both = (9 * 0.75 + 2.0 / 3 + 0.5) / 11;
        // j in D2 only (13): (9 * 2/4 + 2/3 + 1/2) / 11
        double d2only = (9 * 0.5 + 2.0 / 3 + 0.5) / 11;
        assertEquals(none, s[0], EPS);
        assertEquals(d1only, s[1], EPS);
        for (int n = 3; n <= 12; n++) {
            assertEquals("number " + n, both, s[n - 1], EPS);
        }
        assertEquals(d2only, s[12], EPS);
        for (int n = 14; n <= 22; n++) {
            assertEquals("number " + n, none, s[n - 1], EPS);
        }
        assertEquals(0.280303030303, s[0], 1e-9);
        assertEquals(0.719696969697, s[4], 1e-9);
    }

    @Test
    public void transitionWithoutLastDrawIsUniform() {
        assertAll("transition empty", 1.0, strategy("transition").scores(history(new ArrayList<Draw>())));
    }

    @Test
    public void transitionWithOneDrawIsFlat() {
        List<Draw> one = new ArrayList<Draw>();
        one.add(new Draw("2026-10-16", range(1, 11)));
        // no transitions yet: (0 + 1) / (0 + 2) for every pair
        assertAll("transition one", 0.5, strategy("transition").scores(history(one)));
    }

    @Test
    public void transitionMatchesReferenceOnRandomData() {
        for (long seed = 1; seed <= 5; seed++) {
            List<Draw> ds = randomDraws(60 + (int) seed * 17, seed);
            assertArrayEquals("seed " + seed, refTransitionScores(ds),
                    strategy("transition").scores(history(ds)), 1e-12);
        }
    }

    @Test
    public void transitionLearnsACycle() {
        // A -> B -> C -> A ... with overlapping sets; after A the next draw is always B
        int[] a = range(1, 11);
        int[] b = range(6, 16);
        int[] c = range(12, 22);
        int[][] cycle = {a, b, c};
        List<Draw> ds = new ArrayList<Draw>();
        String day = "2026-01-01";
        for (int t = 0; t < 91; t++) {                   // 91 = 3 * 30 + 1: ends on A
            ds.add(new Draw(day, cycle[t % 3]));
            day = DateUtil.addDays(day, 1);
        }
        assertArrayEquals(a, ds.get(ds.size() - 1).numbers);
        double[] s = strategy("transition").scores(history(ds));
        assertArrayEquals(b, topEleven(s));
        // and the reference formula agrees
        assertArrayEquals(refTransitionScores(ds), s, 1e-12);
    }

    @Test
    public void repeatOnHandBuiltHistory() {
        double[] s = strategy("repeat").scores(history(handBuilt()));
        double r = 10.0 / 11.0;              // |D0 & D1| = |D1 & D2| = 10
        for (int n = 1; n <= 22; n++) {
            double expected = n >= 3 && n <= 13 ? r : 1 - r;
            assertEquals("number " + n, expected, s[n - 1], EPS);
        }
    }

    @Test
    public void repeatEdgeCases() {
        assertAll("repeat empty", 1.0, strategy("repeat").scores(history(new ArrayList<Draw>())));
        List<Draw> one = new ArrayList<Draw>();
        one.add(new Draw("2026-10-16", range(1, 11)));
        // repeatRate() == 0.5 with fewer than 2 draws
        assertAll("repeat one", 0.5, strategy("repeat").scores(history(one)));
        List<Draw> alt = new ArrayList<Draw>();
        alt.add(new Draw("2026-10-15", range(12, 22)));
        alt.add(new Draw("2026-10-16", range(1, 11)));
        double[] s = strategy("repeat").scores(history(alt));
        for (int n = 1; n <= 11; n++) {
            assertEquals(0.0, s[n - 1], EPS);
        }
        for (int n = 12; n <= 22; n++) {
            assertEquals(1.0, s[n - 1], EPS);
        }
    }

    @Test
    public void pairsFavoursCompanionsOfTheLastDraw() {
        // 1..11 always appear together and 12..22 always together; the last draw is 1..11
        List<Draw> ds = new ArrayList<Draw>();
        String day = "2026-01-01";
        for (int i = 0; i < 21; i++) {
            ds.add(new Draw(day, i % 2 == 0 ? range(1, 11) : range(12, 22)));
            day = DateUtil.addDays(day, 1);
        }
        double[] s = strategy("pairs").scores(history(ds));
        for (int a = 1; a <= 11; a++) {
            for (int b = 12; b <= 22; b++) {
                assertTrue(a + " vs " + b, s[a - 1] > s[b - 1]);
            }
        }
        assertAll("pairs empty", 1.0, strategy("pairs").scores(history(new ArrayList<Draw>())));
    }

    // ================================================================== History

    @Test
    public void historyOnHandBuiltDraws() {
        History h = history(handBuilt());
        assertEquals(3, h.size());
        assertEquals("2026-10-16", h.last().date);
        int[] c = h.counts();
        assertEquals(1, c[0]);
        assertEquals(2, c[1]);
        assertEquals(3, c[2]);
        assertEquals(2, c[11]);
        assertEquals(1, c[12]);
        assertEquals(0, c[13]);
        int[] l2 = h.countsInLast(2);
        assertEquals(0, l2[0]);
        assertEquals(1, l2[1]);
        assertEquals(2, l2[2]);
        assertEquals(2, l2[11]);
        assertEquals(1, l2[12]);
        int[] since = h.drawsSinceSeen();
        assertEquals(2, since[0]);
        assertEquals(1, since[1]);
        assertEquals(0, since[2]);
        assertEquals(3, since[13]);
        int[][] t = h.transitionCounts();
        assertEquals("1 -> 12 (D0 -> D1)", 1, t[0][11]);
        assertEquals("1 -> 13", 0, t[0][12]);
        assertEquals("2 -> 13 (D1 -> D2)", 1, t[1][12]);
        assertEquals("2 -> 5 (twice)", 2, t[1][4]);
        assertEquals("13 has no successor", 0, t[12][0]);
        int[] cc = h.containsCounts();
        assertEquals(1, cc[0]);
        assertEquals(2, cc[1]);
        assertEquals(2, cc[10]);
        assertEquals(1, cc[11]);
        assertEquals(0, cc[12]);
        assertEquals(10.0 / 11.0, h.repeatRate(), EPS);
        int[][] p = h.pairCounts();
        assertEquals(1, p[0][1]);
        assertEquals(2, p[1][2]);
        assertEquals(0, p[0][12]);
        assertEquals(3, p[2][3]);
    }

    @Test
    public void historyEmpty() {
        History h = history(new ArrayList<Draw>());
        assertEquals(0, h.size());
        assertNull(h.last());
        assertArrayEquals(new int[22], h.counts());
        assertArrayEquals(new int[22], h.countsInLast(10));
        assertArrayEquals(new double[22], h.decayedCounts(15), 0.0);
        int[] since = h.drawsSinceSeen();
        assertArrayEquals(new int[22], since);              // size() == 0 for never seen
        assertEquals(0.5, h.repeatRate(), 0.0);
        assertArrayEquals(new int[22], h.containsCounts());
        History nullHistory = new History(null);
        assertEquals(0, nullHistory.size());
    }

    @Test
    public void historyRepeatRateWithOneDraw() {
        List<Draw> one = new ArrayList<Draw>();
        one.add(new Draw("2026-10-16", range(1, 11)));
        assertEquals(0.5, history(one).repeatRate(), 0.0);
    }

    @Test
    public void historyMatchesReferenceOnRandomData() {
        for (long seed = 1; seed <= 6; seed++) {
            List<Draw> ds = randomDraws((int) (seed * 37), seed);
            assertHistoryMatchesReference("seed " + seed, history(ds), ds);
        }
    }

    @Test
    public void pairCountsAreSymmetric() {
        int[][] p = history(randomDraws(90, 21)).pairCounts();
        for (int i = 0; i < 22; i++) {
            for (int j = 0; j < 22; j++) {
                assertEquals(p[i][j], p[j][i]);
            }
        }
    }

    @Test
    public void historyIsAnImmutableSnapshot() {
        List<Draw> ds = randomDraws(20, 4);
        History h = history(ds);
        int[] before = h.counts();
        ds.add(new Draw("2030-01-01", range(1, 11)));
        assertEquals(20, h.size());
        assertArrayEquals(before, h.counts());
        int[] copy = h.counts();
        copy[0] = 999;
        assertArrayEquals(before, h.counts());
        int[][] t = h.transitionCounts();
        t[0][0] = 999;
        assertTrue(h.transitionCounts()[0][0] != 999);
    }

    @Test
    public void historyGetRejectsBadIndex() {
        History h = history(handBuilt());
        try {
            h.get(3);
            throw new AssertionError("expected IndexOutOfBoundsException");
        } catch (IndexOutOfBoundsException expected) {
            // ok
        }
        try {
            h.get(-1);
            throw new AssertionError("expected IndexOutOfBoundsException");
        } catch (IndexOutOfBoundsException expected) {
            // ok
        }
    }

    @Test
    public void incrementalRunningStatsMatchFreshHistories() {
        List<Draw> ds = randomDraws(160, 31);
        RunningStats rs = new RunningStats();
        List<History> views = new ArrayList<History>();
        for (int t = 0; t < ds.size(); t++) {
            History v = rs.view();
            List<Draw> prefix = ds.subList(0, t);
            assertHistoryMatchesReference("step " + t, v, prefix);
            for (Strategy s : Strategies.defaults(new ModelParams())) {
                assertArrayEquals(s.id() + " step " + t, s.scores(history(prefix)), s.scores(v), 1e-9);
            }
            if (t % 40 == 0) {
                views.add(v);
            }
            rs.push(ds.get(t));
        }
        // views taken earlier are still correct after more draws were pushed
        for (int k = 0; k < views.size(); k++) {
            History v = rs.view();
            History old = views.get(k);
            assertHistoryMatchesReference("old view " + k, old, ds.subList(0, old.size()));
            assertEquals(ds.size(), v.size());
        }
    }

    // ------------------------------------------------------------------ helpers

    private static int[] topEleven(double[] s) {
        Integer[] idx = new Integer[22];
        for (int i = 0; i < 22; i++) {
            idx[i] = Integer.valueOf(i);
        }
        final double[] sc = s;
        Arrays.sort(idx, new java.util.Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                int c = Double.compare(sc[b.intValue()], sc[a.intValue()]);
                return c != 0 ? c : a.compareTo(b);
            }
        });
        int[] out = new int[11];
        for (int i = 0; i < 11; i++) {
            out[i] = idx[i].intValue() + 1;
        }
        Arrays.sort(out);
        return out;
    }
}
