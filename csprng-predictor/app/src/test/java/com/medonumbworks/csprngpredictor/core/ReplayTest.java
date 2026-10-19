package com.medonumbworks.csprngpredictor.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.junit.Test;

/**
 * Walk-forward replay (SPEC 1.10) and Stats: record bookkeeping, next prediction, stats
 * arithmetic, the hypergeometric baseline, random-data sanity and planted patterns.
 */
public class ReplayTest {

    private static int[] range(int from, int to) {
        int[] r = new int[to - from + 1];
        for (int i = 0; i < r.length; i++) {
            r[i] = from + i;
        }
        return r;
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

    /** Draw t equals draw t-1 with every number rotated by one (n -> n % 22 + 1). */
    private static List<Draw> rotatingDraws(int n, long seed) {
        int[] cur = randomDraw(new Random(seed));
        List<Draw> list = new ArrayList<Draw>();
        String day = "2021-03-01";
        for (int i = 0; i < n; i++) {
            list.add(new Draw(day, cur));
            int[] next = new int[11];
            for (int k = 0; k < 11; k++) {
                next[k] = cur[k] % 22 + 1;
            }
            cur = next;
            day = DateUtil.addDays(day, 1);
        }
        return list;
    }

    private static List<Integer> hitsOf(List<PredictionRecord> records) {
        List<Integer> out = new ArrayList<Integer>();
        for (PredictionRecord r : records) {
            out.add(Integer.valueOf(r.hits));
        }
        return out;
    }

    // ================================================================== records

    @Test
    public void recordCountIsNMinusOne() {
        int[] sizes = {0, 1, 2, 3, 10, 57};
        for (int n : sizes) {
            ReplayResult r = Replay.run(randomDraws(n, n), new ModelParams());
            assertEquals("n = " + n, Math.max(0, n - 1), r.records.size());
            assertEquals(Math.max(0, n - 1), r.stats.n);
            assertEquals(Math.max(0, n - 1), r.model.observed());
        }
    }

    @Test
    public void recordsAreWalkForward() {
        List<Draw> ds = randomDraws(40, 3);
        ReplayResult r = Replay.run(ds, new ModelParams());
        for (int t = 1; t < ds.size(); t++) {
            PredictionRecord rec = r.records.get(t - 1);
            assertEquals(ds.get(t), rec.actual);
            assertEquals(t, rec.prediction.basedOnCount);
            assertEquals(ds.get(t - 1).date, rec.prediction.basedOnDate);
            assertEquals(DateUtil.addDays(ds.get(t - 1).date, 1), rec.prediction.targetDate);
            assertEquals(rec.actual.hitsAgainst(rec.prediction.numbers), rec.hits);
            assertEquals(9, rec.expertHits.length);
            for (int k = 0; k < 9; k++) {
                assertEquals(rec.actual.hitsAgainst(rec.prediction.expertPicks[k]), rec.expertHits[k]);
            }
            assertTrue(Game.isValidDraw(rec.prediction.numbers));
        }
    }

    @Test
    public void walkForwardMatchesAManualLoop() {
        List<Draw> ds = randomDraws(60, 4);
        ModelParams params = new ModelParams();
        ReplayResult r = Replay.run(ds, params);
        Model m = new Model(params, Strategies.defaults(params));
        for (int t = 1; t < ds.size(); t++) {
            Prediction p = m.predict(new History(ds.subList(0, t)));
            assertArrayEquals("step " + t, p.numbers, r.records.get(t - 1).prediction.numbers);
            assertArrayEquals("step " + t, p.scores, r.records.get(t - 1).prediction.scores, 1e-12);
            m.observe(p, ds.get(t));
        }
        assertArrayEquals(m.weights(), r.model.weights(), 1e-12);
        assertArrayEquals(m.biases(), r.model.biases(), 1e-12);
        assertArrayEquals(m.predict(new History(ds)).numbers, r.next.numbers);
    }

    @Test
    public void expertTotalsMatchRecords() {
        List<Draw> ds = randomDraws(50, 8);
        ReplayResult r = Replay.run(ds, new ModelParams());
        int[] totals = new int[9];
        for (PredictionRecord rec : r.records) {
            for (int k = 0; k < 9; k++) {
                totals[k] += rec.expertHits[k];
            }
        }
        assertArrayEquals(totals, r.model.expertHitsTotal());
        assertEquals(1.0, sum(r.model.weights()), 1e-12);
        assertEquals(1.0, sum(r.next.expertWeights), 1e-12);
    }

    // ================================================================== next prediction

    @Test
    public void nextTargetsTheDayAfterTheLatestDraw() {
        List<Draw> ds = randomDraws(25, 1);
        ReplayResult r = Replay.run(ds, new ModelParams());
        String latest = ds.get(ds.size() - 1).date;
        assertEquals(DateUtil.addDays(latest, 1), r.next.targetDate);
        assertEquals(latest, r.next.basedOnDate);
        assertEquals(25, r.next.basedOnCount);
        assertTrue(Game.isValidDraw(r.next.numbers));
        assertEquals(11.0, sum(r.next.scores), 1e-9);
    }

    @Test
    public void nextAcrossMonthAndYearEnd() {
        List<Draw> ds = new ArrayList<Draw>();
        ds.add(new Draw("2026-12-30", range(1, 11)));
        ds.add(new Draw("2026-12-31", range(12, 22)));
        assertEquals("2027-01-01", Replay.run(ds, new ModelParams()).next.targetDate);
    }

    @Test
    public void nextWithoutDrawsTargetsToday() {
        ReplayResult r = Replay.run(new ArrayList<Draw>(), new ModelParams());
        assertEquals(DateUtil.today(), r.next.targetDate);
        assertEquals("", r.next.basedOnDate);
        assertEquals(0, r.next.basedOnCount);
        assertTrue(r.records.isEmpty());
        assertEquals(0, r.stats.n);
        assertTrue(Game.isValidDraw(r.next.numbers));
        ReplayResult nul = Replay.run(null, null);
        assertTrue(nul.records.isEmpty());
        assertNotNull(nul.next);
    }

    @Test
    public void singleDrawHasNoRecordsButAPrediction() {
        List<Draw> ds = new ArrayList<Draw>();
        ds.add(new Draw("2026-10-16", range(1, 11)));
        ReplayResult r = Replay.run(ds, new ModelParams());
        assertTrue(r.records.isEmpty());
        assertEquals("2026-10-17", r.next.targetDate);
        assertEquals(1, r.next.basedOnCount);
        assertEquals(0, r.model.observed());
        assertEquals(0, r.stats.n);
        assertEquals(-1, r.stats.best);
    }

    @Test
    public void nextEqualsAFreshPredictionFromTheFinalModel() {
        List<Draw> ds = randomDraws(200, 42);
        ReplayResult r = Replay.run(ds, new ModelParams());
        Prediction fresh = r.model.predict(new History(ds));
        assertArrayEquals(fresh.numbers, r.next.numbers);
        assertArrayEquals(fresh.scores, r.next.scores, 1e-12);
    }

    @Test
    public void replayIsDeterministic() {
        List<Draw> ds = randomDraws(150, 7);
        ReplayResult a = Replay.run(ds, new ModelParams());
        ReplayResult b = Replay.run(new ArrayList<Draw>(ds), new ModelParams());
        assertArrayEquals(a.next.numbers, b.next.numbers);
        assertArrayEquals(a.next.scores, b.next.scores, 0.0);
        assertEquals(hitsOf(a.records), hitsOf(b.records));
        assertArrayEquals(a.model.weights(), b.model.weights(), 0.0);
    }

    @Test
    public void unsortedInputIsReplayedChronologically() {
        List<Draw> ds = randomDraws(60, 12);
        List<Draw> shuffled = new ArrayList<Draw>(ds);
        Collections.shuffle(shuffled, new Random(1));
        ReplayResult a = Replay.run(ds, new ModelParams());
        ReplayResult b = Replay.run(shuffled, new ModelParams());
        assertArrayEquals(a.next.numbers, b.next.numbers);
        assertEquals(hitsOf(a.records), hitsOf(b.records));
    }

    @Test
    public void paramsAreKeptClamped() {
        ModelParams p = new ModelParams();
        p.eta = 0.3;
        p.halfLife = 9999;
        ReplayResult r = Replay.run(randomDraws(20, 2), p);
        assertEquals(0.3, r.params.eta, 0.0);
        assertEquals(500.0, r.params.halfLife, 0.0);
        assertEquals(9999.0, p.halfLife, 0.0);       // the caller's object is not modified
    }

    // ================================================================== evaluateLastN

    @Test
    public void statsOverTheLastNRecords() {
        List<Draw> ds = randomDraws(80, 21);
        ReplayResult all = Replay.run(ds, new ModelParams());
        ReplayResult last = Replay.run(ds, new ModelParams(), 30);
        assertEquals(79, last.records.size());
        assertEquals(30, last.stats.n);
        Stats expected = Stats.of(all.records.subList(49, 79));
        assertEquals(expected.meanHits, last.stats.meanHits, 1e-12);
        assertArrayEquals(expected.histogram, last.stats.histogram);
        assertArrayEquals("the model still learns from everything", all.next.numbers, last.next.numbers);
        assertEquals(79, Replay.run(ds, new ModelParams(), 0).stats.n);
        assertEquals(79, Replay.run(ds, new ModelParams(), 500).stats.n);
        assertEquals(79, Replay.run(ds, new ModelParams(), -1).stats.n);
        assertEquals(1, Replay.run(ds, new ModelParams(), 1).stats.n);
    }

    // ================================================================== Stats arithmetic

    @Test
    public void statsOfReplayRecords() {
        ReplayResult r = Replay.run(randomDraws(120, 33), new ModelParams());
        Stats s = r.stats;
        assertEquals(119, s.n);
        int total = 0;
        int histSum = 0;
        for (int k = 0; k <= 11; k++) {
            histSum += s.histogram[k];
            total += k * s.histogram[k];
        }
        assertEquals(12, s.histogram.length);
        assertEquals("histogram sums to n", s.n, histSum);
        double mean = total / (double) s.n;
        assertEquals(mean, s.meanHits, 1e-12);
        assertEquals(mean - 5.5, s.edge, 1e-12);
        assertEquals((mean - 5.5) / (1.2 / Math.sqrt(s.n)), s.z, 1e-9);
        double ss = 0;
        int best = -1;
        int worst = 99;
        for (PredictionRecord rec : r.records) {
            ss += (rec.hits - mean) * (rec.hits - mean);
            best = Math.max(best, rec.hits);
            worst = Math.min(worst, rec.hits);
        }
        assertEquals(Math.sqrt(ss / (s.n - 1)), s.sd, 1e-12);
        assertEquals(best, s.best);
        assertEquals(worst, s.worst);
        double lastTen = 0;
        for (int i = r.records.size() - 10; i < r.records.size(); i++) {
            lastTen += r.records.get(i).hits;
        }
        assertEquals(lastTen / 10, s.lastTenMean, 1e-12);
        assertNotNull(s.summaryText());
        assertFalse(s.summaryText().trim().isEmpty());
    }

    @Test
    public void statsOfHitLists() {
        Stats s = Stats.ofHits(new int[] {5, 6, 7});
        assertEquals(3, s.n);
        assertEquals(6.0, s.meanHits, 1e-12);
        assertEquals(1.0, s.sd, 1e-12);
        assertEquals(0.5, s.edge, 1e-12);
        assertEquals(0.5 / (1.2 / Math.sqrt(3)), s.z, 1e-12);
        assertEquals(7, s.best);
        assertEquals(5, s.worst);
        assertEquals(6.0, s.lastTenMean, 1e-12);
        assertEquals(1, s.histogram[5]);
        assertEquals(1, s.histogram[6]);
        assertEquals(1, s.histogram[7]);
        Stats l = Stats.ofHits(Arrays.asList(Integer.valueOf(5), Integer.valueOf(6), Integer.valueOf(7)));
        assertEquals(s.meanHits, l.meanHits, 0.0);
        assertEquals(s.sd, l.sd, 0.0);
        assertArrayEquals(s.histogram, l.histogram);
    }

    @Test
    public void lastTenMeanUsesOnlyTheLastTen() {
        int[] hits = {0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11};
        Stats s = Stats.ofHits(hits);
        assertEquals(12, s.n);
        assertEquals(11.0, s.lastTenMean, 1e-12);
        assertEquals(110 / 12.0, s.meanHits, 1e-12);
        assertEquals(0, s.worst);
        assertEquals(11, s.best);
    }

    @Test
    public void statsOfOneValue() {
        Stats s = Stats.ofHits(new int[] {8});
        assertEquals(1, s.n);
        assertEquals(8.0, s.meanHits, 0.0);
        assertEquals("sample SD needs two values", 0.0, s.sd, 0.0);
        assertEquals(8.0, s.lastTenMean, 0.0);
        assertEquals((8 - 5.5) / 1.2, s.z, 1e-12);
    }

    @Test
    public void statsOfNothing() {
        Stats[] empties = {Stats.of(new ArrayList<PredictionRecord>()), Stats.of(null),
                Stats.ofHits(new int[0]), Stats.ofHits(new ArrayList<Integer>())};
        for (Stats s : empties) {
            assertEquals(0, s.n);
            assertEquals(0.0, s.meanHits, 0.0);
            assertEquals(0.0, s.sd, 0.0);
            assertEquals(0.0, s.z, 0.0);
            assertEquals(-1, s.best);
            assertEquals(-1, s.worst);
            assertEquals(12, s.histogram.length);
            assertArrayEquals(new int[12], s.histogram);
            assertNotNull(s.summaryText());
        }
    }

    // ================================================================== hypergeometric baseline

    @Test
    public void hypergeometricPmfSumsToOne() {
        double total = 0;
        for (int k = 0; k <= 11; k++) {
            double p = Stats.hypergeometricPmf(k);
            assertTrue(p > 0);
            total += p;
        }
        assertEquals(1.0, total, 1e-12);
    }

    @Test
    public void hypergeometricPmfValues() {
        // P(k) = C(11,k) * C(11,11-k) / C(22,11)
        assertEquals(1.0 / 705432, Stats.hypergeometricPmf(0), 1e-18);
        assertEquals(1.0 / 705432, Stats.hypergeometricPmf(11), 1e-18);
        assertEquals(462.0 * 462 / 705432, Stats.hypergeometricPmf(5), 1e-15);
        assertEquals(462.0 * 462 / 705432, Stats.hypergeometricPmf(6), 1e-15);
        for (int k = 0; k <= 11; k++) {
            assertEquals("symmetric", Stats.hypergeometricPmf(k), Stats.hypergeometricPmf(11 - k), 1e-15);
        }
        assertEquals(0.0, Stats.hypergeometricPmf(-1), 0.0);
        assertEquals(0.0, Stats.hypergeometricPmf(12), 0.0);
    }

    @Test
    public void hypergeometricMomentsMatchTheConstants() {
        double mean = 0;
        double sq = 0;
        for (int k = 0; k <= 11; k++) {
            mean += k * Stats.hypergeometricPmf(k);
            sq += k * k * Stats.hypergeometricPmf(k);
        }
        assertEquals(5.5, mean, 1e-12);
        assertEquals(5.5, Stats.expectedHits(), 0.0);
        assertEquals(11 * 0.5 * 0.5 * 11 / 21.0, Stats.variance(), 1e-15);
        assertEquals(Stats.variance(), sq - mean * mean, 1e-10);
        assertEquals(1.2, Math.sqrt(Stats.variance()), 0.01);
    }

    @Test
    public void hypergeometricTail() {
        assertEquals(1.0, Stats.hypergeometricTail(0), 1e-15);
        assertEquals(1.0, Stats.hypergeometricTail(-3), 1e-15);
        assertEquals(0.0, Stats.hypergeometricTail(12), 0.0);
        assertEquals(1.0 / 705432, Stats.hypergeometricTail(11), 1e-18);
        for (int k = 0; k <= 11; k++) {
            double s = 0;
            for (int j = k; j <= 11; j++) {
                s += Stats.hypergeometricPmf(j);
            }
            assertEquals("tail(" + k + ")", s, Stats.hypergeometricTail(k), 1e-12);
            if (k > 0) {
                assertTrue(Stats.hypergeometricTail(k) < Stats.hypergeometricTail(k - 1));
            }
        }
        assertEquals(30372.0 / 705432, Stats.hypergeometricTail(8), 1e-12);   // about 4.3 %
        // symmetry: P(hits >= 6) = P(hits <= 5) = 1/2
        assertEquals(0.5, Stats.hypergeometricTail(6), 1e-12);
    }

    // ================================================================== sanity on data

    @Test
    public void randomDrawsScoreNearTheBaseline() {
        ReplayResult r = Replay.run(randomDraws(300, 2026), new ModelParams());
        assertEquals(299, r.stats.n);
        assertTrue("mean " + r.stats.meanHits, r.stats.meanHits > 4.0 && r.stats.meanHits < 7.0);
        assertTrue(r.stats.best <= 11 && r.stats.worst >= 0);
    }

    @Test
    public void severalRandomSeedsStayNearTheBaseline() {
        for (long seed = 100; seed < 105; seed++) {
            Stats s = Replay.run(randomDraws(300, seed), new ModelParams()).stats;
            assertTrue("seed " + seed + " mean " + s.meanHits, s.meanHits > 4.0 && s.meanHits < 7.0);
        }
    }

    @Test
    public void plantedRotationPatternIsLearned() {
        ReplayResult r = Replay.run(rotatingDraws(300, 5), new ModelParams());
        assertTrue("mean " + r.stats.meanHits, r.stats.meanHits > 8.0);
        assertTrue(r.stats.z > 3);
        // the last 50 predictions are better still
        Stats tail = Replay.run(rotatingDraws(300, 5), new ModelParams(), 50).stats;
        assertTrue("tail mean " + tail.meanHits, tail.meanHits > 8.0);
    }

    @Test
    public void plantedConstantDrawIsLearned() {
        List<Draw> ds = new ArrayList<Draw>();
        String day = "2025-01-01";
        int[] same = {2, 3, 5, 7, 11, 13, 14, 17, 19, 20, 22};
        for (int i = 0; i < 100; i++) {
            ds.add(new Draw(day, same));
            day = DateUtil.addDays(day, 1);
        }
        ReplayResult r = Replay.run(ds, new ModelParams());
        assertTrue("mean " + r.stats.meanHits, r.stats.meanHits > 10.0);
        assertArrayEquals(same, r.next.numbers);
    }

    @Test(timeout = 60000)
    public void longReplayIsFast() {
        ReplayResult r = Replay.run(randomDraws(5000, 1), new ModelParams());
        assertEquals(4999, r.records.size());
        assertTrue(r.stats.meanHits > 5.0 && r.stats.meanHits < 6.0);
    }

    @Test
    public void recordsAreUnmodifiable() {
        ReplayResult r = Replay.run(randomDraws(10, 1), new ModelParams());
        try {
            r.records.clear();
            throw new AssertionError("records must be unmodifiable");
        } catch (UnsupportedOperationException expected) {
            // ok
        }
    }

    private static double sum(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s;
    }
}
