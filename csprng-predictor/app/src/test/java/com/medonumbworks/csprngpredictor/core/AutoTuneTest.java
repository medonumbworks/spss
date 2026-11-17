package com.medonumbworks.csprngpredictor.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Test;

/**
 * AutoTune (SPEC 1.11): base below 15 draws, otherwise the best member of the
 * eta x halfLife x biasRate grid with the documented tie rules.
 */
public class AutoTuneTest {

    private static final double[] ETA = {0.1, 0.3, 1.0};
    private static final double[] HALF_LIFE = {5, 15, 40};
    private static final double[] BIAS_RATE = {0, 0.05, 0.2};

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
        int[] d = Arrays.copyOf(pool, 11);
        Arrays.sort(d);
        return d;
    }

    private static List<Draw> randomDraws(int n, long seed) {
        Random rnd = new Random(seed);
        List<Draw> list = new ArrayList<Draw>();
        String day = "2023-05-01";
        for (int i = 0; i < n; i++) {
            list.add(new Draw(day, randomDraw(rnd)));
            day = DateUtil.addDays(day, 1);
        }
        return list;
    }

    private static List<Draw> constantDraws(int n) {
        int[] same = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        List<Draw> list = new ArrayList<Draw>();
        String day = "2023-05-01";
        for (int i = 0; i < n; i++) {
            list.add(new Draw(day, same));
            day = DateUtil.addDays(day, 1);
        }
        return list;
    }

    private static boolean in(double[] grid, double v) {
        for (double g : grid) {
            if (g == v) {
                return true;
            }
        }
        return false;
    }

    private static ModelParams customBase() {
        ModelParams b = new ModelParams();
        b.share = 0.07;
        b.biasDecay = 0.2;
        b.shortWindow = 6;
        b.longWindow = 44;
        b.dayFirst = false;
        return b;
    }

    private static void assertOtherFieldsFromBase(ModelParams base, ModelParams p) {
        assertEquals(base.share, p.share, 0.0);
        assertEquals(base.biasDecay, p.biasDecay, 0.0);
        assertEquals(base.shortWindow, p.shortWindow);
        assertEquals(base.longWindow, p.longWindow);
        assertEquals(base.dayFirst, p.dayFirst);
    }

    // ================================================================== too few draws

    @Test
    public void fewerThan15DrawsReturnsBase() {
        ModelParams base = customBase();
        for (int n : new int[] {0, 1, 2, 10, 14}) {
            List<Draw> ds = randomDraws(n, n);
            AutoTune.Candidate c = AutoTune.tune(ds, base, 200);
            assertNotNull(c);
            assertEquals("n = " + n, base, c.params);
            assertNotNull(c.stats);
            Stats expected = Replay.run(ds, base, 200).stats;
            assertEquals(expected.n, c.stats.n);
            assertEquals(expected.meanHits, c.stats.meanHits, 1e-12);
        }
    }

    @Test
    public void baseIsNotModified() {
        ModelParams base = customBase();
        ModelParams before = base.copy();
        AutoTune.tune(randomDraws(10, 1), base, 50);
        AutoTune.tune(randomDraws(30, 1), base, 50);
        assertEquals(before, base);
    }

    @Test
    public void nullDrawsReturnBase() {
        AutoTune.Candidate c = AutoTune.tune(null, new ModelParams(), 100);
        assertEquals(new ModelParams(), c.params);
        assertEquals(0, c.stats.n);
    }

    // ================================================================== grid search

    @Test
    public void returnsAGridMemberWithStats() {
        ModelParams base = customBase();
        List<Draw> ds = randomDraws(40, 2026);
        AutoTune.Candidate c = AutoTune.tune(ds, base, 30);
        assertNotNull(c.params);
        assertNotNull(c.stats);
        assertTrue("eta " + c.params.eta, in(ETA, c.params.eta));
        assertTrue("halfLife " + c.params.halfLife, in(HALF_LIFE, c.params.halfLife));
        assertTrue("biasRate " + c.params.biasRate, in(BIAS_RATE, c.params.biasRate));
        assertOtherFieldsFromBase(base, c.params);
        assertEquals(30, c.stats.n);
        Stats again = Replay.run(ds, c.params, 30).stats;
        assertEquals(again.meanHits, c.stats.meanHits, 1e-12);
    }

    @Test
    public void exactly15DrawsAreTuned() {
        ModelParams base = new ModelParams();          // eta 0.5 and biasRate 0.1 are not grid values
        AutoTune.Candidate c = AutoTune.tune(randomDraws(15, 3), base, 200);
        assertTrue(in(ETA, c.params.eta));
        assertTrue(in(BIAS_RATE, c.params.biasRate));
        assertEquals(14, c.stats.n);
    }

    @Test
    public void winnerHasTheHighestMeanOfTheGrid() {
        ModelParams base = customBase();
        List<Draw> ds = randomDraws(45, 77);
        int lastN = 25;
        AutoTune.Candidate c = AutoTune.tune(ds, base, lastN);
        double best = -1;
        for (double eta : ETA) {
            for (double hl : HALF_LIFE) {
                for (double br : BIAS_RATE) {
                    ModelParams p = base.copy();
                    p.eta = eta;
                    p.halfLife = hl;
                    p.biasRate = br;
                    double mean = Replay.run(ds, p, lastN).stats.meanHits;
                    assertTrue(mean + " > winner " + c.stats.meanHits, mean <= c.stats.meanHits + 1e-12);
                    best = Math.max(best, mean);
                }
            }
        }
        assertEquals(best, c.stats.meanHits, 1e-12);
    }

    @Test
    public void tiesPreferFewerChangesFromBase() {
        // a constant draw is predicted perfectly by every grid member: all candidates tie
        List<Draw> ds = constantDraws(30);
        ModelParams base = new ModelParams();
        base.eta = 0.3;
        base.halfLife = 40;
        base.biasRate = 0.05;
        AutoTune.Candidate c = AutoTune.tune(ds, base, 200);
        assertEquals(11.0, c.stats.meanHits, 0.0);
        assertEquals("base itself is a grid member with zero changes", base, c.params);
    }

    @Test
    public void tiesThenPreferTheFirstInGridOrder() {
        // default base: eta 0.5 and biasRate 0.1 are off-grid (always changed), halfLife 15 is on it;
        // the fewest changes (2) keep halfLife 15, and the first such grid entry is eta 0.1, biasRate 0
        List<Draw> ds = constantDraws(30);
        AutoTune.Candidate c = AutoTune.tune(ds, new ModelParams(), 200);
        assertEquals(11.0, c.stats.meanHits, 0.0);
        assertEquals(0.1, c.params.eta, 0.0);
        assertEquals(15.0, c.params.halfLife, 0.0);
        assertEquals(0.0, c.params.biasRate, 0.0);
    }

    @Test
    public void tuneIsDeterministic() {
        List<Draw> ds = randomDraws(35, 5);
        AutoTune.Candidate a = AutoTune.tune(ds, new ModelParams(), 20);
        AutoTune.Candidate b = AutoTune.tune(ds, new ModelParams(), 20);
        assertEquals(a.params, b.params);
        assertEquals(a.stats.meanHits, b.stats.meanHits, 0.0);
    }

    @Test
    public void plantedPatternWinnerBeatsRandom() {
        List<Draw> ds = new ArrayList<Draw>();
        int[] cur = {1, 3, 4, 7, 9, 10, 13, 15, 16, 19, 21};
        String day = "2024-01-01";
        for (int i = 0; i < 60; i++) {
            ds.add(new Draw(day, cur));
            int[] next = new int[11];
            for (int k = 0; k < 11; k++) {
                next[k] = cur[k] % 22 + 1;
            }
            cur = next;
            day = DateUtil.addDays(day, 1);
        }
        AutoTune.Candidate c = AutoTune.tune(ds, new ModelParams(), 30);
        assertTrue("mean " + c.stats.meanHits, c.stats.meanHits > 7.0);
    }
}
