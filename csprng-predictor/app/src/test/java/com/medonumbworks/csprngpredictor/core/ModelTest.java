package com.medonumbworks.csprngpredictor.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Test;

/**
 * The learner (SPEC 1.9): prediction shape and determinism, Hedge weights with fixed share,
 * the per-number bias, and the Prediction / PredictionRecord helpers (SPEC 1.8).
 */
public class ModelTest {

    private static final int[] LOW = range(1, 11);
    private static final int[] HIGH = range(12, 22);

    private static int[] range(int from, int to) {
        int[] r = new int[to - from + 1];
        for (int i = 0; i < r.length; i++) {
            r[i] = from + i;
        }
        return r;
    }

    /** A fake expert that always returns the same scores. */
    private static final class Fixed implements Strategy {
        private final String id;
        private final double[] scores;

        Fixed(String id, double[] scores) {
            this.id = id;
            this.scores = scores.clone();
        }

        /** 1 for the given numbers, 0 for the others. */
        static Fixed favouring(String id, int[] numbers) {
            double[] s = new double[22];
            for (int n : numbers) {
                s[n - 1] = 1.0;
            }
            return new Fixed(id, s);
        }

        @Override
        public String id() {
            return id;
        }

        @Override
        public String name() {
            return "Fixed " + id;
        }

        @Override
        public String description() {
            return "Test expert";
        }

        @Override
        public double[] scores(History h) {
            return scores.clone();
        }
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
        String day = "2024-01-01";
        for (int i = 0; i < n; i++) {
            list.add(new Draw(day, randomDraw(rnd)));
            day = DateUtil.addDays(day, 1);
        }
        return list;
    }

    private static double sum(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s;
    }

    private static void assertValidPrediction(Prediction p, int experts) {
        assertNotNull(p);
        assertTrue(Arrays.toString(p.numbers), Game.isValidDraw(p.numbers));
        for (int i = 1; i < p.numbers.length; i++) {
            assertTrue("sorted", p.numbers[i - 1] < p.numbers[i]);
        }
        assertEquals(22, p.scores.length);
        for (double s : p.scores) {
            assertFalse(Double.isNaN(s));
            assertTrue(s >= 0);
        }
        assertEquals(11.0, sum(p.scores), 1e-9);
        assertEquals(experts, p.expertIds.length);
        assertEquals(experts, p.expertWeights.length);
        assertEquals(experts, p.expertPicks.length);
        assertEquals(1.0, sum(p.expertWeights), 1e-12);
        for (int[] picks : p.expertPicks) {
            assertTrue(Arrays.toString(picks), Game.isValidDraw(picks));
            for (int i = 1; i < picks.length; i++) {
                assertTrue(picks[i - 1] < picks[i]);
            }
        }
        // the picks are exactly the 11 best-ranked numbers
        for (int n = 1; n <= 22; n++) {
            boolean picked = p.contains(n);
            assertEquals("rank of " + n, picked, p.rankOf(n) <= 11);
        }
        for (int n : p.numbers) {
            for (int m = 1; m <= 22; m++) {
                if (!p.contains(m)) {
                    assertTrue(p.scoreOf(n) >= p.scoreOf(m));
                }
            }
        }
    }

    // ================================================================== predict

    @Test
    public void predictReturnsElevenDistinctSortedNumbers() {
        Model m = new Model(new ModelParams(), Strategies.defaults(new ModelParams()));
        for (int n : new int[] {0, 1, 2, 10, 100, 400}) {
            assertValidPrediction(m.predict(new History(randomDraws(n, n + 1))), 9);
        }
    }

    @Test
    public void predictIsDeterministic() {
        List<Draw> ds = randomDraws(150, 77);
        Model a = new Model(new ModelParams(), Strategies.defaults(new ModelParams()));
        Model b = new Model(new ModelParams(), Strategies.defaults(new ModelParams()));
        Prediction pa = a.predict(new History(ds));
        Prediction pb = b.predict(new History(ds));
        Prediction pa2 = a.predict(new History(ds));
        assertArrayEquals(pa.numbers, pb.numbers);
        assertArrayEquals(pa.scores, pb.scores, 0.0);
        assertArrayEquals(pa.numbers, pa2.numbers);
        assertArrayEquals(pa.scores, pa2.scores, 0.0);
        assertArrayEquals(pa.expertWeights, pb.expertWeights, 0.0);
        for (int k = 0; k < pa.expertPicks.length; k++) {
            assertArrayEquals(pa.expertPicks[k], pb.expertPicks[k]);
        }
    }

    @Test
    public void predictIsDeterministicAfterIdenticalLearning() {
        List<Draw> ds = randomDraws(80, 5);
        Model a = new Model(new ModelParams(), null);
        Model b = new Model(new ModelParams(), null);
        for (int t = 1; t < ds.size(); t++) {
            History h = new History(ds.subList(0, t));
            Prediction pa = a.predict(h);
            Prediction pb = b.predict(h);
            assertArrayEquals(pa.scores, pb.scores, 0.0);
            a.observe(pa, ds.get(t));
            b.observe(pb, ds.get(t));
        }
        assertArrayEquals(a.weights(), b.weights(), 0.0);
        assertArrayEquals(a.biases(), b.biases(), 0.0);
    }

    @Test
    public void predictDoesNotMutateTheModel() {
        Model m = new Model(new ModelParams(), null);
        double[] w = m.weights();
        double[] b = m.biases();
        m.predict(new History(randomDraws(50, 1)));
        m.predict(new History(randomDraws(50, 2)));
        assertArrayEquals(w, m.weights(), 0.0);
        assertArrayEquals(b, m.biases(), 0.0);
        assertEquals(0, m.observed());
    }

    @Test
    public void predictOnEmptyHistory() {
        Model m = new Model(new ModelParams(), null);
        Prediction p = m.predict(new History(new ArrayList<Draw>()));
        assertValidPrediction(p, 9);
        assertEquals("", p.basedOnDate);
        assertEquals(0, p.basedOnCount);
        assertEquals(DateUtil.today(), p.targetDate);
        // every expert is flat on an empty history: 0.5 each, ties -> lowest numbers
        for (int n = 1; n <= 22; n++) {
            assertEquals(0.5, p.scoreOf(n), 1e-9);
        }
        assertArrayEquals(LOW, p.numbers);
    }

    @Test
    public void predictionMetadata() {
        List<Draw> ds = randomDraws(30, 9);
        Model m = new Model(new ModelParams(), null);
        Prediction p = m.predict(new History(ds));
        Draw last = ds.get(ds.size() - 1);
        assertEquals(last.date, p.basedOnDate);
        assertEquals(30, p.basedOnCount);
        assertEquals(DateUtil.addDays(last.date, 1), p.targetDate);
        String[] ids = {"uniform", "freq", "recent10", "recent30", "decay", "gap", "transition", "repeat", "pairs"};
        assertArrayEquals(ids, p.expertIds);
        assertArrayEquals(ids, m.expertIds());
    }

    @Test
    public void initialWeightsAreUniform() {
        Model m = new Model(new ModelParams(), null);
        double[] w = m.weights();
        assertEquals(9, w.length);
        for (double x : w) {
            assertEquals(1.0 / 9, x, 1e-15);
        }
        assertArrayEquals(new double[22], m.biases(), 0.0);
        assertEquals(0, m.observed());
        assertArrayEquals(new int[9], m.expertHitsTotal());
        assertArrayEquals(new double[9], m.expertMeanHits(), 0.0);
    }

    @Test
    public void expertScoresAreNormalisedToElevenWithEpsilon() {
        double[] raw = new double[22];
        for (int i = 0; i < 22; i++) {
            raw[i] = i + 1;
        }
        List<Strategy> one = new ArrayList<Strategy>();
        one.add(new Fixed("linear", raw));
        Model m = new Model(new ModelParams(), one);
        Prediction p = m.predict(new History(null));
        double total = 0;
        for (int i = 0; i < 22; i++) {
            total += raw[i] + 1e-9;
        }
        for (int i = 0; i < 22; i++) {
            assertEquals(11 * (raw[i] + 1e-9) / total, p.scores[i], 1e-12);
        }
        assertArrayEquals(HIGH, p.numbers);
        assertArrayEquals(HIGH, p.expertPicks[0]);
        assertEquals(1, p.rankOf(22));
        assertEquals(22, p.rankOf(1));
    }

    @Test
    public void negativeAndZeroScoresAreClamped() {
        double[] s = new double[22];
        for (int i = 0; i < 22; i++) {
            s[i] = i < 11 ? -5.0 : 0.0;
        }
        s[21] = 3.0;
        List<Strategy> one = new ArrayList<Strategy>();
        one.add(new Fixed("neg", s));
        Prediction p = new Model(new ModelParams(), one).predict(new History(null));
        assertValidPrediction(p, 1);
        // negatives become 0 (+1e-9) and so tie with the zeros: lower numbers first
        assertTrue(p.contains(22));
        assertArrayEquals(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22}, p.numbers);
        assertEquals(p.scoreOf(1), p.scoreOf(12), 1e-15);
    }

    @Test
    public void allZeroScoresBecomeUniform() {
        List<Strategy> one = new ArrayList<Strategy>();
        one.add(new Fixed("zero", new double[22]));
        Prediction p = new Model(new ModelParams(), one).predict(new History(null));
        assertValidPrediction(p, 1);
        for (int n = 1; n <= 22; n++) {
            assertEquals(0.5, p.scoreOf(n), 1e-12);
        }
        assertArrayEquals(LOW, p.numbers);
    }

    @Test
    public void tiesAreBrokenByTheLowerNumber() {
        double[] s = new double[22];
        Arrays.fill(s, 1.0);
        s[21] = 2.0;      // 22 is best; everything else ties
        List<Strategy> one = new ArrayList<Strategy>();
        one.add(new Fixed("tie", s));
        Prediction p = new Model(new ModelParams(), one).predict(new History(null));
        assertArrayEquals(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22}, p.numbers);
        assertEquals(1, p.rankOf(22));
        assertEquals(2, p.rankOf(1));
        assertEquals(3, p.rankOf(2));
        assertEquals(22, p.rankOf(21));
    }

    @Test
    public void combinedScoreIsTheWeightedMixture() {
        List<Strategy> two = new ArrayList<Strategy>();
        two.add(Fixed.favouring("low", LOW));
        two.add(Fixed.favouring("high", HIGH));
        Model m = new Model(new ModelParams(), two);
        Prediction p = m.predict(new History(null));
        // 0.5 * (about 1 for LOW) + 0.5 * (about 1 for HIGH) = 0.5 everywhere
        for (int n = 1; n <= 22; n++) {
            assertEquals(0.5, p.scoreOf(n), 1e-6);
        }
        assertArrayEquals(LOW, p.expertPicks[0]);
        assertArrayEquals(HIGH, p.expertPicks[1]);
    }

    // ================================================================== observe: Hedge

    @Test
    public void weightsSumToOneAfterObserve() {
        List<Draw> ds = randomDraws(120, 17);
        Model m = new Model(new ModelParams(), null);
        for (int t = 1; t < ds.size(); t++) {
            Prediction p = m.predict(new History(ds.subList(0, t)));
            m.observe(p, ds.get(t));
            assertEquals("after " + t, 1.0, sum(m.weights()), 1e-12);
            for (double w : m.weights()) {
                assertTrue(w > 0);
            }
            assertEquals(t, m.observed());
        }
    }

    @Test
    public void hedgeUpdateWithFixedShareMatchesTheFormula() {
        ModelParams params = new ModelParams();     // eta 0.5, share 0.02
        List<Strategy> two = new ArrayList<Strategy>();
        two.add(Fixed.favouring("good", LOW));
        two.add(Fixed.favouring("bad", HIGH));
        Model m = new Model(params, two);
        Prediction p = m.predict(new History(null));
        m.observe(p, new Draw("2026-10-16", LOW));
        // good: 11 hits, loss 0; bad: 0 hits, loss 1
        double good = 0.5;
        double bad = 0.5 * Math.exp(-0.5);
        double s = good + bad;
        good = 0.98 * good / s + 0.01;
        bad = 0.98 * bad / s + 0.01;
        assertEquals(good, m.weights()[0], 1e-12);
        assertEquals(bad, m.weights()[1], 1e-12);
        assertArrayEquals(new int[] {11, 0}, m.expertHitsTotal());
        assertArrayEquals(new double[] {11.0, 0.0}, m.expertMeanHits(), 0.0);
        assertEquals(1, m.observed());
    }

    @Test
    public void perfectExpertGainsWeightOverAlwaysWrongExpert() {
        List<Strategy> two = new ArrayList<Strategy>();
        two.add(Fixed.favouring("wrong", HIGH));
        two.add(Fixed.favouring("perfect", LOW));
        Model m = new Model(new ModelParams(), two);
        Draw actual = new Draw("2026-10-16", LOW);
        double previous = m.weights()[1];
        for (int i = 0; i < 60; i++) {
            Prediction p = m.predict(new History(null));
            m.observe(p, actual);
            double w = m.weights()[1];
            assertTrue("perfect weight must not drop: " + w + " < " + previous, w >= previous - 1e-15);
            previous = w;
        }
        double[] w = m.weights();
        assertTrue(Arrays.toString(w), w[1] > w[0]);
        assertTrue("fixed share keeps the wrong expert alive", w[0] >= 0.02 / 2 - 1e-12);
        assertTrue(Arrays.toString(w), w[1] > 0.95);
        assertArrayEquals(new int[] {0, 60 * 11}, m.expertHitsTotal());
        assertArrayEquals(new double[] {0.0, 11.0}, m.expertMeanHits(), 1e-12);
        assertArrayEquals(LOW, m.predict(new History(null)).numbers);
    }

    @Test
    public void etaZeroKeepsWeightsUniform() {
        ModelParams p = new ModelParams();
        p.eta = 0;
        List<Strategy> two = new ArrayList<Strategy>();
        two.add(Fixed.favouring("a", LOW));
        two.add(Fixed.favouring("b", HIGH));
        Model m = new Model(p, two);
        for (int i = 0; i < 10; i++) {
            m.observe(m.predict(new History(null)), new Draw("2026-10-16", LOW));
        }
        assertArrayEquals(new double[] {0.5, 0.5}, m.weights(), 1e-12);
    }

    @Test
    public void shareMixesTowardUniform() {
        ModelParams p = new ModelParams();
        p.eta = 5;
        p.share = 0.5;
        List<Strategy> two = new ArrayList<Strategy>();
        two.add(Fixed.favouring("a", LOW));
        two.add(Fixed.favouring("b", HIGH));
        Model m = new Model(p, two);
        for (int i = 0; i < 50; i++) {
            m.observe(m.predict(new History(null)), new Draw("2026-10-16", LOW));
        }
        // with share 0.5 the losing expert can never fall below share / K = 0.25
        assertTrue(m.weights()[1] >= 0.25 - 1e-12);
        assertTrue(m.weights()[0] <= 0.75 + 1e-12);
    }

    // ================================================================== observe: bias

    @Test
    public void biasMovesTowardObservedNumbers() {
        List<Strategy> one = new ArrayList<Strategy>();
        one.add(new Strategies.Uniform());
        Model m = new Model(new ModelParams(), one);
        Draw actual = new Draw("2026-10-16", HIGH);
        Prediction p = m.predict(new History(null));
        assertArrayEquals("uniform alone picks the lowest numbers", LOW, p.numbers);
        m.observe(p, actual);
        double[] b = m.biases();
        for (int n = 1; n <= 22; n++) {
            if (actual.contains(n)) {
                assertTrue("bias of drawn " + n + " = " + b[n - 1], b[n - 1] > 0);
            } else {
                assertTrue("bias of not drawn " + n + " = " + b[n - 1], b[n - 1] < 0);
            }
        }
        for (int i = 0; i < 20; i++) {
            m.observe(m.predict(new History(null)), actual);
        }
        b = m.biases();
        for (int n = 12; n <= 22; n++) {
            for (int k = 1; k <= 11; k++) {
                assertTrue(b[n - 1] > b[k - 1]);
            }
        }
        // the bias now overrides the flat expert
        assertArrayEquals(HIGH, m.predict(new History(null)).numbers);
    }

    @Test
    public void biasRateZeroLeavesBiasesAtZero() {
        ModelParams p = new ModelParams();
        p.biasRate = 0;
        Model m = new Model(p, null);
        List<Draw> ds = randomDraws(40, 2);
        for (int t = 1; t < ds.size(); t++) {
            m.observe(m.predict(new History(ds.subList(0, t))), ds.get(t));
        }
        assertArrayEquals(new double[22], m.biases(), 0.0);
    }

    @Test
    public void biasesAreClampedToPlusMinusOne() {
        ModelParams p = new ModelParams();
        p.biasRate = 1;
        p.biasDecay = 0;
        List<Strategy> one = new ArrayList<Strategy>();
        one.add(new Strategies.Uniform());
        Model m = new Model(p, one);
        Draw actual = new Draw("2026-10-16", HIGH);
        for (int i = 0; i < 500; i++) {
            m.observe(m.predict(new History(null)), actual);
            for (double b : m.biases()) {
                assertTrue(b >= -1.0 && b <= 1.0);
            }
        }
        for (int n = 12; n <= 22; n++) {
            assertEquals(1.0, m.biases()[n - 1], 0.0);
        }
        Prediction pr = m.predict(new History(null));
        assertValidPrediction(pr, 1);
        assertArrayEquals(HIGH, pr.numbers);
    }

    @Test
    public void biasesStayBoundedOnRandomData() {
        List<Draw> ds = randomDraws(400, 23);
        Model m = new Model(new ModelParams(), null);
        for (int t = 1; t < ds.size(); t++) {
            Prediction p = m.predict(new History(ds.subList(0, t)));
            m.observe(p, ds.get(t));
        }
        for (double b : m.biases()) {
            assertFalse(Double.isNaN(b));
            assertTrue(b >= -1.0 && b <= 1.0);
        }
        assertValidPrediction(m.predict(new History(ds)), 9);
    }

    @Test
    public void biasesStayCentredOnRandomData() {
        // regression: comparing the 0/1 target with scores/11 instead of scores
        // made every bias drift toward +1 and saturate at the clamp
        List<Draw> ds = randomDraws(400, 23);
        Model m = new Model(new ModelParams(), null);
        for (int t = 1; t < ds.size(); t++) {
            Prediction p = m.predict(new History(ds.subList(0, t)));
            m.observe(p, ds.get(t));
        }
        double sum = 0;
        int pinned = 0;
        for (double b : m.biases()) {
            sum += b;
            if (Math.abs(b) >= 0.999) {
                pinned++;
            }
        }
        double mean = sum / 22.0;
        assertTrue("mean bias should stay near 0, was " + mean, Math.abs(mean) < 0.2);
        assertEquals("no bias should be pinned at the clamp", 0, pinned);
    }

    // ================================================================== accessors

    @Test
    public void accessorsReturnCopies() {
        Model m = new Model(new ModelParams(), null);
        double[] w = m.weights();
        w[0] = 42;
        assertEquals(1.0 / 9, m.weights()[0], 1e-15);
        double[] b = m.biases();
        b[0] = 42;
        assertEquals(0.0, m.biases()[0], 0.0);
        int[] h = m.expertHitsTotal();
        h[0] = 42;
        assertEquals(0, m.expertHitsTotal()[0]);
    }

    @Test
    public void strategiesAndParams() {
        ModelParams p = new ModelParams();
        p.eta = 1.5;
        List<Strategy> list = Strategies.defaults(p);
        Model m = new Model(p, list);
        assertEquals(9, m.strategies().size());
        for (int i = 0; i < 9; i++) {
            assertEquals(list.get(i).id(), m.strategies().get(i).id());
        }
        assertEquals(p, m.params());
        ModelParams copy = m.params();
        copy.eta = 3;
        assertEquals(1.5, m.params().eta, 0.0);
    }

    @Test
    public void paramsAreClamped() {
        ModelParams p = new ModelParams();
        p.eta = 100;
        Model m = new Model(p, null);
        assertEquals(5.0, m.params().eta, 0.0);
    }

    @Test
    public void expertMeanHitsTrackObservations() {
        List<Draw> ds = randomDraws(60, 99);
        Model m = new Model(new ModelParams(), null);
        int[] totals = new int[9];
        for (int t = 1; t < ds.size(); t++) {
            Prediction p = m.predict(new History(ds.subList(0, t)));
            for (int k = 0; k < 9; k++) {
                totals[k] += ds.get(t).hitsAgainst(p.expertPicks[k]);
            }
            m.observe(p, ds.get(t));
        }
        assertArrayEquals(totals, m.expertHitsTotal());
        double[] mean = m.expertMeanHits();
        for (int k = 0; k < 9; k++) {
            assertEquals(totals[k] / (double) (ds.size() - 1), mean[k], 1e-12);
            assertTrue(mean[k] >= 0 && mean[k] <= 11);
        }
    }

    // ================================================================== Prediction / PredictionRecord

    @Test
    public void predictionHelpers() {
        double[] scores = new double[22];
        for (int i = 0; i < 22; i++) {
            scores[i] = (i + 1) / 253.0 * 11;      // sums to 11
        }
        int[] picks = {22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12};
        Prediction p = new Prediction("2026-10-16", 5, "2026-10-17", picks, scores,
                new String[] {"x"}, new double[] {1.0}, new int[][] {picks});
        assertArrayEquals("numbers are sorted", HIGH, p.numbers);
        assertArrayEquals(HIGH, p.expertPicks[0]);
        assertEquals("12 13 14 15 16 17 18 19 20 21 22", p.numbersString(" "));
        assertEquals("12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22", p.numbersString(", "));
        assertEquals(scores[21], p.scoreOf(22), 0.0);
        assertEquals(scores[0], p.scoreOf(1), 0.0);
        assertEquals(1, p.rankOf(22));
        assertEquals(22, p.rankOf(1));
        assertEquals(0.0, p.scoreOf(0), 0.0);
        assertEquals(0.0, p.scoreOf(23), 0.0);
        picks[0] = 1;
        scores[0] = 99;
        assertArrayEquals("defensive copies", HIGH, p.numbers);
        assertEquals(1 / 253.0 * 11, p.scoreOf(1), 1e-15);
    }

    @Test
    public void predictionRecordScoresPredictionAndExperts() {
        List<Strategy> two = new ArrayList<Strategy>();
        two.add(Fixed.favouring("low", LOW));
        two.add(Fixed.favouring("high", HIGH));
        Prediction p = new Model(new ModelParams(), two).predict(new History(null));
        int[] actual = {1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16};
        Draw d = new Draw("2026-10-16", actual);
        PredictionRecord r = new PredictionRecord(p, d);
        assertEquals(d.hitsAgainst(p.numbers), r.hits);
        assertArrayEquals(new int[] {6, 5}, r.expertHits);
        assertTrue(r.prediction == p);
        assertTrue(r.actual == d);
    }
}
