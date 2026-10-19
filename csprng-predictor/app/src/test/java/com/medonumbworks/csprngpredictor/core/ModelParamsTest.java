package com.medonumbworks.csprngpredictor.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertTrue;

import java.util.HashSet;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

import org.junit.Test;

/**
 * ModelParams (SPEC 1.7): defaults, serialize / parse round trip, clamp bounds, tolerance to
 * garbage, copy and equals / hashCode.
 */
public class ModelParamsTest {

    private static ModelParams custom() {
        ModelParams p = new ModelParams();
        p.eta = 1.25;
        p.share = 0.1;
        p.biasRate = 0.3;
        p.biasDecay = 0.15;
        p.shortWindow = 7;
        p.longWindow = 99;
        p.halfLife = 22.5;
        p.dayFirst = false;
        return p;
    }

    private static void assertInRange(ModelParams p) {
        assertTrue("eta " + p.eta, p.eta >= 0 && p.eta <= 5);
        assertTrue("share " + p.share, p.share >= 0 && p.share <= 0.5);
        assertTrue("biasRate " + p.biasRate, p.biasRate >= 0 && p.biasRate <= 1);
        assertTrue("biasDecay " + p.biasDecay, p.biasDecay >= 0 && p.biasDecay <= 1);
        assertTrue("shortWindow " + p.shortWindow, p.shortWindow >= 2 && p.shortWindow <= 200);
        assertTrue("longWindow " + p.longWindow, p.longWindow >= 2 && p.longWindow <= 1000);
        assertTrue("halfLife " + p.halfLife, p.halfLife >= 1 && p.halfLife <= 500);
    }

    // ================================================================== defaults

    @Test
    public void defaults() {
        ModelParams p = new ModelParams();
        assertEquals(0.5, p.eta, 0.0);
        assertEquals(0.02, p.share, 0.0);
        assertEquals(0.1, p.biasRate, 0.0);
        assertEquals(0.05, p.biasDecay, 0.0);
        assertEquals(10, p.shortWindow);
        assertEquals(30, p.longWindow);
        assertEquals(15.0, p.halfLife, 0.0);
        assertTrue(p.dayFirst);
    }

    @Test
    public void defaultsAreInRangeAndClampIsANoOpOnThem() {
        ModelParams p = new ModelParams();
        assertInRange(p);
        p.clamp();
        assertEquals(new ModelParams(), p);
    }

    // ================================================================== serialize / parse

    @Test
    public void roundTripDefaults() {
        ModelParams p = new ModelParams();
        assertEquals(p, ModelParams.parse(p.serialize()));
    }

    @Test
    public void roundTripCustomValues() {
        ModelParams p = custom();
        ModelParams back = ModelParams.parse(p.serialize());
        assertEquals(p, back);
        assertEquals(p.hashCode(), back.hashCode());
        assertFalse(back.dayFirst);
    }

    @Test
    public void roundTripIsExactForAwkwardDoubles() {
        ModelParams p = new ModelParams();
        p.eta = 0.1 + 0.2;                  // 0.30000000000000004
        p.share = 1.0 / 3.0;
        p.biasRate = 1e-7;
        p.biasDecay = 0.123456789012345;
        p.halfLife = Math.PI;
        ModelParams back = ModelParams.parse(p.serialize());
        assertEquals(p.eta, back.eta, 0.0);
        assertEquals(p.share, back.share, 0.0);
        assertEquals(p.biasRate, back.biasRate, 0.0);
        assertEquals(p.biasDecay, back.biasDecay, 0.0);
        assertEquals(p.halfLife, back.halfLife, 0.0);
    }

    @Test
    public void roundTripOfBoundaryValues() {
        ModelParams lo = new ModelParams();
        lo.eta = 0;
        lo.share = 0;
        lo.biasRate = 0;
        lo.biasDecay = 0;
        lo.shortWindow = 2;
        lo.longWindow = 2;
        lo.halfLife = 1;
        assertEquals(lo, ModelParams.parse(lo.serialize()));
        ModelParams hi = new ModelParams();
        hi.eta = 5;
        hi.share = 0.5;
        hi.biasRate = 1;
        hi.biasDecay = 1;
        hi.shortWindow = 200;
        hi.longWindow = 1000;
        hi.halfLife = 500;
        assertEquals(hi, ModelParams.parse(hi.serialize()));
    }

    @Test
    public void serializeIsKeyValueLines() {
        String s = custom().serialize();
        String[] keys = {"eta", "share", "biasRate", "biasDecay", "shortWindow", "longWindow", "halfLife", "dayFirst"};
        Set<String> found = new HashSet<String>();
        for (String line : s.split("\n")) {
            if (line.trim().isEmpty()) {
                continue;
            }
            int eq = line.indexOf('=');
            assertTrue(line, eq > 0);
            assertTrue("duplicate key " + line, found.add(line.substring(0, eq)));
        }
        for (String k : keys) {
            assertTrue("missing " + k + " in " + s, found.contains(k));
        }
        assertTrue(s, s.contains("eta=1.25"));
        assertTrue(s, s.contains("dayFirst=false"));
        assertTrue(s, s.contains("shortWindow=7"));
    }

    @Test
    public void serializeIgnoresTheDefaultLocale() {
        Locale saved = Locale.getDefault();
        try {
            Locale.setDefault(Locale.GERMANY);
            String s = custom().serialize();
            assertTrue(s, s.contains("eta=1.25"));
            assertTrue(s, s.contains("halfLife=22.5"));
            assertFalse(s, s.contains("1,25"));
            assertEquals(custom(), ModelParams.parse(s));
            Locale.setDefault(new Locale("ar", "EG"));
            assertEquals(custom(), ModelParams.parse(custom().serialize()));
        } finally {
            Locale.setDefault(saved);
        }
    }

    @Test
    public void parseMissingKeysKeepDefaults() {
        ModelParams p = ModelParams.parse("eta=2\n");
        ModelParams expected = new ModelParams();
        expected.eta = 2;
        assertEquals(expected, p);
        assertEquals(new ModelParams(), ModelParams.parse("shortWindow=10"));
    }

    @Test
    public void parseInvalidValuesKeepDefaults() {
        ModelParams p = ModelParams.parse("eta=abc\nshare=\nbiasRate=NaN\nshortWindow=ten\nlongWindow=\n"
                + "halfLife=--3\ndayFirst=maybe\n");
        assertEquals(new ModelParams(), p);
    }

    @Test
    public void parseGarbageGivesDefaults() {
        String[] garbage = {null, "", "   ", "garbage", "=", "==", "=5", "eta", "\n\n\n", "{\"eta\":1}",
                "eta:1", "\u0000\u0001", "unknown=5\nother=x"};
        for (String g : garbage) {
            assertEquals(String.valueOf(g), new ModelParams(), ModelParams.parse(g));
        }
    }

    @Test
    public void parseNeverThrowsAndAlwaysClamps() {
        Random rnd = new Random(1);
        String alphabet = "etashrbiRDcywWLnglfFd=0123456789.-+eE \nNaInfity";
        for (int round = 0; round < 500; round++) {
            StringBuilder sb = new StringBuilder();
            int len = rnd.nextInt(200);
            for (int i = 0; i < len; i++) {
                sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
            }
            ModelParams p = ModelParams.parse(sb.toString());
            assertNotNull(p);
            assertInRange(p);
        }
    }

    @Test
    public void parseOutOfRangeValuesEndUpInRange() {
        ModelParams p = ModelParams.parse("eta=99\nshare=-1\nbiasRate=7\nbiasDecay=-0.5\nshortWindow=0\n"
                + "longWindow=999999\nhalfLife=0\n");
        assertInRange(p);
        p = ModelParams.parse("eta=Infinity\nshare=-Infinity\nhalfLife=1e308\nshortWindow=99999999999\n");
        assertInRange(p);
    }

    @Test
    public void parseToleratesWhitespaceCrlfAndComments() {
        ModelParams p = ModelParams.parse("# saved by test\r\n  eta = 0.75 \r\n\r\ndayFirst=false\r\nlongWindow= 44\r\n");
        assertEquals(0.75, p.eta, 0.0);
        assertFalse(p.dayFirst);
        assertEquals(44, p.longWindow);
        assertEquals(0.02, p.share, 0.0);
    }

    @Test
    public void parseIgnoresUnknownKeys() {
        ModelParams p = ModelParams.parse("colour=blue\neta=0.25\nversion=3");
        assertEquals(0.25, p.eta, 0.0);
        assertEquals(10, p.shortWindow);
    }

    @Test
    public void parseDayFirstValues() {
        assertTrue(ModelParams.parse("dayFirst=true").dayFirst);
        assertFalse(ModelParams.parse("dayFirst=false").dayFirst);
    }

    // ================================================================== clamp

    @Test
    public void clampLowerBounds() {
        ModelParams p = new ModelParams();
        p.eta = -1;
        p.share = -0.1;
        p.biasRate = -2;
        p.biasDecay = -0.01;
        p.shortWindow = 1;
        p.longWindow = -5;
        p.halfLife = 0.5;
        p.clamp();
        assertEquals(0.0, p.eta, 0.0);
        assertEquals(0.0, p.share, 0.0);
        assertEquals(0.0, p.biasRate, 0.0);
        assertEquals(0.0, p.biasDecay, 0.0);
        assertEquals(2, p.shortWindow);
        assertEquals(2, p.longWindow);
        assertEquals(1.0, p.halfLife, 0.0);
    }

    @Test
    public void clampUpperBounds() {
        ModelParams p = new ModelParams();
        p.eta = 5.0001;
        p.share = 0.9;
        p.biasRate = 1.5;
        p.biasDecay = 2;
        p.shortWindow = 201;
        p.longWindow = 1001;
        p.halfLife = 10000;
        p.clamp();
        assertEquals(5.0, p.eta, 0.0);
        assertEquals(0.5, p.share, 0.0);
        assertEquals(1.0, p.biasRate, 0.0);
        assertEquals(1.0, p.biasDecay, 0.0);
        assertEquals(200, p.shortWindow);
        assertEquals(1000, p.longWindow);
        assertEquals(500.0, p.halfLife, 0.0);
    }

    @Test
    public void clampKeepsExactBoundsAndInRangeValues() {
        ModelParams lo = new ModelParams();
        lo.eta = 0;
        lo.share = 0;
        lo.biasRate = 0;
        lo.biasDecay = 0;
        lo.shortWindow = 2;
        lo.longWindow = 2;
        lo.halfLife = 1;
        ModelParams before = lo.copy();
        lo.clamp();
        assertEquals(before, lo);
        ModelParams hi = new ModelParams();
        hi.eta = 5;
        hi.share = 0.5;
        hi.biasRate = 1;
        hi.biasDecay = 1;
        hi.shortWindow = 200;
        hi.longWindow = 1000;
        hi.halfLife = 500;
        before = hi.copy();
        hi.clamp();
        assertEquals(before, hi);
        ModelParams mid = custom();
        mid.clamp();
        assertEquals(custom(), mid);
    }

    @Test
    public void clampExtremeValues() {
        ModelParams p = new ModelParams();
        p.eta = Double.POSITIVE_INFINITY;
        p.share = Double.NEGATIVE_INFINITY;
        p.biasRate = Double.NaN;
        p.biasDecay = Double.MAX_VALUE;
        p.shortWindow = Integer.MIN_VALUE;
        p.longWindow = Integer.MAX_VALUE;
        p.halfLife = Double.NaN;
        p.clamp();
        assertInRange(p);
        assertEquals(5.0, p.eta, 0.0);
        assertEquals(0.0, p.share, 0.0);
        assertEquals(1.0, p.biasDecay, 0.0);
        assertEquals(2, p.shortWindow);
        assertEquals(1000, p.longWindow);
        assertFalse(Double.isNaN(p.biasRate));
        assertFalse(Double.isNaN(p.halfLife));
    }

    @Test
    public void clampDoesNotTouchDayFirst() {
        ModelParams p = new ModelParams();
        p.dayFirst = false;
        p.eta = 100;
        p.clamp();
        assertFalse(p.dayFirst);
    }

    // ================================================================== copy / equals

    @Test
    public void copyIsEqualAndIndependent() {
        ModelParams p = custom();
        ModelParams c = p.copy();
        assertNotSame(p, c);
        assertEquals(p, c);
        assertEquals(p.hashCode(), c.hashCode());
        c.eta = 4;
        c.dayFirst = true;
        assertEquals(1.25, p.eta, 0.0);
        assertFalse(p.dayFirst);
        assertNotEquals(p, c);
    }

    @Test
    public void equalsLooksAtEveryField() {
        ModelParams base = new ModelParams();
        ModelParams[] variants = new ModelParams[8];
        for (int i = 0; i < variants.length; i++) {
            variants[i] = new ModelParams();
        }
        variants[0].eta = 0.6;
        variants[1].share = 0.03;
        variants[2].biasRate = 0.2;
        variants[3].biasDecay = 0.06;
        variants[4].shortWindow = 11;
        variants[5].longWindow = 31;
        variants[6].halfLife = 16;
        variants[7].dayFirst = false;
        for (int i = 0; i < variants.length; i++) {
            assertNotEquals("variant " + i, base, variants[i]);
            assertNotEquals("variant " + i, variants[i], base);
        }
        assertEquals(base, new ModelParams());
        assertNotEquals(base, null);
        assertNotEquals(base, "eta=0.5");
    }

    @Test
    public void toStringMentionsTheValues() {
        String s = custom().toString();
        assertNotNull(s);
        assertTrue(s, s.contains("eta"));
        assertTrue(s, s.contains("1.25"));
    }
}
