package com.medonumbworks.csprngpredictor.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Locale;
import java.util.TimeZone;

import org.junit.Test;

/**
 * DateUtil: every accepted format of normalize (SPEC 1.3), the ambiguity rule, invalid
 * dates, year limits, findDate inside text, ISO arithmetic and display.
 */
public class DateUtilTest {

    private static final String ISO = "2026-10-16";

    // ------------------------------------------------------------------ isIso

    @Test
    public void isIsoAcceptsRealZeroPaddedDates() {
        assertTrue(DateUtil.isIso("2026-10-16"));
        assertTrue(DateUtil.isIso("2024-02-29"));
        assertTrue(DateUtil.isIso("2000-02-29"));
        assertTrue(DateUtil.isIso("1999-12-31"));
        assertTrue(DateUtil.isIso("2026-01-01"));
    }

    @Test
    public void isIsoRejectsInvalidCalendarDates() {
        assertFalse(DateUtil.isIso("2026-02-29"));
        assertFalse(DateUtil.isIso("1900-02-29"));
        assertFalse(DateUtil.isIso("2026-02-30"));
        assertFalse(DateUtil.isIso("2026-04-31"));
        assertFalse(DateUtil.isIso("2026-13-01"));
        assertFalse(DateUtil.isIso("2026-00-10"));
        assertFalse(DateUtil.isIso("2026-10-00"));
        assertFalse(DateUtil.isIso("2026-10-32"));
    }

    @Test
    public void isIsoIsStrictAboutShape() {
        assertFalse(DateUtil.isIso(null));
        assertFalse(DateUtil.isIso(""));
        assertFalse(DateUtil.isIso("2026-1-5"));
        assertFalse(DateUtil.isIso("2026/10/16"));
        assertFalse(DateUtil.isIso("16.10.2026"));
        assertFalse(DateUtil.isIso("20261016"));
        assertFalse(DateUtil.isIso(" 2026-10-16"));
        assertFalse(DateUtil.isIso("2026-10-16 "));
        assertFalse(DateUtil.isIso("2026-10-16T20:00"));
        assertFalse(DateUtil.isIso("abcd-ef-gh"));
        assertFalse(DateUtil.isIso("2026-1a-16"));
    }

    // ------------------------------------------------------------------ normalize: formats

    @Test
    public void normalizeIsoForms() {
        assertEquals(ISO, DateUtil.normalize("2026-10-16", true));
        assertEquals(ISO, DateUtil.normalize("2026/10/16", true));
        assertEquals(ISO, DateUtil.normalize("2026.10.16", true));
        assertEquals(ISO, DateUtil.normalize("2026-10-16", false));
    }

    @Test
    public void normalizeEightDigitCompactForm() {
        assertEquals(ISO, DateUtil.normalize("20261016", true));
        assertEquals("2024-02-29", DateUtil.normalize("20240229", false));
        assertNull(DateUtil.normalize("20261332", true));
        assertNull(DateUtil.normalize("20260230", true));
        assertNull(DateUtil.normalize("2026101", true));
        assertNull(DateUtil.normalize("202610160", true));
    }

    @Test
    public void normalizeDayFirstForms() {
        assertEquals(ISO, DateUtil.normalize("16.10.2026", true));
        assertEquals(ISO, DateUtil.normalize("16-10-2026", true));
        assertEquals(ISO, DateUtil.normalize("16/10/2026", true));
        // a > 12 forces day-first even when the flag says month-first
        assertEquals(ISO, DateUtil.normalize("16.10.2026", false));
        assertEquals(ISO, DateUtil.normalize("16-10-2026", false));
        assertEquals(ISO, DateUtil.normalize("16/10/2026", false));
    }

    @Test
    public void normalizeMonthFirstForms() {
        assertEquals(ISO, DateUtil.normalize("10/16/2026", false));
        assertEquals(ISO, DateUtil.normalize("10-16-2026", false));
        // b > 12 forces month-first even when the flag says day-first
        assertEquals(ISO, DateUtil.normalize("10/16/2026", true));
        assertEquals(ISO, DateUtil.normalize("10-16-2026", true));
        assertEquals(ISO, DateUtil.normalize("10.16.2026", true));
    }

    @Test
    public void normalizeSingleDigitDayAndMonth() {
        assertEquals("2026-02-01", DateUtil.normalize("1.2.2026", true));
        assertEquals("2026-01-02", DateUtil.normalize("1.2.2026", false));
        assertEquals("2026-03-05", DateUtil.normalize("5/3/2026", true));
        assertEquals("2026-05-03", DateUtil.normalize("5/3/2026", false));
    }

    @Test
    public void normalizeTwoDigitYearsAre20yy() {
        assertEquals(ISO, DateUtil.normalize("16.10.26", true));
        assertEquals(ISO, DateUtil.normalize("16/10/26", true));
        assertEquals(ISO, DateUtil.normalize("16-10-26", true));
        assertEquals(ISO, DateUtil.normalize("10/16/26", false));
        assertEquals(ISO, DateUtil.normalize("10-16-26", true));
        assertEquals("2000-01-01", DateUtil.normalize("01.01.00", true));
        assertEquals("2099-12-31", DateUtil.normalize("31.12.99", true));
        assertEquals("2026-06-05", DateUtil.normalize("05.06.26", true));
        assertEquals("2026-05-06", DateUtil.normalize("05.06.26", false));
    }

    @Test
    public void normalizeIsoDateTimesKeepOnlyTheDate() {
        assertEquals(ISO, DateUtil.normalize("2026-10-16T20:00:00", true));
        assertEquals(ISO, DateUtil.normalize("2026-10-16 20:00", true));
        assertEquals(ISO, DateUtil.normalize("2026-10-16T20:00", false));
        assertEquals(ISO, DateUtil.normalize("2026-10-16 20:00:59", true));
        assertEquals(ISO, DateUtil.normalize("2026-10-16T20:00:00Z", true));
        assertEquals(ISO, DateUtil.normalize("2026-10-16T20:00:00.000+02:00", true));
        assertEquals(ISO, DateUtil.normalize("2026-10-16T23:59:59", true));
    }

    @Test
    public void normalizeTrimsAndStripsQuotes() {
        assertEquals(ISO, DateUtil.normalize("  2026-10-16  ", true));
        assertEquals(ISO, DateUtil.normalize("\"2026-10-16\"", true));
        assertEquals(ISO, DateUtil.normalize("'16.10.2026'", true));
        assertEquals(ISO, DateUtil.normalize(" \"16/10/2026\" ", true));
        assertEquals(ISO, DateUtil.normalize("\t2026-10-16\t", true));
        assertEquals(ISO, DateUtil.normalize("\"2026-10-16 20:00\"", true));
    }

    // ------------------------------------------------------------------ normalize: ambiguity rule

    @Test
    public void ambiguousDatesFollowTheDayFirstFlag() {
        assertEquals("2026-06-05", DateUtil.normalize("05.06.2026", true));
        assertEquals("2026-05-06", DateUtil.normalize("05.06.2026", false));
        assertEquals("2026-06-05", DateUtil.normalize("05/06/2026", true));
        assertEquals("2026-05-06", DateUtil.normalize("05/06/2026", false));
        assertEquals("2026-06-05", DateUtil.normalize("05-06-2026", true));
        assertEquals("2026-05-06", DateUtil.normalize("05-06-2026", false));
        assertEquals("2026-12-12", DateUtil.normalize("12/12/2026", true));
        assertEquals("2026-12-12", DateUtil.normalize("12/12/2026", false));
    }

    @Test
    public void firstPartAbove12IsAlwaysTheDay() {
        assertEquals("2026-05-13", DateUtil.normalize("13/05/2026", false));
        assertEquals("2026-05-13", DateUtil.normalize("13/05/2026", true));
        assertEquals("2026-01-31", DateUtil.normalize("31-01-2026", false));
    }

    @Test
    public void secondPartAbove12IsAlwaysTheDay() {
        assertEquals("2026-05-13", DateUtil.normalize("05/13/2026", true));
        assertEquals("2026-05-13", DateUtil.normalize("05/13/2026", false));
        assertEquals("2026-01-31", DateUtil.normalize("01-31-2026", true));
    }

    @Test
    public void bothPartsAbove12IsInvalid() {
        assertNull(DateUtil.normalize("13/13/2026", true));
        assertNull(DateUtil.normalize("13/13/2026", false));
        assertNull(DateUtil.normalize("31.31.2026", true));
    }

    @Test
    public void ambiguityFallbackStillNeedsARealDate() {
        // 31/04 day-first is invalid (April has 30 days); the flag does not "rescue" it
        assertNull(DateUtil.normalize("31/04/2026", true));
        // 02/30 month-first: b > 12 -> month-first -> Feb 30 -> invalid
        assertNull(DateUtil.normalize("02/30/2026", true));
    }

    // ------------------------------------------------------------------ normalize: invalid input

    @Test
    public void invalidCalendarDatesAreNull() {
        assertNull(DateUtil.normalize("31.02.2026", true));
        assertNull(DateUtil.normalize("31.02.2026", false));
        assertNull(DateUtil.normalize("30.02.2024", true));
        assertNull(DateUtil.normalize("29.02.2025", true));
        assertNull(DateUtil.normalize("2026-02-29", true));
        assertNull(DateUtil.normalize("2026-13-01", true));
        assertNull(DateUtil.normalize("2026-00-01", true));
        assertNull(DateUtil.normalize("2026-01-00", true));
        assertNull(DateUtil.normalize("00.10.2026", true));
        assertNull(DateUtil.normalize("16.00.2026", true));
        assertNull(DateUtil.normalize("31.04.2026", true));
        assertNull(DateUtil.normalize("2026-02-29T10:00:00", true));
    }

    @Test
    public void leapDaysAreHandled() {
        assertEquals("2024-02-29", DateUtil.normalize("29.02.2024", true));
        assertEquals("2000-02-29", DateUtil.normalize("2000-02-29", true));
        assertNull(DateUtil.normalize("1900-02-29", true));
        assertNull(DateUtil.normalize("2100-02-29", true));
    }

    @Test
    public void yearsMustBeBetween1900And2100() {
        assertEquals("1900-01-01", DateUtil.normalize("1900-01-01", true));
        assertEquals("2100-12-31", DateUtil.normalize("2100-12-31", true));
        assertEquals("1900-01-01", DateUtil.normalize("01.01.1900", true));
        assertEquals("2100-12-31", DateUtil.normalize("31.12.2100", true));
        assertNull(DateUtil.normalize("1899-12-31", true));
        assertNull(DateUtil.normalize("2101-01-01", true));
        assertNull(DateUtil.normalize("16.10.1899", true));
        assertNull(DateUtil.normalize("16.10.2101", true));
        assertNull(DateUtil.normalize("18991231", true));
        assertNull(DateUtil.normalize("21010101", true));
        assertNull(DateUtil.normalize("0001-01-01", true));
        assertNull(DateUtil.normalize("9999-12-31", true));
    }

    @Test
    public void garbageIsNull() {
        assertNull(DateUtil.normalize(null, true));
        assertNull(DateUtil.normalize("", true));
        assertNull(DateUtil.normalize("   ", true));
        assertNull(DateUtil.normalize("\"\"", true));
        assertNull(DateUtil.normalize("abc", true));
        assertNull(DateUtil.normalize("date", true));
        assertNull(DateUtil.normalize("2026", true));
        assertNull(DateUtil.normalize("12345", true));
        assertNull(DateUtil.normalize("16.10", true));
        assertNull(DateUtil.normalize("16.10.", true));
        assertNull(DateUtil.normalize("1 4 7 9 11", true));
        assertNull(DateUtil.normalize("1-4-7-9-11-13", true));
        assertNull(DateUtil.normalize("2026-10-16x", true));
        assertNull(DateUtil.normalize("x2026-10-16", true));
        assertNull(DateUtil.normalize("2026-10/16", true));
        assertNull(DateUtil.normalize("16.10/2026", true));
        assertNull(DateUtil.normalize("20:00", true));
        assertNull(DateUtil.normalize("-2026-10-16", true));
        assertNull(DateUtil.normalize("2026--10-16", true));
        assertNull(DateUtil.normalize("16.10.202", true));
        assertNull(DateUtil.normalize("16.10.20261", true));
    }

    @Test
    public void normalizeOutputIsAlwaysIso() {
        String[] inputs = {"2026-10-16", "2026/1/2", "1.2.2026", "20261016", "16.10.26",
                "2026-10-16T20:00:00", "10/16/2026", "'2026.10.16'"};
        for (String in : inputs) {
            String out = DateUtil.normalize(in, true);
            assertNotNull(in, out);
            assertTrue(in + " -> " + out, DateUtil.isIso(out));
            assertEquals("idempotent for " + in, out, DateUtil.normalize(out, false));
        }
    }

    // ------------------------------------------------------------------ findDate

    @Test
    public void findDateInsideText() {
        assertEquals(ISO, DateUtil.findDate("Draw 2026-10-16: 1 4 7 9 11", true));
        assertEquals(ISO, DateUtil.findDate("Ziehung vom 16.10.2026, Zahlen folgen", true));
        assertEquals(ISO, DateUtil.findDate("drawn on 10/16/2026 at noon", true));
        assertEquals(ISO, DateUtil.findDate("[2026/10/16]", true));
        assertEquals(ISO, DateUtil.findDate("id 77 date 20261016 numbers", true));
        assertEquals(ISO, DateUtil.findDate("2026-10-16", true));
    }

    @Test
    public void findDateWithTimeInsideText() {
        assertEquals(ISO, DateUtil.findDate("drawn 2026-10-16T20:00:00 numbers 1 2 3", true));
        assertEquals(ISO, DateUtil.findDate("drawn 2026-10-16 20:00 numbers 1 2 3", true));
    }

    @Test
    public void findDateReturnsTheFirstDate() {
        assertEquals("2026-01-02", DateUtil.findDate("from 2026-01-02 to 2026-03-04", true));
        assertEquals("2026-03-04", DateUtil.findDate("from 2026-03-04 to 2026-01-02", true));
    }

    @Test
    public void findDateSkipsInvalidCandidates() {
        assertEquals("2026-03-01", DateUtil.findDate("bad 31.02.2026 good 01.03.2026", true));
        assertNull(DateUtil.findDate("only 31.02.2026 here", true));
    }

    @Test
    public void findDateUsesTheDayFirstFlag() {
        assertEquals("2026-06-05", DateUtil.findDate("on 05/06/2026 ok", true));
        assertEquals("2026-05-06", DateUtil.findDate("on 05/06/2026 ok", false));
    }

    @Test
    public void findDateTwoDigitYearInText() {
        assertEquals(ISO, DateUtil.findDate("Draw of 16.10.26: 1 4 7", true));
    }

    @Test
    public void findDateIgnoresNumberLists() {
        assertNull(DateUtil.findDate("1-4-7-9-11-13-15-17-19-21-22", true));
        assertNull(DateUtil.findDate("1/4/7/9/11/13/15/17/19/21/22", true));
        assertNull(DateUtil.findDate("1 4 7 9 11 13 15 17 19 21 22", true));
        assertNull(DateUtil.findDate("1,4,7,9,11,13,15,17,19,21,22", true));
        assertNull(DateUtil.findDate("1.4.7.9.11.13.15", true));
    }

    @Test
    public void findDateWithoutDateIsNull() {
        assertNull(DateUtil.findDate(null, true));
        assertNull(DateUtil.findDate("", true));
        assertNull(DateUtil.findDate("no date here", true));
        assertNull(DateUtil.findDate("year 2026 only", true));
        assertNull(DateUtil.findDate("123456789", true));
    }

    // ------------------------------------------------------------------ arithmetic

    @Test
    public void addDaysAcrossMonthBoundaries() {
        assertEquals("2026-02-01", DateUtil.addDays("2026-01-31", 1));
        assertEquals("2026-01-31", DateUtil.addDays("2026-02-01", -1));
        assertEquals("2026-05-01", DateUtil.addDays("2026-04-30", 1));
        assertEquals("2026-10-17", DateUtil.addDays(ISO, 1));
        assertEquals("2026-10-15", DateUtil.addDays(ISO, -1));
        assertEquals(ISO, DateUtil.addDays(ISO, 0));
        assertEquals("2026-11-15", DateUtil.addDays(ISO, 30));
    }

    @Test
    public void addDaysAcrossYearBoundaries() {
        assertEquals("2027-01-01", DateUtil.addDays("2026-12-31", 1));
        assertEquals("2026-12-31", DateUtil.addDays("2027-01-01", -1));
        assertEquals("2027-10-16", DateUtil.addDays(ISO, 365));
        assertEquals("2025-10-16", DateUtil.addDays(ISO, -365));
        assertEquals("2000-01-01", DateUtil.addDays("1999-12-31", 1));
    }

    @Test
    public void addDaysAcrossLeapDays() {
        assertEquals("2024-02-29", DateUtil.addDays("2024-02-28", 1));
        assertEquals("2024-03-01", DateUtil.addDays("2024-02-29", 1));
        assertEquals("2023-03-01", DateUtil.addDays("2023-02-28", 1));
        assertEquals("2000-02-29", DateUtil.addDays("2000-02-28", 1));
        assertEquals("1900-03-01", DateUtil.addDays("1900-02-28", 1));
        assertEquals("2023-03-01", DateUtil.addDays("2024-03-01", -366));
        assertEquals("2024-02-29", DateUtil.addDays("2024-03-01", -1));
    }

    @Test
    public void addDaysMatchesGregorianCalendar() {
        GregorianCalendar cal = new GregorianCalendar(TimeZone.getTimeZone("UTC"), Locale.US);
        cal.clear();
        cal.set(1995, Calendar.JANUARY, 1);
        String iso = "1995-01-01";
        for (int i = 0; i < 15000; i++) {
            String expected = String.format(Locale.US, "%04d-%02d-%02d", cal.get(Calendar.YEAR),
                    cal.get(Calendar.MONTH) + 1, cal.get(Calendar.DAY_OF_MONTH));
            assertEquals("step " + i, expected, iso);
            iso = DateUtil.addDays(iso, 1);
            cal.add(Calendar.DAY_OF_MONTH, 1);
        }
    }

    @Test
    public void addDaysLargeJumps() {
        // 2026-10-16 .. 2126-10-16 has 24 leap days (2100 is not a leap year)
        assertEquals("2126-10-16", DateUtil.addDays(ISO, 36500 + 24));
        // 1926-10-16 .. 2026-10-16 has 25 leap days (2000 is a leap year)
        assertEquals("1926-10-16", DateUtil.addDays(ISO, -(36500 + 25)));
    }

    @Test
    public void daysBetweenIsBMinusA() {
        assertEquals(1, DateUtil.daysBetween("2026-10-16", "2026-10-17"));
        assertEquals(-1, DateUtil.daysBetween("2026-10-17", "2026-10-16"));
        assertEquals(0, DateUtil.daysBetween(ISO, ISO));
        assertEquals(1, DateUtil.daysBetween("2025-12-31", "2026-01-01"));
        assertEquals(366, DateUtil.daysBetween("2024-01-01", "2025-01-01"));
        assertEquals(365, DateUtil.daysBetween("2025-01-01", "2026-01-01"));
        assertEquals(29, DateUtil.daysBetween("2024-02-01", "2024-03-01"));
        assertEquals(28, DateUtil.daysBetween("2026-02-01", "2026-03-01"));
    }

    @Test
    public void daysBetweenIsTheInverseOfAddDays() {
        for (int k = -2000; k <= 2000; k += 7) {
            String other = DateUtil.addDays(ISO, k);
            assertEquals(k, DateUtil.daysBetween(ISO, other));
            assertEquals(-k, DateUtil.daysBetween(other, ISO));
            assertEquals(ISO, DateUtil.addDays(other, -k));
        }
    }

    @Test
    public void arithmeticRejectsNonIsoInput() {
        try {
            DateUtil.addDays("16.10.2026", 1);
            fail("addDays accepted a non-ISO date");
        } catch (IllegalArgumentException expected) {
            // documented contract
        }
        try {
            DateUtil.daysBetween("2026-02-30", ISO);
            fail("daysBetween accepted an invalid date");
        } catch (IllegalArgumentException expected) {
            // documented contract
        }
    }

    // ------------------------------------------------------------------ compare, display, today

    @Test
    public void compareOrdersIsoDates() {
        assertTrue(DateUtil.compare("2026-01-01", "2026-01-02") < 0);
        assertTrue(DateUtil.compare("2026-01-02", "2026-01-01") > 0);
        assertEquals(0, DateUtil.compare(ISO, ISO));
        assertTrue(DateUtil.compare("2025-12-31", "2026-01-01") < 0);
        assertTrue(DateUtil.compare("2026-09-30", "2026-10-01") < 0);
    }

    @Test
    public void displayIsDayMonthYear() {
        assertEquals("16.10.2026", DateUtil.display(ISO));
        assertEquals("01.02.2026", DateUtil.display("2026-02-01"));
        assertEquals(ISO, DateUtil.normalize(DateUtil.display(ISO), true));
    }

    @Test
    public void displayRoundTripsForEveryDayOfAYear() {
        String iso = "2024-01-01";
        for (int i = 0; i < 366; i++) {
            assertEquals(iso, DateUtil.normalize(DateUtil.display(iso), true));
            iso = DateUtil.addDays(iso, 1);
        }
    }

    @Test
    public void displayDoesNotCrashOnBadInput() {
        assertNotNull(DateUtil.display(null));
        assertNotNull(DateUtil.display("garbage"));
    }

    @Test
    public void todayIsIsoAndCloseToTheSystemClock() {
        String today = DateUtil.today();
        assertTrue(today, DateUtil.isIso(today));
        Calendar c = Calendar.getInstance();
        String sys = String.format(Locale.US, "%04d-%02d-%02d", c.get(Calendar.YEAR),
                c.get(Calendar.MONTH) + 1, c.get(Calendar.DAY_OF_MONTH));
        assertTrue(today + " vs " + sys, Math.abs(DateUtil.daysBetween(today, sys)) <= 1);
    }
}
