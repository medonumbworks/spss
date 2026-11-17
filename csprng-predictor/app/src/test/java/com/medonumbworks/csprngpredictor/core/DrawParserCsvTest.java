package com.medonumbworks.csprngpredictor.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.junit.Test;

/**
 * DrawParser CSV / TSV / TXT import (SPEC 1.4) plus the static helpers extractNumbers,
 * asDraw and assignDates.
 */
public class DrawParserCsvTest {

    /** The draw used in most rows. It deliberately contains none of 10, 16, 20, 26. */
    private static final int[] N = {1, 4, 7, 9, 11, 13, 15, 17, 19, 21, 22};
    private static final String ISO = "2026-10-16";

    private static String join(int[] nums, String sep) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < nums.length; i++) {
            if (i > 0) {
                sb.append(sep);
            }
            sb.append(nums[i]);
        }
        return sb.toString();
    }

    private static ParseResult csv(String content) {
        return DrawParser.parseCsv(content, true);
    }

    private static void assertSingleDated(ParseResult r, String date, int[] nums) {
        assertEquals("dated: " + r + " " + r.warnings, 1, r.dated.size());
        assertEquals(0, r.undated.size());
        assertEquals(date, r.dated.get(0).date);
        assertArrayEquals(nums, r.dated.get(0).numbers);
    }

    private static void assertNoDraw(ParseResult r) {
        assertTrue("expected no draw but got " + r.dated + " / undated " + r.undated.size(), r.isEmpty());
    }

    private static boolean hasWarningContaining(ParseResult r, String fragment) {
        for (String w : r.warnings) {
            if (w.contains(fragment)) {
                return true;
            }
        }
        return false;
    }

    // ================================================================== delimiters

    @Test
    public void commaDelimited() {
        assertSingleDated(csv(ISO + "," + join(N, ",")), ISO, N);
    }

    @Test
    public void semicolonDelimited() {
        assertSingleDated(csv("16.10.2026;" + join(N, ";")), ISO, N);
    }

    @Test
    public void tabDelimited() {
        assertSingleDated(csv(ISO + "\t" + join(N, "\t")), ISO, N);
    }

    @Test
    public void pipeDelimited() {
        assertSingleDated(csv(ISO + "|" + join(N, "|")), ISO, N);
    }

    @Test
    public void pipeTableWithOuterPipes() {
        assertSingleDated(csv("| " + ISO + " | " + join(N, " | ") + " |"), ISO, N);
    }

    @Test
    public void whitespaceDelimited() {
        assertSingleDated(csv(ISO + " " + join(N, " ")), ISO, N);
        assertSingleDated(csv(ISO + "   " + join(N, "  \t ")), ISO, N);
    }

    @Test
    public void spacesAroundDelimitedCells() {
        assertSingleDated(csv(" " + ISO + " , " + join(N, " , ") + " "), ISO, N);
    }

    @Test
    public void delimiterIsDetectedPerLine() {
        String content = "2026-10-14," + join(N, ",") + "\n"
                + "2026-10-15;" + join(N, ";") + "\n"
                + "2026-10-16\t" + join(N, "\t") + "\n"
                + "2026-10-17|" + join(N, "|") + "\n"
                + "2026-10-18 " + join(N, " ") + "\n";
        ParseResult r = csv(content);
        assertEquals(r.warnings.toString(), 5, r.dated.size());
        String[] dates = {"2026-10-14", "2026-10-15", "2026-10-16", "2026-10-17", "2026-10-18"};
        for (int i = 0; i < dates.length; i++) {
            assertEquals(dates[i], r.dated.get(i).date);
            assertArrayEquals(N, r.dated.get(i).numbers);
        }
        assertEquals(5, r.totalLines);
        assertEquals(0, r.skippedLines);
        assertTrue(r.warnings.isEmpty());
    }

    // ================================================================== headers

    @Test
    public void headerLineIsSkippedSilently() {
        ParseResult r = csv(DrawCsv.HEADER + "\n" + ISO + "," + join(N, ","));
        assertSingleDated(r, ISO, N);
        assertTrue(r.warnings.toString(), r.warnings.isEmpty());
        assertEquals(1, r.skippedLines);
        assertEquals(2, r.totalLines);
    }

    @Test
    public void ownHeaderIsNotMistakenForADraw() {
        // n1..n11 hold the digits 1..11, which must not become an undated draw
        ParseResult r = csv(DrawCsv.HEADER);
        assertNoDraw(r);
        assertTrue(r.warnings.isEmpty());
        assertEquals(1, r.skippedLines);
    }

    @Test
    public void variousHeaderStylesAreSkipped() {
        String content = "Date;Numbers\n"
                + "\"Datum\",\"Zahl 1\",\"Zahl 2\"\n"
                + "Draw results export\n"
                + "id,date,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11\n"
                + "# comment line\n"
                + ISO + ";" + join(N, ";") + "\n";
        ParseResult r = csv(content);
        assertSingleDated(r, ISO, N);
        assertTrue(r.warnings.toString(), r.warnings.isEmpty());
        assertEquals(5, r.skippedLines);
    }

    // ================================================================== numbers in one cell

    @Test
    public void allNumbersInOneQuotedCell() {
        assertSingleDated(csv(ISO + ",\"" + join(N, " ") + "\""), ISO, N);
    }

    @Test
    public void numbersSeparatedByDashSlashPlusInOneCell() {
        assertSingleDated(csv(ISO + ";" + join(N, "-")), ISO, N);
        assertSingleDated(csv(ISO + ";" + join(N, "/")), ISO, N);
        assertSingleDated(csv(ISO + ";" + join(N, "+")), ISO, N);
        assertSingleDated(csv(ISO + "," + join(N, " - ")), ISO, N);
    }

    @Test
    public void numbersSplitOverSeveralCells() {
        assertSingleDated(csv(ISO + ",1 4 7,9 11 13,15,17 19 21 22"), ISO, N);
    }

    @Test
    public void spacedNumbersCellWithTabDelimiter() {
        assertSingleDated(csv(ISO + "\t" + join(N, " ")), ISO, N);
    }

    // ================================================================== date position / formats

    @Test
    public void dateInTheMiddle() {
        assertSingleDated(csv("1,4,7,9,11," + ISO + ",13,15,17,19,21,22"), ISO, N);
    }

    @Test
    public void dateAtTheEnd() {
        assertSingleDated(csv(join(N, ";") + ";16.10.2026"), ISO, N);
    }

    @Test
    public void dateFormatsInCells() {
        String[] dates = {"2026-10-16", "2026/10/16", "2026.10.16", "20261016", "16.10.2026",
                "16-10-2026", "16/10/2026", "10/16/2026", "16.10.26", "2026-10-16T20:00:00"};
        for (String d : dates) {
            ParseResult r = csv(d + "," + join(N, ","));
            assertEquals(d, 1, r.dated.size());
            assertEquals(d, ISO, r.dated.get(0).date);
            assertArrayEquals(d, N, r.dated.get(0).numbers);
        }
    }

    @Test
    public void dayFirstFlagDecidesAmbiguousDates() {
        String line = "05/06/2026," + join(N, ",");
        assertEquals("2026-06-05", DrawParser.parseCsv(line, true).dated.get(0).date);
        assertEquals("2026-05-06", DrawParser.parseCsv(line, false).dated.get(0).date);
    }

    @Test
    public void dateInsideFreeText() {
        ParseResult r = csv("Draw 2026-10-16: " + join(N, " "));
        assertSingleDated(r, ISO, N);
    }

    @Test
    public void dateInsideFreeTextWithCommaNumbers() {
        ParseResult r = csv("Ziehung vom 16.10.2026: " + join(N, ", "));
        assertSingleDated(r, ISO, N);
    }

    @Test
    public void firstDateCellWinsAndOtherDatesDoNotLeak() {
        ParseResult r = csv(ISO + ",2026-10-15," + join(N, ","));
        assertSingleDated(r, ISO, N);
    }

    // ================================================================== date digits never leak

    @Test
    public void dateDigitsDoNotCompleteATenNumberRow() {
        // 16, 10, 20 would complete the row if the date digits leaked
        String[] rows = {
                ISO + ",1,4,7,9,11,13,15,17,19,21",
                "16.10.2026;1;4;7;9;11;13;15;17;19;21",
                "16/10/2026 1 4 7 9 11 13 15 17 19 21",
                "Draw 2026-10-16: 1 4 7 9 11 13 15 17 19 21",
                "1 4 7 9 11 drawn on 16.10.2026 13 15 17 19 21",
                "20261016,1,4,7,9,11,13,15,17,19,21",
        };
        for (String row : rows) {
            ParseResult r = csv(row);
            assertNoDraw(r);
            assertEquals(row, 1, r.skippedLines);
            assertTrue(row + " -> " + r.warnings, hasWarningContaining(r, "10 numbers found, expected 11"));
        }
    }

    @Test
    public void dateDigitsAreNotAddedToElevenNumbers() {
        int[] nums = {1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13};
        ParseResult r = csv("16.10.2026 " + join(nums, " "));
        assertSingleDated(r, ISO, nums);
        r = csv("Draw of 2026-10-16 -> " + join(nums, " "));
        assertSingleDated(r, ISO, nums);
    }

    @Test
    public void timeCellsDoNotLeak() {
        assertSingleDated(csv(ISO + ",20:00," + join(N, ",")), ISO, N);
        assertSingleDated(csv(ISO + " 20:00 " + join(N, " ")), ISO, N);
        assertSingleDated(csv("2026-10-16 20:00," + join(N, ",")), ISO, N);
        assertSingleDated(csv("2026-10-16T20:00:00;" + join(N, ";")), ISO, N);
        // a ten-number row must not be completed by the hour
        assertNoDraw(csv("2026-10-16 20:00,1,4,7,9,11,13,15,17,19,21"));
        assertNoDraw(csv(ISO + ",20:00,1,4,7,9,11,13,15,17,19,21"));
    }

    // ================================================================== extra columns / run-of-11

    @Test
    public void idColumnBeforeDateUsesRunAfterDate() {
        // the id 17 is a valid number, giving 12 candidates: the run of 11 after the date is used
        assertSingleDated(csv("17," + ISO + "," + join(N, ",")), ISO, N);
        assertSingleDated(csv("3;16.10.2026;" + join(N, ";")), ISO, N);
    }

    @Test
    public void idColumnOutsideRangeIsIgnored() {
        assertSingleDated(csv("12345," + ISO + "," + join(N, ",")), ISO, N);
        assertSingleDated(csv(ISO + ",987," + join(N, ",")), ISO, N);
        assertSingleDated(csv("0," + ISO + "," + join(N, ",")), ISO, N);
    }

    @Test
    public void runAfterADescriptiveCell() {
        assertSingleDated(csv(ISO + ",Draw #5," + join(N, ",")), ISO, N);
    }

    @Test
    public void runImmediatelyAfterTheDateIsPreferred() {
        String line = "1,2,3,4,5,6,7,8,9,10,11," + ISO + ",12,13,14,15,16,17,18,19,20,21,22";
        assertSingleDated(csv(line), ISO, new int[] {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22});
    }

    @Test
    public void firstRunIsUsedWhenNoneFollowsTheDate() {
        String line = "1,2,3,4,5,6,7,8,9,10,11,x,12,13,14,15,16,17,18,19,20,21,22," + ISO;
        assertSingleDated(csv(line), ISO, new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    }

    @Test
    public void undatedRowWithIdColumn() {
        ParseResult r = csv("#12;x;" + join(N, ";"));
        assertEquals(r.warnings.toString(), 1, r.undated.size());
        assertArrayEquals(N, r.undated.get(0));
    }

    // ================================================================== wrong counts

    @Test
    public void tenNumbersGiveAWarning() {
        ParseResult r = csv("line one header\n" + ISO + ",1,4,7,9,11,13,15,17,19,21");
        assertNoDraw(r);
        assertTrue(r.warnings.toString(), r.warnings.contains("line 2: 10 numbers found, expected 11"));
    }

    @Test
    public void twelveNumbersGiveAWarning() {
        ParseResult r = csv(ISO + ",1,2,4,7,9,11,13,15,17,19,21,22");
        assertNoDraw(r);
        assertTrue(r.warnings.toString(), r.warnings.contains("line 1: 12 numbers found, expected 11"));
        assertEquals(1, r.skippedLines);
    }

    @Test
    public void thirteenNumbersGiveTheSpecWarning() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 11; i++) {
            sb.append("caption line without numbers\n");
        }
        // lines 1..11 are captions (skipped silently); line 12 has 13 numbers
        String content = sb.toString() + ISO + ",1,2,3,4,5,6,7,8,9,10,11,12,13";
        ParseResult r = csv(content);
        assertNoDraw(r);
        assertEquals(1, r.warnings.size());
        assertEquals("line 12: 13 numbers found, expected 11", r.warnings.get(0));
        assertEquals(12, r.skippedLines);
    }

    @Test
    public void twelveUndatedNumbersGiveAWarning() {
        ParseResult r = csv("1;2;3;4;5;6;7;8;9;10;11;12");
        assertNoDraw(r);
        assertTrue(hasWarningContaining(r, "12 numbers found, expected 11"));
    }

    @Test
    public void duplicateNumbersAreNotADraw() {
        ParseResult r = csv(ISO + ",1,1,4,7,9,11,13,15,17,19,21");
        assertNoDraw(r);
        assertEquals(1, r.skippedLines);
        assertEquals(1, r.warnings.size());
        assertTrue(r.warnings.get(0), r.warnings.get(0).startsWith("line 1:"));
    }

    @Test
    public void elevenDistinctPlusOneDuplicateIsADraw() {
        // spec step 3: "exactly 11 distinct valid numbers -> draw"
        assertSingleDated(csv(ISO + ",1,1,4,7,9,11,13,15,17,19,21,22"), ISO, N);
    }

    @Test
    public void numbersAbove22AreIgnored() {
        assertNoDraw(csv(ISO + ",1,4,7,9,11,13,15,17,19,21,23"));
        assertNoDraw(csv(ISO + ",1,4,7,9,11,13,15,17,19,21,99"));
        assertSingleDated(csv(ISO + ",1,4,7,9,11,13,15,17,19,21,22,23"), ISO, N);
        assertSingleDated(csv(ISO + ",100,1,4,7,9,11,13,15,17,19,21,22"), ISO, N);
    }

    @Test
    public void zeroAndDoubleZeroAreNotNumbers() {
        assertNoDraw(csv(ISO + ",00,4,7,9,11,13,15,17,19,21,22"));
        assertNoDraw(csv(ISO + ",0,4,7,9,11,13,15,17,19,21,22"));
        assertSingleDated(csv(ISO + ",00," + join(N, ",")), ISO, N);
    }

    @Test
    public void zeroPaddedNumbersAreAccepted() {
        assertSingleDated(csv(ISO + ",01,04,07,09,11,13,15,17,19,21,22"), ISO, N);
    }

    @Test
    public void negativeNumbersAreNotDrawNumbers() {
        // "-3" is minus three, not 3: the row has only 10 valid numbers
        ParseResult r = csv(ISO + ",1,4,7,9,11,13,15,17,19,21,-3");
        assertNoDraw(r);
        assertTrue(r.warnings.toString(), hasWarningContaining(r, "10 numbers found"));
        assertNoDraw(csv(ISO + ";-1;4;7;9;11;13;15;17;19;21;22"));
        // a negative extra value next to 11 valid numbers is ignored like any other invalid one
        assertSingleDated(csv(ISO + ",-5," + join(N, ",")), ISO, N);
    }

    @Test
    public void dateOnlyLineIsSkippedWithWarning() {
        ParseResult r = csv(ISO);
        assertNoDraw(r);
        assertEquals(1, r.skippedLines);
        assertEquals(1, r.warnings.size());
    }

    @Test
    public void invalidDateRowIsNotImported() {
        ParseResult r = csv("31.02.2026," + join(N, ","));
        assertNoDraw(r);
        assertEquals(1, r.skippedLines);
        assertFalse(r.warnings.isEmpty());
        assertTrue(r.warnings.get(0), r.warnings.get(0).startsWith("line 1:"));
    }

    @Test
    public void invalidDateDigitsDoNotLeak() {
        // "31.02.2026" is not a date; its digits 2 must not turn ten numbers into a draw
        ParseResult r = csv("31.02.2026,1,4,7,9,11,13,15,17,19,21");
        assertNoDraw(r);
    }

    // ================================================================== undated rows

    @Test
    public void undatedRowsKeepFileOrder() {
        String content = "1;2;3;4;5;6;7;8;9;10;11\n"
                + "12;13;14;15;16;17;18;19;20;21;22\n"
                + "2;4;6;8;10;12;14;16;18;20;22\n";
        ParseResult r = csv(content);
        assertTrue(r.dated.isEmpty());
        assertEquals(3, r.undated.size());
        assertArrayEquals(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, r.undated.get(0));
        assertArrayEquals(new int[] {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}, r.undated.get(1));
        assertArrayEquals(new int[] {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}, r.undated.get(2));
        assertFalse(r.isEmpty());
    }

    @Test
    public void undatedRowsAreSortedAndDistinct() {
        ParseResult r = csv("22 21 19 17 15 13 11 9 7 4 1");
        assertEquals(1, r.undated.size());
        assertArrayEquals(N, r.undated.get(0));
    }

    @Test
    public void mixedDatedAndUndatedRows() {
        String content = DrawCsv.HEADER + "\n"
                + "2026-10-15," + join(N, ",") + "\n"
                + join(N, ",") + "\n"
                + "garbage 1 2 3\n"
                + "\n"
                + "2026-10-16," + join(N, ",") + "\n";
        ParseResult r = csv(content);
        assertEquals(2, r.dated.size());
        assertEquals(1, r.undated.size());
        assertEquals("2026-10-15", r.dated.get(0).date);
        assertEquals(ISO, r.dated.get(1).date);
        assertEquals(5, r.totalLines);          // the blank line is not counted
        assertEquals(2, r.skippedLines);        // header + garbage
        assertEquals(1, r.warnings.size());     // only the garbage line has numbers
        assertTrue(r.warnings.get(0), r.warnings.get(0).startsWith("line 4:"));
    }

    // ================================================================== line endings, BOM, quoting

    @Test
    public void crlfLineEndings() {
        String content = DrawCsv.HEADER + "\r\n2026-10-15," + join(N, ",") + "\r\n2026-10-16,"
                + join(N, ",") + "\r\n";
        ParseResult r = csv(content);
        assertEquals(r.warnings.toString(), 2, r.dated.size());
        assertEquals(ISO, r.dated.get(1).date);
        assertArrayEquals(N, r.dated.get(1).numbers);
        assertTrue(r.warnings.isEmpty());
    }

    @Test
    public void mixedLineEndingsAndLineNumbers() {
        String content = "2026-10-14," + join(N, ",") + "\n"
                + "2026-10-15," + join(N, ",") + "\r\n"
                + ISO + ",1,2\r\n"
                + "2026-10-17," + join(N, ",");
        ParseResult r = csv(content);
        assertEquals(3, r.dated.size());
        assertTrue(r.warnings.toString(), r.warnings.contains("line 3: 2 numbers found, expected 11"));
    }

    @Test
    public void byteOrderMarkIsIgnored() {
        ParseResult r = csv("﻿" + DrawCsv.HEADER + "\n" + ISO + "," + join(N, ","));
        assertSingleDated(r, ISO, N);
        assertTrue(r.warnings.isEmpty());
        assertSingleDated(csv("﻿" + ISO + "," + join(N, ",")), ISO, N);
    }

    @Test
    public void doubleQuotedCells() {
        StringBuilder sb = new StringBuilder("\"" + ISO + "\"");
        for (int n : N) {
            sb.append(",\"").append(n).append('"');
        }
        assertSingleDated(csv(sb.toString()), ISO, N);
    }

    @Test
    public void singleQuotedCells() {
        StringBuilder sb = new StringBuilder("'16.10.2026'");
        for (int n : N) {
            sb.append(";'").append(n).append('\'');
        }
        assertSingleDated(csv(sb.toString()), ISO, N);
    }

    @Test
    public void quotedCellMayContainTheDelimiter() {
        assertSingleDated(csv("\"" + ISO + "\",\"" + join(N, ",") + "\""), ISO, N);
    }

    @Test
    public void unbalancedQuotesAreTreatedAsText() {
        assertSingleDated(csv("\"" + ISO + "," + join(N, ",")), ISO, N);
        assertSingleDated(csv(ISO + "," + join(N, ",") + "\""), ISO, N);
    }

    @Test
    public void escapedQuotesInsideAQuotedCell() {
        ParseResult r = csv("\"Draw \"\"A\"\"\"," + ISO + "," + join(N, ","));
        assertSingleDated(r, ISO, N);
    }

    @Test
    public void idDateAndAllNumbersInOneCell() {
        // 17 (id) + 11 numbers in one quoted cell: no run of single cells, but one cell holds a draw
        assertSingleDated(csv("17;" + ISO + ";\"" + join(N, " ") + "\""), ISO, N);
    }

    @Test
    public void dayFirstDateTimeCell() {
        assertSingleDated(csv("16.10.2026 20:00;" + join(N, ";")), ISO, N);
        assertSingleDated(csv("\"2026-10-16 20:00:00\"," + join(N, ",")), ISO, N);
    }

    @Test
    public void ownExportIsImportable() {
        List<Draw> list = new ArrayList<Draw>();
        Random rnd = new Random(2);
        String day = "2026-01-01";
        for (int i = 0; i < 50; i++) {
            list.add(new Draw(day, randomDraw(rnd)));
            day = DateUtil.addDays(day, 1);
        }
        ParseResult r = DrawParser.parse(DrawCsv.write(list), "draws-2026-10-16.csv", true);
        assertEquals(list, r.dated);
        assertTrue(r.undated.isEmpty());
        assertTrue(r.warnings.toString(), r.warnings.isEmpty());
        assertEquals(1, r.skippedLines);       // the header
    }

    @Test
    public void textFileNamesAreParsedAsCsv() {
        assertSingleDated(DrawParser.parse(ISO + "\t" + join(N, "\t"), "draws.TSV", true), ISO, N);
        assertSingleDated(DrawParser.parse(ISO + " " + join(N, " "), "draws.txt", true), ISO, N);
    }

    @Test
    public void trailingDelimitersAndEmptyCells() {
        assertSingleDated(csv(ISO + "," + join(N, ",") + ",,,"), ISO, N);
        assertSingleDated(csv(ISO + ";" + join(N, ";") + ";"), ISO, N);
        assertSingleDated(csv(ISO + ",," + join(N, ",,")), ISO, N);
        assertSingleDated(csv("," + ISO + "," + join(N, ",")), ISO, N);
    }

    // ================================================================== empty / garbage

    @Test
    public void emptyInput() {
        ParseResult r = csv("");
        assertTrue(r.isEmpty());
        assertEquals(0, r.totalLines);
        assertEquals(0, r.skippedLines);
        assertTrue(r.warnings.isEmpty());
        assertTrue(csv(null).isEmpty());
        assertTrue(DrawParser.parse(null, null, true).isEmpty());
        assertTrue(DrawParser.parse("", "draws.csv", true).isEmpty());
    }

    @Test
    public void blankLinesOnly() {
        ParseResult r = csv("\n\n   \r\n\t\n﻿\n");
        assertTrue(r.isEmpty());
        assertEquals(0, r.skippedLines);
        assertTrue(r.warnings.isEmpty());
    }

    @Test
    public void garbageInputNeverThrows() {
        String alphabet = "0123456789012345678901234567890123456789,;|\t -/.:+#\"'<>()abcXYZ\n\r﻿\u0000";
        Random rnd = new Random(12345);
        for (int round = 0; round < 20; round++) {
            StringBuilder sb = new StringBuilder();
            int len = 2000 + rnd.nextInt(5000);
            for (int i = 0; i < len; i++) {
                sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
            }
            ParseResult r = csv(sb.toString());
            assertNotNull(r);
            for (Draw d : r.dated) {
                assertTrue(Game.isValidDraw(d.numbers));
                assertTrue(DateUtil.isIso(d.date));
            }
            for (int[] u : r.undated) {
                assertTrue(Game.isValidDraw(u));
            }
            assertTrue(r.skippedLines <= r.totalLines);
            assertTrue(r.warnings.size() <= ParseResult.MAX_WARNINGS + 1);
        }
    }

    @Test
    public void binaryLikeInputNeverThrows() {
        StringBuilder sb = new StringBuilder();
        Random rnd = new Random(7);
        for (int i = 0; i < 20000; i++) {
            sb.append((char) rnd.nextInt(0x3000));
        }
        assertNotNull(csv(sb.toString()));
        assertNotNull(DrawParser.parse(sb.toString(), "x.txt", false));
    }

    @Test
    public void veryLongLineNeverThrows() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 200000; i++) {
            sb.append(i % 10);
            if (i % 3 == 0) {
                sb.append('-');
            }
        }
        ParseResult r = csv(sb.toString());
        assertTrue(r.isEmpty());
        assertEquals(1, r.skippedLines);
    }

    @Test
    public void warningsAreCapped() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 100; i++) {
            sb.append(ISO).append(",1,2,3\n");
        }
        ParseResult r = csv(sb.toString());
        assertEquals(100, r.skippedLines);
        assertEquals(ParseResult.MAX_WARNINGS + 1, r.warnings.size());
        String last = r.warnings.get(r.warnings.size() - 1);
        assertTrue(last, last.startsWith("... and "));
        assertTrue(last, last.contains("80"));
    }

    @Test(timeout = 20000)
    public void largeInputIsLinear() {
        int rows = 100000;
        StringBuilder sb = new StringBuilder(rows * 40);
        sb.append(DrawCsv.HEADER).append('\n');
        Random rnd = new Random(1);
        for (int i = 0; i < rows; i++) {
            // years are limited to 1900..2100, so the dates cycle (duplicates are fine here)
            sb.append(DateUtil.addDays("1950-01-01", i % 50000));
            for (int n : randomDraw(rnd)) {
                sb.append(',').append(n);
            }
            sb.append('\n');
        }
        ParseResult r = csv(sb.toString());
        assertEquals(rows, r.dated.size());
        assertTrue(r.warnings.isEmpty());
    }

    // ================================================================== parse() dispatch

    @Test
    public void parsePicksCsvForCsvContent() {
        ParseResult r = DrawParser.parse(ISO + "," + join(N, ","), "draws.csv", true);
        assertSingleDated(r, ISO, N);
        r = DrawParser.parse(ISO + "," + join(N, ","), null, true);
        assertSingleDated(r, ISO, N);
    }

    // ================================================================== extractNumbers

    @Test
    public void extractNumbersBasics() {
        assertArrayEquals(new int[] {1, 4, 7}, DrawParser.extractNumbers("1 4 7"));
        assertArrayEquals(new int[] {1, 4, 7, 9, 11}, DrawParser.extractNumbers("1-4-7/9+11"));
        assertArrayEquals(new int[] {1, 4, 7}, DrawParser.extractNumbers("1,4;7"));
        assertArrayEquals(new int[] {22, 1}, DrawParser.extractNumbers("22 1"));
    }

    @Test
    public void extractNumbersKeepsDuplicatesAndOrder() {
        assertArrayEquals(new int[] {5, 5, 3, 5}, DrawParser.extractNumbers("5 5 3 5"));
    }

    @Test
    public void extractNumbersIgnoresOutOfRangeValues() {
        assertArrayEquals(new int[] {5}, DrawParser.extractNumbers("0 23 100 2026 5 00"));
        assertArrayEquals(new int[] {7, 22}, DrawParser.extractNumbers("07 022"));
    }

    @Test
    public void extractNumbersIgnoresNegativeValues() {
        assertArrayEquals(new int[0], DrawParser.extractNumbers("-3"));
        assertArrayEquals(new int[] {4}, DrawParser.extractNumbers("-3 4"));
        assertArrayEquals(new int[] {4}, DrawParser.extractNumbers("4,-3"));
        assertArrayEquals(new int[] {1, 3}, DrawParser.extractNumbers("1-3"));
        assertArrayEquals(new int[] {1, 3}, DrawParser.extractNumbers("1 - 3"));
    }

    @Test
    public void extractNumbersIgnoresDigitsGluedToLetters() {
        assertArrayEquals(new int[] {3}, DrawParser.extractNumbers("n1 n2 3 1st x5"));
        assertArrayEquals(new int[0], DrawParser.extractNumbers(DrawCsv.HEADER));
    }

    @Test
    public void extractNumbersEmpty() {
        assertArrayEquals(new int[0], DrawParser.extractNumbers(null));
        assertArrayEquals(new int[0], DrawParser.extractNumbers(""));
        assertArrayEquals(new int[0], DrawParser.extractNumbers("no numbers"));
    }

    // ================================================================== asDraw

    @Test
    public void asDrawSortsElevenDistinctNumbers() {
        int[] in = {22, 1, 19, 4, 17, 7, 15, 9, 13, 11, 21};
        int[] copy = in.clone();
        assertArrayEquals(N, DrawParser.asDraw(in));
        assertArrayEquals("input must not be modified", copy, in);
    }

    @Test
    public void asDrawDedupesWhenElevenDistinctRemain() {
        int[] in = {1, 1, 4, 7, 9, 11, 13, 15, 17, 19, 21, 22, 22};
        assertArrayEquals(N, DrawParser.asDraw(in));
    }

    @Test
    public void asDrawRejectsWrongCounts() {
        assertNull(DrawParser.asDraw(null));
        assertNull(DrawParser.asDraw(new int[0]));
        assertNull(DrawParser.asDraw(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
        assertNull(DrawParser.asDraw(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10}));
        assertNull(DrawParser.asDraw(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
        assertNull(DrawParser.asDraw(new int[] {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
        assertNull(DrawParser.asDraw(new int[] {23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
        assertNull(DrawParser.asDraw(new int[] {-1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    }

    // ================================================================== assignDates

    private static List<int[]> threeRows() {
        List<int[]> rows = new ArrayList<int[]>();
        rows.add(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
        rows.add(new int[] {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22});
        rows.add(N.clone());
        return rows;
    }

    private static Map<String, int[]> byDate(List<Draw> draws) {
        Map<String, int[]> m = new HashMap<String, int[]>();
        for (Draw d : draws) {
            m.put(d.date, d.numbers);
        }
        return m;
    }

    @Test
    public void assignDatesOldestFirst() {
        List<int[]> rows = threeRows();
        List<Draw> out = DrawParser.assignDates(rows, ISO, true);
        assertEquals(3, out.size());
        Map<String, int[]> m = byDate(out);
        assertArrayEquals(rows.get(0), m.get("2026-10-14"));
        assertArrayEquals(rows.get(1), m.get("2026-10-15"));
        assertArrayEquals(rows.get(2), m.get("2026-10-16"));
    }

    @Test
    public void assignDatesNewestFirst() {
        List<int[]> rows = threeRows();
        List<Draw> out = DrawParser.assignDates(rows, ISO, false);
        assertEquals(3, out.size());
        Map<String, int[]> m = byDate(out);
        assertArrayEquals(rows.get(0), m.get("2026-10-16"));
        assertArrayEquals(rows.get(1), m.get("2026-10-15"));
        assertArrayEquals(rows.get(2), m.get("2026-10-14"));
    }

    @Test
    public void assignDatesAcrossMonthAndYearBoundaries() {
        Map<String, int[]> m = byDate(DrawParser.assignDates(threeRows(), "2026-03-01", true));
        assertNotNull(m.get("2026-02-27"));
        assertNotNull(m.get("2026-02-28"));
        assertNotNull(m.get("2026-03-01"));
        m = byDate(DrawParser.assignDates(threeRows(), "2027-01-01", true));
        assertNotNull(m.get("2026-12-30"));
        assertNotNull(m.get("2026-12-31"));
        assertNotNull(m.get("2027-01-01"));
    }

    @Test
    public void assignDatesSingleRowGetsTheEndDate() {
        List<int[]> rows = new ArrayList<int[]>();
        rows.add(N.clone());
        assertEquals(ISO, DrawParser.assignDates(rows, ISO, true).get(0).date);
        assertEquals(ISO, DrawParser.assignDates(rows, ISO, false).get(0).date);
    }

    @Test
    public void assignDatesEmpty() {
        assertTrue(DrawParser.assignDates(new ArrayList<int[]>(), ISO, true).isEmpty());
        assertTrue(DrawParser.assignDates(null, ISO, true).isEmpty());
    }

    @Test
    public void assignDatesFromParsedUndatedRows() {
        String content = "1;2;3;4;5;6;7;8;9;10;11\n12;13;14;15;16;17;18;19;20;21;22\n";
        ParseResult r = csv(content);
        List<Draw> out = DrawParser.assignDates(r.undated, ISO, true);
        DrawSet set = new DrawSet(out);
        assertEquals(2, set.size());
        assertEquals("2026-10-15", set.all().get(0).date);
        assertEquals(1, set.all().get(0).numbers[0]);
        assertEquals(ISO, set.latest().date);
        assertEquals(12, set.latest().numbers[0]);
    }

    // ================================================================== helpers

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
        int[] d = new int[11];
        System.arraycopy(pool, 0, d, 0, 11);
        java.util.Arrays.sort(d);
        return d;
    }
}
