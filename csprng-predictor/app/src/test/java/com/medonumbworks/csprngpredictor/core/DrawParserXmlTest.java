package com.medonumbworks.csprngpredictor.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

/**
 * DrawParser XML import (SPEC 1.4): canonical forms, wrappers, date rules (a)-(e), the
 * deepest-element rule, malformed input and parse() dispatch.
 */
public class DrawParserXmlTest {

    private static final int[] A = {1, 4, 7, 9, 11, 13, 15, 17, 19, 21, 22};
    private static final int[] B = {2, 5, 6, 8, 9, 12, 13, 17, 19, 20, 22};
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

    private static String children(int[] nums, String tag) {
        StringBuilder sb = new StringBuilder();
        for (int n : nums) {
            sb.append('<').append(tag).append('>').append(n).append("</").append(tag).append('>');
        }
        return sb.toString();
    }

    private static ParseResult xml(String content) {
        return DrawParser.parseXml(content, true);
    }

    private static void assertSingleDated(ParseResult r, String date, int[] nums) {
        assertEquals("dated " + r + " " + r.warnings, 1, r.dated.size());
        assertEquals("undated " + r, 0, r.undated.size());
        assertEquals(date, r.dated.get(0).date);
        assertArrayEquals(nums, r.dated.get(0).numbers);
    }

    // ================================================================== canonical forms

    @Test
    public void formOneChildElementsWithDateAttribute() {
        ParseResult r = xml("<draws><draw date=\"2026-10-16\">" + children(A, "n") + "</draw></draws>");
        assertSingleDated(r, ISO, A);
        assertTrue(r.warnings.toString(), r.warnings.isEmpty());
    }

    @Test
    public void formTwoNumbersAttribute() {
        ParseResult r = xml("<draws><draw date=\"2026-10-15\" numbers=\"" + join(B, " ") + "\"/></draws>");
        assertSingleDated(r, "2026-10-15", B);
    }

    @Test
    public void formThreeDateAndNumbersChildren() {
        ParseResult r = xml("<draws><draw><date>15.10.2026</date><numbers>" + join(B, ",")
                + "</numbers></draw></draws>");
        assertSingleDated(r, "2026-10-15", B);
    }

    @Test
    public void allThreeCanonicalFormsTogether() {
        String content = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                + "<draws>\n"
                + "  <draw date=\"2026-10-16\">" + children(A, "n") + "</draw>\n"
                + "  <draw date=\"2026-10-15\" numbers=\"" + join(B, " ") + "\"/>\n"
                + "  <draw>\n    <date>14.10.2026</date>\n    <numbers>" + join(A, ",")
                + "</numbers>\n  </draw>\n"
                + "</draws>\n";
        ParseResult r = xml(content);
        assertEquals(r.warnings.toString(), 3, r.dated.size());
        assertTrue(r.undated.isEmpty());
        assertTrue(r.warnings.toString(), r.warnings.isEmpty());
        assertEquals(ISO, r.dated.get(0).date);
        assertArrayEquals(A, r.dated.get(0).numbers);
        assertEquals("2026-10-15", r.dated.get(1).date);
        assertArrayEquals(B, r.dated.get(1).numbers);
        assertEquals("2026-10-14", r.dated.get(2).date);
        assertArrayEquals(A, r.dated.get(2).numbers);
        assertEquals(3, r.totalLines);
        assertEquals(0, r.skippedLines);
    }

    @Test
    public void numbersAsTextOfTheDrawElement() {
        assertSingleDated(xml("<draws><draw date=\"2026-10-16\">" + join(A, " ") + "</draw></draws>"), ISO, A);
        assertSingleDated(xml("<draw date=\"2026-10-16\">" + join(A, ";") + "</draw>"), ISO, A);
    }

    @Test
    public void numberChildrenWithDifferentNames() {
        StringBuilder sb = new StringBuilder("<draw date=\"2026-10-16\">");
        for (int i = 0; i < A.length; i++) {
            sb.append("<n").append(i + 1).append('>').append(A[i]).append("</n").append(i + 1).append('>');
        }
        sb.append("</draw>");
        assertSingleDated(xml(sb.toString()), ISO, A);
    }

    @Test
    public void unsortedNumbersAreSorted() {
        int[] shuffled = {22, 1, 19, 4, 17, 7, 15, 9, 13, 11, 21};
        assertSingleDated(xml("<draw date=\"2026-10-16\" numbers=\"" + join(shuffled, " ") + "\"/>"), ISO, A);
    }

    // ================================================================== wrappers

    @Test
    public void nestedWrappers() {
        String content = "<export><meta><source>test</source><version>3</version></meta>"
                + "<data><year value=\"2026\"><month value=\"10\"><list>"
                + "<draw date=\"2026-10-16\">" + children(A, "n") + "</draw>"
                + "<draw date=\"2026-10-17\" numbers=\"" + join(B, " ") + "\"/>"
                + "</list></month></year></data></export>";
        ParseResult r = xml(content);
        assertEquals(r.warnings.toString(), 2, r.dated.size());
        assertEquals(ISO, r.dated.get(0).date);
        assertArrayEquals(A, r.dated.get(0).numbers);
        assertEquals("2026-10-17", r.dated.get(1).date);
        assertArrayEquals(B, r.dated.get(1).numbers);
    }

    @Test
    public void manyDrawsInDocumentOrder() {
        StringBuilder sb = new StringBuilder("<draws>");
        String day = "2026-01-01";
        for (int i = 0; i < 50; i++) {
            int[] nums = i % 2 == 0 ? A : B;
            sb.append("<draw date=\"").append(day).append("\">").append(join(nums, " ")).append("</draw>");
            day = DateUtil.addDays(day, 1);
        }
        sb.append("</draws>");
        ParseResult r = xml(sb.toString());
        assertEquals(50, r.dated.size());
        day = "2026-01-01";
        for (int i = 0; i < 50; i++) {
            assertEquals(day, r.dated.get(i).date);
            assertArrayEquals(i % 2 == 0 ? A : B, r.dated.get(i).numbers);
            day = DateUtil.addDays(day, 1);
        }
    }

    // ================================================================== date rules

    @Test
    public void attributeDatesInDifferentFormats() {
        String[] attrs = {
                "date=\"2026-10-16\"", "date=\"16.10.2026\"", "date=\"2026/10/16\"", "date=\"20261016\"",
                "datum=\"16-10-2026\"", "day=\"10/16/2026\"", "drawdate=\"2026.10.16\"",
                "draw_date=\"16.10.26\"", "time=\"2026-10-16T20:00:00\"", "datetime=\"2026-10-16 20:00\"",
                "DATE=\"2026-10-16\"", "drawDate=\"16/10/2026\"",
        };
        for (String attr : attrs) {
            ParseResult r = xml("<draws><draw " + attr + " numbers=\"" + join(A, " ") + "\"/></draws>");
            assertEquals(attr + " " + r.warnings, 1, r.dated.size());
            assertEquals(attr, ISO, r.dated.get(0).date);
            assertArrayEquals(attr, A, r.dated.get(0).numbers);
        }
    }

    @Test
    public void anyAttributeValueThatIsADate() {
        // rule (b): attribute not named like a date, but its value is one
        ParseResult r = xml("<draw stamp=\"16.10.2026\" numbers=\"" + join(A, " ") + "\"/>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void dateAttributeDigitsDoNotLeak() {
        // ten numbers + a date whose digits (16, 10, 20) could complete the draw
        ParseResult r = xml("<draw date=\"2026-10-16\" numbers=\"1 4 7 9 11 13 15 17 19 21\"/>");
        assertTrue(r.isEmpty());
        assertFalse(r.warnings.isEmpty());
        r = xml("<draw stamp=\"16.10.2026\">1 4 7 9 11 13 15 17 19 21</draw>");
        assertTrue(r.isEmpty());
    }

    @Test
    public void dateChildElementVariants() {
        String[] dates = {"<date>2026-10-16</date>", "<datum>16.10.2026</datum>", "<day>16/10/2026</day>",
                "<drawDate>2026-10-16</drawDate>", "<draw_date>20261016</draw_date>",
                "<datetime>2026-10-16T20:00:00</datetime>", "<time>2026-10-16 20:00</time>"};
        for (String d : dates) {
            ParseResult r = xml("<draws><draw>" + d + "<numbers>" + join(A, " ") + "</numbers></draw></draws>");
            assertEquals(d + " " + r.warnings, 1, r.dated.size());
            assertEquals(d, ISO, r.dated.get(0).date);
            assertArrayEquals(d, A, r.dated.get(0).numbers);
        }
    }

    @Test
    public void dateChildNextToNumberChildren() {
        ParseResult r = xml("<draw><date>2026-10-16</date>" + children(A, "n") + "</draw>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void dateChildAfterNumbersAttribute() {
        ParseResult r = xml("<draw numbers=\"" + join(A, " ") + "\"><date>16.10.2026</date></draw>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void dateInsideOwnText() {
        // rule (d): findDate over the element's own text; its digits are not numbers
        ParseResult r = xml("<draws><draw>Draw of 16.10.2026: " + join(A, " ") + "</draw></draws>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void dateWithTimeInsideOwnText() {
        ParseResult r = xml("<draw>2026-10-16 20:00 -> " + join(A, " ") + "</draw>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void dateOnAnAncestorAttribute() {
        ParseResult r = xml("<draws><day date=\"2026-10-16\"><draw>" + join(A, " ") + "</draw></day></draws>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void dateOnAnAncestorChild() {
        ParseResult r = xml("<draws><week><date>16.10.2026</date><draw>" + join(A, " ")
                + "</draw></week></draws>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void nearestAncestorDateWins() {
        String content = "<draws date=\"2000-01-01\">"
                + "<day date=\"2026-10-16\"><result><draw>" + join(A, " ") + "</draw></result></day>"
                + "<day date=\"2026-10-17\"><result><draw>" + join(B, " ") + "</draw></result></day>"
                + "<result><draw>" + join(A, " ") + "</draw></result>"
                + "</draws>";
        ParseResult r = xml(content);
        assertEquals(r.warnings.toString(), 3, r.dated.size());
        assertEquals(ISO, r.dated.get(0).date);
        assertArrayEquals(A, r.dated.get(0).numbers);
        assertEquals("2026-10-17", r.dated.get(1).date);
        assertArrayEquals(B, r.dated.get(1).numbers);
        assertEquals("2000-01-01", r.dated.get(2).date);
    }

    @Test
    public void ownDateBeatsAncestorDate() {
        ParseResult r = xml("<day date=\"2026-01-01\"><draw date=\"2026-10-16\">" + join(A, " ")
                + "</draw></day>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void dayFirstFlagForAmbiguousAttributeDates() {
        String content = "<draw date=\"05/06/2026\" numbers=\"" + join(A, " ") + "\"/>";
        assertEquals("2026-06-05", DrawParser.parseXml(content, true).dated.get(0).date);
        assertEquals("2026-05-06", DrawParser.parseXml(content, false).dated.get(0).date);
    }

    @Test
    public void undatedXmlDraws() {
        ParseResult r = xml("<draws><draw>" + join(A, " ") + "</draw><draw numbers=\"" + join(B, ",")
                + "\"/></draws>");
        assertTrue(r.dated.isEmpty());
        assertEquals(2, r.undated.size());
        assertArrayEquals(A, r.undated.get(0));
        assertArrayEquals(B, r.undated.get(1));
    }

    @Test
    public void invalidAttributeDateIsNotImported() {
        ParseResult r = xml("<draws><draw date=\"31.02.2026\" numbers=\"" + join(A, " ") + "\"/></draws>");
        assertTrue(r.dated.isEmpty());
        assertTrue(r.undated.isEmpty());
        assertFalse(r.warnings.isEmpty());
    }

    // ================================================================== deepest-element rule

    @Test
    public void deepestElementWins() {
        // <week>'s own text (leaf children) would also be a draw; only the deepest counts
        ParseResult r = xml("<draws><week date=\"2026-10-16\"><draw>" + join(A, " ")
                + "</draw></week></draws>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void deepNumbersElementInsideDraw() {
        ParseResult r = xml("<draws><draw date=\"2026-10-16\"><result><numbers>" + join(A, " ")
                + "</numbers></result></draw></draws>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void deeperDrawSuppressesTheOuterElement() {
        // the outer <draw> has 11 numbers of its own children, but a descendant is a draw too:
        // only the deepest one is imported
        String content = "<draw date=\"2026-10-16\">" + children(A, "n")
                + "<inner><numbers>" + join(B, " ") + "</numbers></inner></draw>";
        ParseResult r = xml(content);
        assertSingleDated(r, ISO, B);
    }

    @Test
    public void siblingsAreIndependentDraws() {
        String content = "<draws date=\"2026-10-16\"><a>" + join(A, " ") + "</a><b>" + join(B, " ")
                + "</b></draws>";
        ParseResult r = xml(content);
        assertEquals(2, r.dated.size());
        assertArrayEquals(A, r.dated.get(0).numbers);
        assertArrayEquals(B, r.dated.get(1).numbers);
        assertEquals(ISO, r.dated.get(0).date);
        assertEquals(ISO, r.dated.get(1).date);
        assertTrue(r.warnings.toString(), r.warnings.isEmpty());
    }

    // ================================================================== extra content

    @Test
    public void idAttributeNextToNumbers() {
        assertSingleDated(xml("<draw id=\"5\" date=\"2026-10-16\" numbers=\"" + join(A, " ") + "\"/>"), ISO, A);
        assertSingleDated(xml("<draw id=\"3\" date=\"2026-10-16\">" + children(A, "n") + "</draw>"), ISO, A);
    }

    @Test
    public void idChildNextToNumberChildren() {
        assertSingleDated(xml("<draw date=\"2026-10-16\"><id>3</id>" + children(A, "n") + "</draw>"), ISO, A);
    }

    @Test
    public void bonusChildNextToNumberChildren() {
        String content = "<draw date=\"2026-10-16\">" + children(A, "n") + "<bonus>5</bonus></draw>";
        assertSingleDated(xml(content), ISO, A);
    }

    @Test
    public void uppercaseElementAndAttributeNames() {
        String content = "<DRAWS><Draw Date=\"16.10.2026\"><Numbers>" + join(A, " ") + "</Numbers></Draw></DRAWS>";
        assertSingleDated(xml(content), ISO, A);
    }

    @Test
    public void wrongNumberCountGivesAWarning() {
        ParseResult r = xml("<draws><draw date=\"2026-10-16\">1 4 7 9 11 13 15 17 19 21</draw>"
                + "<draw date=\"2026-10-17\">" + join(B, " ") + "</draw></draws>");
        assertEquals(1, r.dated.size());
        assertEquals("2026-10-17", r.dated.get(0).date);
        assertEquals(1, r.skippedLines);
        assertEquals(2, r.totalLines);
        boolean found = false;
        for (String w : r.warnings) {
            if (w.contains("10 numbers found")) {
                found = true;
            }
        }
        assertTrue(r.warnings.toString(), found);
    }

    @Test
    public void twelveNumbersGiveAWarning() {
        ParseResult r = xml("<draws><draw>1 2 4 7 9 11 13 15 17 19 21 22</draw></draws>");
        assertTrue(r.isEmpty());
        assertFalse(r.warnings.isEmpty());
    }

    @Test
    public void numbersOutOfRangeAreIgnored() {
        ParseResult r = xml("<draw date=\"2026-10-16\">0 23 99 " + join(A, " ") + "</draw>");
        assertSingleDated(r, ISO, A);
    }

    // ================================================================== namespaces, CDATA, comments

    @Test
    public void namespacePrefixes() {
        String content = "<ns:draws xmlns:ns=\"urn:example:draws\">"
                + "<ns:draw ns:date=\"2026-10-16\">" + children(A, "ns:n") + "</ns:draw>"
                + "</ns:draws>";
        assertSingleDated(xml(content), ISO, A);
    }

    @Test
    public void defaultNamespaceWithDigitsInTheUri() {
        String content = "<draws xmlns=\"http://example.com/schema/2026/v2\">"
                + "<draw date=\"2026-10-16\" numbers=\"" + join(A, " ") + "\"/></draws>";
        assertSingleDated(xml(content), ISO, A);
    }

    @Test
    public void cdataSection() {
        ParseResult r = xml("<draws><draw date=\"2026-10-16\"><![CDATA[" + join(A, " ") + "]]></draw></draws>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void cdataSplitAcrossTextNodes() {
        ParseResult r = xml("<draw date=\"2026-10-16\">1 4 7 <![CDATA[9 11 13]]> 15 17 19 21 22</draw>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void commentsAreIgnored() {
        // without ignoring the comment there would be 13 candidates
        ParseResult r = xml("<!-- header 1 2 --><draws><draw date=\"2026-10-16\"><!-- 3 5 -->"
                + join(A, " ") + "</draw></draws>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void processingInstructionsAndEntities() {
        ParseResult r = xml("<?xml version=\"1.0\"?><?app hint=\"1 2 3\"?><draws><draw date=\"2026-10-16\" "
                + "note=\"a &amp; b\">" + join(A, " ") + "</draw></draws>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void byteOrderMarkAndLeadingWhitespace() {
        ParseResult r = xml("﻿  \n<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<draws><draw date=\"2026-10-16\">"
                + join(A, " ") + "</draw></draws>");
        assertSingleDated(r, ISO, A);
    }

    @Test
    public void crlfAndIndentation() {
        String content = "<draws>\r\n\t<draw date=\"2026-10-16\">\r\n\t\t<n>1</n>\r\n\t\t<n>4</n>\r\n"
                + "\t\t<n>7</n><n>9</n><n>11</n><n>13</n><n>15</n><n>17</n><n>19</n><n>21</n><n>22</n>\r\n"
                + "\t</draw>\r\n</draws>\r\n";
        assertSingleDated(xml(content), ISO, A);
    }

    // ================================================================== malformed / hostile

    @Test
    public void malformedXmlGivesWarningNoException() {
        String[] bad = {
                "<draws><draw date=\"2026-10-16\">1 4 7",
                "<a></b>",
                "<",
                "<<<>>>",
                "<?xml version=\"1.0\"?>",
                "<draws><draw date=\"2026-10-16\" numbers=\"1 2\"></draws>",
                "<draws attr=unquoted>1</draws>",
                "<draws>&undefined;</draws>",
                "<draws></draws><extra/>",
        };
        for (String content : bad) {
            ParseResult r = xml(content);
            assertNotNull(content, r);
            assertTrue(content, r.isEmpty());
            assertFalse(content, r.warnings.isEmpty());
        }
    }

    @Test
    public void emptyOrBlankXml() {
        assertTrue(xml("").isEmpty());
        assertFalse(xml("").warnings.isEmpty());
        assertTrue(xml(null).isEmpty());
        assertTrue(xml("   \n ").isEmpty());
    }

    @Test
    public void validXmlWithoutDrawsGivesAWarning() {
        ParseResult r = xml("<draws/>");
        assertTrue(r.isEmpty());
        assertFalse(r.warnings.isEmpty());
        r = xml("<html><body><p>Hello 1 2 3</p></body></html>");
        assertTrue(r.isEmpty());
        assertFalse(r.warnings.isEmpty());
    }

    @Test(timeout = 10000)
    public void externalEntitiesAreNeverResolved() {
        String content = "<?xml version=\"1.0\"?>\n"
                + "<!DOCTYPE draws [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]>\n"
                + "<draws><draw date=\"2026-10-16\">&xxe; " + join(A, " ") + "</draw></draws>";
        ParseResult r = xml(content);
        assertNotNull(r);
        for (Draw d : r.dated) {
            assertArrayEquals(A, d.numbers);
        }
        assertTrue(r.undated.isEmpty());
    }

    @Test(timeout = 10000)
    public void entityExpansionBombDoesNotHang() {
        String content = "<?xml version=\"1.0\"?>\n<!DOCTYPE lolz [\n"
                + " <!ENTITY lol \"lol\">\n"
                + " <!ENTITY lol2 \"&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;\">\n"
                + " <!ENTITY lol3 \"&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;\">\n"
                + " <!ENTITY lol4 \"&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;\">\n"
                + " <!ENTITY lol5 \"&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;\">\n"
                + " <!ENTITY lol6 \"&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;\">\n"
                + " <!ENTITY lol7 \"&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;\">\n"
                + " <!ENTITY lol8 \"&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;\">\n"
                + " <!ENTITY lol9 \"&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;\">\n"
                + "]>\n<lolz>&lol9;</lolz>";
        ParseResult r = xml(content);
        assertNotNull(r);
        assertTrue(r.isEmpty());
    }

    @Test
    public void randomGarbageStartingWithAngleBracketNeverThrows() {
        String alphabet = "<>/=\"' abc123456789:;&#![]-?\n";
        Random rnd = new Random(99);
        for (int round = 0; round < 200; round++) {
            StringBuilder sb = new StringBuilder("<");
            int len = 10 + rnd.nextInt(400);
            for (int i = 0; i < len; i++) {
                sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
            }
            ParseResult r = xml(sb.toString());
            assertNotNull(r);
            ParseResult p = DrawParser.parse(sb.toString(), "garbage.xml", true);
            assertNotNull(p);
        }
    }

    @Test
    public void deeplyNestedXmlNeverThrows() {
        StringBuilder sb = new StringBuilder();
        int depth = 3000;
        for (int i = 0; i < depth; i++) {
            sb.append("<e>");
        }
        sb.append(join(A, " "));
        for (int i = 0; i < depth; i++) {
            sb.append("</e>");
        }
        ParseResult r = xml(sb.toString());
        assertNotNull(r);
        if (!r.isEmpty()) {
            assertEquals(1, r.undated.size());
            assertArrayEquals(A, r.undated.get(0));
        }
    }

    @Test(timeout = 30000)
    public void largeXmlParses() {
        int count = 20000;
        StringBuilder sb = new StringBuilder(count * 80);
        sb.append("<draws>");
        String day = "1970-01-01";
        for (int i = 0; i < count; i++) {
            sb.append("<draw date=\"").append(day).append("\" numbers=\"")
                    .append(join(i % 2 == 0 ? A : B, " ")).append("\"/>");
            day = DateUtil.addDays(day, 1);
        }
        sb.append("</draws>");
        ParseResult r = xml(sb.toString());
        assertEquals(count, r.dated.size());
    }

    // ================================================================== parse() dispatch

    @Test
    public void parsePicksXmlByFileName() {
        String content = "<draws><draw date=\"2026-10-16\">" + join(A, " ") + "</draw></draws>";
        assertSingleDated(DrawParser.parse(content, "export.xml", true), ISO, A);
        assertSingleDated(DrawParser.parse(content, "EXPORT.XML", true), ISO, A);
    }

    @Test
    public void parsePicksXmlByContent() {
        String content = "  \n<draws><draw date=\"2026-10-16\">" + join(A, " ") + "</draw></draws>";
        assertSingleDated(DrawParser.parse(content, "draws.csv", true), ISO, A);
        assertSingleDated(DrawParser.parse(content, null, true), ISO, A);
        assertSingleDated(DrawParser.parse("﻿" + content, "draws.txt", true), ISO, A);
    }

    @Test
    public void parseXmlByNameWithCsvContentStillFindsTheDraws() {
        ParseResult r = DrawParser.parse(ISO + "," + join(A, ","), "draws.xml", true);
        assertSingleDated(r, ISO, A);
        assertFalse(r.warnings.isEmpty());
    }

    @Test
    public void parseMalformedXmlByContentDoesNotThrow() {
        ParseResult r = DrawParser.parse("<draws><draw>1 2 3", "draws.csv", true);
        assertTrue(r.isEmpty());
        assertFalse(r.warnings.isEmpty());
    }

    @Test
    public void parseUsesTheDayFirstFlag() {
        String content = "<draw date=\"05.06.2026\">" + join(A, " ") + "</draw>";
        assertEquals("2026-06-05", DrawParser.parse(content, "a.xml", true).dated.get(0).date);
        assertEquals("2026-05-06", DrawParser.parse(content, "a.xml", false).dated.get(0).date);
    }
}
