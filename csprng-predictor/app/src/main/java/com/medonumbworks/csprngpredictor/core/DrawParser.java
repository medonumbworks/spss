package com.medonumbworks.csprngpredictor.core;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.xml.sax.EntityResolver;
import org.xml.sax.ErrorHandler;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;

/**
 * Tolerant importer for draw files: CSV / TSV / TXT (any common delimiter, with or without
 * dates) and XML (any element structure that holds 11 numbers per draw).
 *
 * <p>All methods are linear in the input size and never throw on bad input; problems are
 * reported as warnings in the {@link ParseResult}.
 */
public final class DrawParser {

    /** A bare time cell such as "20:00", "20:00:00" or "8:15 pm". */
    private static final Pattern TIME_CELL = Pattern.compile(
            "^[0-9]{1,2}:[0-9]{2}(?:[:][0-9]{2}(?:[.,][0-9]+)?)?\\s*(?:[aApP]\\.?[mM]\\.?|h)?$");
    /** A time inside free text. */
    private static final Pattern TIME_TOKEN = Pattern.compile(
            "(?<![0-9:])[0-9]{1,2}:[0-9]{2}(?:[:][0-9]{2}(?:[.,][0-9]+)?)?(?![0-9])");

    private DrawParser() {
    }

    // =================================================================== public API

    /**
     * Picks XML when the file name ends with ".xml" or the trimmed content starts with '&lt;',
     * otherwise CSV. A ".xml" file whose content is plainly not XML is read as CSV instead
     * (with a note in the warnings).
     */
    public static ParseResult parse(String content, String fileNameHint, boolean dayFirst) {
        String c = content == null ? "" : content;
        boolean byName = fileNameHint != null
                && fileNameHint.trim().toLowerCase(Locale.US).endsWith(".xml");
        int first = firstNonBlank(c);
        boolean byContent = first >= 0 && c.charAt(first) == '<';
        if (byName || byContent) {
            ParseResult xml = parseXml(c, dayFirst);
            if (xml.isEmpty() && !byContent && first >= 0) {
                ParseResult csv = parseCsv(c, dayFirst);
                if (!csv.isEmpty()) {
                    csv.addWarningFirst("The file name ends with .xml but the content is not XML;"
                            + " it was read as CSV / text.");
                    return csv;
                }
            }
            return xml;
        }
        return parseCsv(c, dayFirst);
    }

    /** CSV / TSV / TXT: one draw per line, delimiter detected per line. */
    public static ParseResult parseCsv(String content, boolean dayFirst) {
        ParseResult r = new ParseResult();
        if (content == null) {
            return r;
        }
        try {
            csvInto(content, dayFirst, r);
        } catch (RuntimeException e) {
            r.addWarning("Unexpected problem while reading the file: " + describe(e));
        } catch (StackOverflowError e) {
            r.addWarning("Unexpected problem while reading the file: input too complex.");
        }
        return r;
    }

    /** XML: every deepest element that holds exactly 11 distinct numbers 1..22 is a draw. */
    public static ParseResult parseXml(String content, boolean dayFirst) {
        ParseResult r = new ParseResult();
        try {
            xmlInto(content, dayFirst, r);
        } catch (RuntimeException e) {
            r.addWarning("XML could not be read: " + describe(e));
        } catch (StackOverflowError e) {
            r.addWarning("XML could not be read: elements are nested too deeply.");
        }
        return r;
    }

    /**
     * All integers 1..22 found in {@code text}, in order, duplicates kept. Digit runs glued to
     * letters ("n1", "11th", "x5"), negative values ("-3"; a dash between two numbers as in
     * "1-3" is a separator) and values outside 1..22 (ids, years) are ignored.
     */
    public static int[] extractNumbers(String text) {
        IntList out = new IntList(16);
        appendNumbers(text, out);
        return out.toArray();
    }

    /** Sorted distinct copy if {@code candidates} holds exactly 11 distinct valid numbers, else null. */
    public static int[] asDraw(int[] candidates) {
        if (candidates == null) {
            return null;
        }
        boolean[] seen = new boolean[Game.MAX_NUMBER + 1];
        int distinct = 0;
        for (int v : candidates) {
            if (Game.isValidNumber(v) && !seen[v]) {
                seen[v] = true;
                distinct++;
            }
        }
        if (distinct != Game.PICK) {
            return null;
        }
        int[] out = new int[Game.PICK];
        int k = 0;
        for (int n = 1; n <= Game.MAX_NUMBER; n++) {
            if (seen[n]) {
                out[k++] = n;
            }
        }
        return out;
    }

    /**
     * Gives undated rows consecutive daily dates. {@code oldestFirst=true}: the first row is
     * the oldest; the chronologically last row gets {@code endDateIso}, earlier ones
     * {@code endDate - k} days. The returned list keeps the input (file) order; rows that are
     * not valid draws are skipped (the other rows keep their positional dates). An invalid
     * end date falls back to today.
     */
    public static List<Draw> assignDates(List<int[]> undated, String endDateIso, boolean oldestFirst) {
        List<Draw> out = new ArrayList<Draw>();
        if (undated == null || undated.isEmpty()) {
            return out;
        }
        String end = endDateIso == null ? "" : endDateIso.trim();
        if (!DateUtil.isIso(end)) {
            String n = DateUtil.normalize(end, true);
            end = n != null ? n : DateUtil.today();
        }
        int n = undated.size();
        for (int k = 0; k < n; k++) {
            int[] row = undated.get(k);
            int[] nums = Game.isValidDraw(row) ? row : asDraw(row);
            if (nums == null) {
                continue;
            }
            int back = oldestFirst ? (n - 1 - k) : k;
            out.add(new Draw(DateUtil.addDays(end, -back), nums));
        }
        return out;
    }

    // =================================================================== CSV

    private static void csvInto(String content, boolean dayFirst, ParseResult r) {
        int len = content.length();
        int start = 0;
        int lineNo = 0;
        for (int i = 0; i <= len; i++) {
            char ch = i < len ? content.charAt(i) : '\n';
            if (ch == '\n' || ch == '\r') {
                lineNo++;
                if (i > start) {
                    processCsvLine(content.substring(start, i), lineNo, r, dayFirst);
                }
                if (ch == '\r' && i + 1 < len && content.charAt(i + 1) == '\n') {
                    i++;
                }
                start = i + 1;
            }
        }
    }

    private static void processCsvLine(String rawLine, int lineNo, ParseResult r, boolean dayFirst) {
        String line = rawLine;
        if (line.indexOf('\u0000') >= 0) {
            line = line.replace("\u0000", "");          // UTF-16 read as UTF-8
        }
        line = line.replace('\uFEFF', ' ');              // BOMs (also of concatenated files)
        if (firstNonBlank(line) < 0) {
            return;
        }
        r.totalLines++;

        // 1. the date: first cell that is a date, else the first date-like substring
        List<String> cells = splitCells(line);
        boolean[] excluded = new boolean[cells.size()];
        String date = null;
        int dateCell = -1;
        String badDate = null;
        for (int i = 0; i < cells.size(); i++) {
            String cell = cells.get(i);
            String iso = DateUtil.normalize(cell, dayFirst);
            if (iso != null) {
                excluded[i] = true;                      // date digits are never numbers
                if (date == null) {
                    date = iso;
                    dateCell = i;
                }
            } else if (isTimeLike(cell)) {
                excluded[i] = true;
            } else if (DateUtil.looksLikeDate(cell)) {
                excluded[i] = true;                      // e.g. 31.02.2026: not a date, not numbers
                if (badDate == null) {
                    badDate = cell;
                }
            }
        }
        if (date == null) {
            DateUtil.DateMatch dm = DateUtil.findDateMatch(line, dayFirst);
            if (dm != null) {
                date = dm.iso;
                String rest = line.substring(0, dm.start) + " " + line.substring(dm.end);
                cells = splitCells(rest);
                excluded = new boolean[cells.size()];
                for (int i = 0; i < cells.size(); i++) {
                    String cell = cells.get(i);
                    excluded[i] = DateUtil.normalize(cell, dayFirst) != null || isTimeLike(cell)
                            || DateUtil.looksLikeDate(cell);
                }
            }
        }
        if (date == null && badDate != null) {
            // never import a row with a broken date as "undated" (it would get a made-up date)
            r.skippedLines++;
            r.addWarning("line " + lineNo + ": invalid date \"" + abbreviate(badDate) + "\"");
            return;
        }
        if (date == null && looksLikeHeader(cells, line)) {
            r.skippedLines++;                            // header / caption: skipped silently
            return;
        }

        // 2. candidate numbers from every non-date cell
        IntList cand = new IntList(16);
        for (int i = 0; i < cells.size(); i++) {
            if (!excluded[i]) {
                appendNumbers(cells.get(i), cand);
            }
        }
        int[] c = cand.toArray();

        // 3. exactly 11 -> draw; more -> look for a clean run of 11 single-number cells
        int[] draw = null;
        if (c.length == Game.PICK) {
            draw = asDraw(c);
        } else if (c.length > Game.PICK) {
            draw = findRun(cells, excluded, dateCell);
            if (draw == null) {
                draw = singleCellDraw(cells, excluded);
            }
            if (draw == null && c.length == Game.PICK + 1) {
                draw = asDraw(c);                        // 11 distinct + one stray duplicate (e.g. an id)
            }
        }
        if (draw == null) {
            r.skippedLines++;
            if (c.length > 0 || date != null) {
                r.addWarning("line " + lineNo + ": " + problem(c));
            }
            return;
        }

        // 4. dated / undated
        if (date != null) {
            r.dated.add(new Draw(date, draw));
        } else {
            r.undated.add(draw);
        }
    }

    private static String problem(int[] c) {
        return problem(c.length, distinctCount(c));
    }

    private static String problem(int count, int distinct) {
        if (count == Game.PICK) {
            return "duplicate numbers (" + distinct + " different), expected 11 different numbers";
        }
        return count + (count == 1 ? " number" : " numbers") + " found, expected 11";
    }

    private static String abbreviate(String s) {
        return s.length() > 40 ? s.substring(0, 40) + "..." : s;
    }

    /**
     * Splits a line into trimmed, non-empty cells. The delimiter is the first of , ; TAB |
     * that occurs outside double quotes, else runs of whitespace. Double quotes group text
     * (RFC 4180 style) when they are balanced; unbalanced quotes are treated as plain text.
     */
    static List<String> splitCells(String line) {
        int len = line.length();
        int quotes = 0;
        for (int i = 0; i < len; i++) {
            if (line.charAt(i) == '"') {
                quotes++;
            }
        }
        boolean quoteAware = quotes > 0 && quotes % 2 == 0;
        boolean inQ = false;
        boolean comma = false;
        boolean semi = false;
        boolean tab = false;
        boolean pipe = false;
        for (int i = 0; i < len; i++) {
            char ch = line.charAt(i);
            if (ch == '"' && quoteAware) {
                inQ = !inQ;
            } else if (!inQ) {
                if (ch == ',') {
                    comma = true;
                } else if (ch == ';') {
                    semi = true;
                } else if (ch == '\t') {
                    tab = true;
                } else if (ch == '|') {
                    pipe = true;
                }
            }
        }
        char delim = comma ? ',' : (semi ? ';' : (tab ? '\t' : (pipe ? '|' : 0)));
        List<String> cells = new ArrayList<String>(16);
        StringBuilder cur = new StringBuilder();
        inQ = false;
        for (int i = 0; i < len; i++) {
            char ch = line.charAt(i);
            if (ch == '"' && quoteAware) {
                if (inQ && i + 1 < len && line.charAt(i + 1) == '"') {
                    cur.append('"');
                    i++;
                } else {
                    inQ = !inQ;
                }
                continue;
            }
            boolean isDelim = !inQ && (delim == 0 ? Character.isWhitespace(ch) : ch == delim);
            if (isDelim) {
                addCell(cells, cur);
                cur.setLength(0);
            } else {
                cur.append(ch);
            }
        }
        addCell(cells, cur);
        return cells;
    }

    private static void addCell(List<String> cells, StringBuilder cur) {
        int a = 0;
        int b = cur.length();
        while (a < b && (Character.isWhitespace(cur.charAt(a)) || cur.charAt(a) == '"')) {
            a++;
        }
        while (b > a && (Character.isWhitespace(cur.charAt(b - 1)) || cur.charAt(b - 1) == '"')) {
            b--;
        }
        if (b > a) {
            cells.add(cur.substring(a, b));
        }
    }

    /** A line without any date, with letters, and without a single purely numeric cell. */
    private static boolean looksLikeHeader(List<String> cells, String line) {
        boolean letter = false;
        for (int i = 0, len = line.length(); i < len; i++) {
            if (Character.isLetter(line.charAt(i))) {
                letter = true;
                break;
            }
        }
        if (!letter) {
            return false;
        }
        for (String cell : cells) {
            if (isNumericCell(cell)) {
                return false;
            }
        }
        return true;
    }

    /** Digits plus only separators / whitespace, e.g. "7", "1 4 7", "1-4-7", "#5". */
    private static boolean isNumericCell(String cell) {
        boolean digit = false;
        for (int i = 0, len = cell.length(); i < len; i++) {
            char ch = cell.charAt(i);
            if (ch >= '0' && ch <= '9') {
                digit = true;
            } else if (!Character.isWhitespace(ch) && "-/+,.;:#()".indexOf(ch) < 0) {
                return false;
            }
        }
        return digit;
    }

    private static boolean isTimeLike(String s) {
        if (s == null) {
            return false;
        }
        String t = s.trim();
        return t.length() >= 4 && t.length() <= 16 && t.indexOf(':') > 0 && TIME_CELL.matcher(t).matches();
    }

    /** Value 1..22 of a cell that is just one number ("7", "07"), else -1. */
    private static int singleValue(String cell) {
        int len = cell.length();
        if (len < 1 || len > 3) {
            return -1;
        }
        int v = 0;
        for (int i = 0; i < len; i++) {
            char ch = cell.charAt(i);
            if (ch < '0' || ch > '9') {
                return -1;
            }
            v = v * 10 + (ch - '0');
        }
        return Game.isValidNumber(v) ? v : -1;
    }

    /**
     * A maximal run of exactly 11 consecutive single-number cells (date / time cells break
     * runs). Prefers the run that starts right after the date cell, else the first one.
     */
    private static int[] findRun(List<String> cells, boolean[] excluded, int dateCell) {
        int size = cells.size();
        int[] vals = new int[size];
        for (int i = 0; i < size; i++) {
            vals[i] = excluded[i] ? -1 : singleValue(cells.get(i));
        }
        int[] first = null;
        int i = 0;
        while (i < size) {
            if (vals[i] > 0) {
                int j = i;
                while (j < size && vals[j] > 0) {
                    j++;
                }
                if (j - i == Game.PICK) {
                    int[] d = asDraw(Arrays.copyOfRange(vals, i, j));
                    if (d != null) {
                        if (dateCell >= 0 && i == dateCell + 1) {
                            return d;
                        }
                        if (first == null) {
                            first = d;
                        }
                    }
                }
                i = j;
            } else {
                i++;
            }
        }
        return first;
    }

    /** Fallback: one cell that by itself holds exactly 11 different numbers ("1 4 7 ..."). */
    private static int[] singleCellDraw(List<String> cells, boolean[] excluded) {
        for (int i = 0; i < cells.size(); i++) {
            if (excluded[i]) {
                continue;
            }
            int[] nums = extractNumbers(cells.get(i));
            if (nums.length == Game.PICK) {
                int[] d = asDraw(nums);
                if (d != null) {
                    return d;
                }
            }
        }
        return null;
    }

    private static void appendNumbers(String text, IntList out) {
        if (text == null) {
            return;
        }
        int len = text.length();
        int i = 0;
        while (i < len) {
            char ch = text.charAt(i);
            if (ch >= '0' && ch <= '9') {
                int j = i;
                int v = 0;
                while (j < len) {
                    char d = text.charAt(j);
                    if (d < '0' || d > '9') {
                        break;
                    }
                    if (j - i < 4) {
                        v = v * 10 + (d - '0');
                    }
                    j++;
                }
                boolean glued = (i > 0 && isWordChar(text.charAt(i - 1)))
                        || (j < len && isWordChar(text.charAt(j)));
                // "-3" is minus three (not a draw number); in "1-3" the dash only separates
                boolean negative = i > 0 && text.charAt(i - 1) == '-'
                        && (i < 2 || text.charAt(i - 2) < '0' || text.charAt(i - 2) > '9');
                if (!glued && !negative && j - i <= 4 && Game.isValidNumber(v)) {
                    out.add(v);
                }
                i = j;
            } else {
                i++;
            }
        }
    }

    private static boolean isWordChar(char c) {
        return c == '_' || Character.isLetter(c);
    }

    private static int distinctCount(int[] c) {
        long m = Draw.mask(c);
        return Long.bitCount(m);
    }

    // =================================================================== XML

    /** Result of evaluating one element's own text. */
    private static final class Eval {
        final int[] draw;
        final int candidates;
        final int distinct;

        Eval(int[] draw, int candidates, int distinct) {
            this.draw = draw;
            this.candidates = candidates;
            this.distinct = distinct;
        }
    }

    private static void xmlInto(String content, boolean dayFirst, ParseResult r) {
        String c = content == null ? "" : content;
        int first = firstNonBlank(c);
        if (first < 0) {
            r.addWarning("The XML file is empty.");
            return;
        }
        c = stripDoctype(c.substring(first));           // also drops a BOM / leading whitespace
        Document doc;
        try {
            DocumentBuilder builder = newSafeBuilder();
            doc = builder.parse(new InputSource(new StringReader(c)));
        } catch (SAXParseException e) {
            r.addWarning("XML could not be parsed (line " + e.getLineNumber() + "): " + describe(e));
            return;
        } catch (Exception e) {
            r.addWarning("XML could not be parsed: " + describe(e));
            return;
        } catch (OutOfMemoryError e) {
            r.addWarning("XML could not be parsed: the file is too large for the available memory.");
            return;
        }
        Element root = doc == null ? null : doc.getDocumentElement();
        if (root == null) {
            r.addWarning("XML has no root element.");
            return;
        }

        // Pre-order list of all elements with parent indices (iterative: no recursion limits).
        ArrayList<Element> order = new ArrayList<Element>();
        IntList parents = new IntList(64);
        ArrayList<Element> stack = new ArrayList<Element>();
        IntList stackParents = new IntList(64);
        ArrayList<Element> kids = new ArrayList<Element>();
        stack.add(root);
        stackParents.add(-1);
        while (!stack.isEmpty()) {
            Element e = stack.remove(stack.size() - 1);
            int p = stackParents.removeLast();
            int idx = order.size();
            order.add(e);
            parents.add(p);
            kids.clear();
            for (Node ch = e.getFirstChild(); ch != null; ch = ch.getNextSibling()) {
                if (ch.getNodeType() == Node.ELEMENT_NODE) {
                    kids.add((Element) ch);
                }
            }
            for (int k = kids.size() - 1; k >= 0; k--) {
                stack.add(kids.get(k));
                stackParents.add(idx);
            }
        }

        // Reverse pre-order = every element after all of its descendants (deepest rule).
        int n = order.size();
        int[][] drawNums = new int[n][];
        boolean[] drawBelow = new boolean[n];
        boolean[] failed = new boolean[n];
        boolean[] failBelow = new boolean[n];
        int[] failCandidates = new int[n];
        int[] failDistinct = new int[n];
        for (int idx = n - 1; idx >= 0; idx--) {
            if (!drawBelow[idx]) {
                Eval ev = evaluate(order.get(idx), dayFirst);
                if (ev.draw != null) {
                    drawNums[idx] = ev.draw;
                } else if (!failBelow[idx] && ev.candidates >= 2) {
                    failed[idx] = true;
                    failCandidates[idx] = ev.candidates;
                    failDistinct[idx] = ev.distinct;
                }
            }
            int p = parents.get(idx);
            if (p >= 0) {
                if (drawNums[idx] != null || drawBelow[idx]) {
                    drawBelow[p] = true;
                }
                if (failed[idx] || failBelow[idx]) {
                    failBelow[p] = true;
                }
            }
        }

        // Document order: collect draws (with dates) and warnings.
        boolean[] insideDraw = new boolean[n];
        HashMap<Node, String> dateCache = new HashMap<Node, String>();
        int record = 0;
        int found = 0;
        for (int idx = 0; idx < n; idx++) {
            int p = parents.get(idx);
            insideDraw[idx] = p >= 0 && (insideDraw[p] || drawNums[p] != null);
            Element e = order.get(idx);
            if (drawNums[idx] != null) {
                record++;
                r.totalLines++;
                String iso = elementDate(e, dayFirst, true);
                if (iso == null) {
                    String bad = badDateOf(e, dayFirst);
                    if (bad != null) {
                        r.skippedLines++;
                        r.addWarning("element <" + e.getNodeName() + "> #" + record
                                + ": invalid date \"" + abbreviate(bad) + "\"");
                        continue;
                    }
                    iso = ancestorDate(e, dayFirst, dateCache);
                }
                found++;
                if (iso != null) {
                    r.dated.add(new Draw(iso, drawNums[idx]));
                } else {
                    r.undated.add(drawNums[idx]);
                }
            } else if (failed[idx] && !insideDraw[idx]) {
                record++;
                r.totalLines++;
                r.skippedLines++;
                r.addWarning("element <" + e.getNodeName() + "> #" + record + ": "
                        + problem(failCandidates[idx], failDistinct[idx]));
            }
        }
        if (record == 0 && found == 0) {
            r.addWarning("No draws found in the XML: a draw element needs exactly 11 different numbers from 1 to 22.");
        }
    }

    /**
     * Own text of an element = attribute values (except date-like ones) + direct text +
     * text of leaf child elements (except date-like ones); dates and times inside the text
     * are removed first so their digits never count as numbers. If that is not a draw, the
     * text without attributes and then each attribute on its own are tried (this tolerates
     * e.g. an {@code id="5"} attribute next to the numbers).
     */
    private static Eval evaluate(Element e, boolean dayFirst) {
        StringBuilder attrText = new StringBuilder();
        ArrayList<String> attrValues = null;
        NamedNodeMap attrs = e.getAttributes();
        if (attrs != null) {
            for (int i = 0; i < attrs.getLength(); i++) {
                Node a = attrs.item(i);
                String value = a.getNodeValue();
                if (value == null || !hasDigit(value) || isDateName(a.getNodeName())
                        || isDateOrTime(value, dayFirst)) {
                    continue;
                }
                attrText.append(value).append(' ');
                if (attrValues == null) {
                    attrValues = new ArrayList<String>();
                }
                attrValues.add(value);
            }
        }
        StringBuilder text = new StringBuilder();
        for (Node ch = e.getFirstChild(); ch != null; ch = ch.getNextSibling()) {
            short t = ch.getNodeType();
            if (t == Node.TEXT_NODE || t == Node.CDATA_SECTION_NODE) {
                appendClean(text, ch.getNodeValue(), dayFirst);
            } else if (t == Node.ELEMENT_NODE && isLeaf(ch) && !isDateName(ch.getNodeName())) {
                String v = ch.getTextContent();
                if (v != null && !isDateOrTime(v, dayFirst)) {
                    appendClean(text, v, dayFirst);
                }
            }
        }
        int[] all = extractNumbers(attrText.toString() + " " + text.toString());
        int[] draw = exactDraw(all);
        if (draw == null && attrValues != null) {
            draw = exactDraw(extractNumbers(text.toString()));
            if (draw == null) {
                for (String v : attrValues) {
                    draw = exactDraw(extractNumbers(v));
                    if (draw != null) {
                        break;
                    }
                }
            }
        }
        if (draw == null) {
            draw = leafGroupDraw(e);
        }
        if (draw == null && all.length <= Game.PICK + 1) {
            // spec rule (11 distinct numbers) for one stray duplicate such as an id equal to
            // a drawn number; larger lists are rejected (they are several records or junk)
            draw = asDraw(all);
        }
        return new Eval(draw, all.length, distinctCount(all));
    }

    /** asDraw, but only for exactly 11 candidates. */
    private static int[] exactDraw(int[] c) {
        return c.length == Game.PICK ? asDraw(c) : null;
    }

    /**
     * Eleven leaf children with the same tag name that each hold one number, e.g.
     * {@code <n>1</n> ... <n>22</n>} next to an {@code <id>7</id>} or {@code <bonus>3</bonus>}.
     */
    private static int[] leafGroupDraw(Element e) {
        ArrayList<String> names = null;
        ArrayList<IntList> groups = null;
        for (Node ch = e.getFirstChild(); ch != null; ch = ch.getNextSibling()) {
            if (ch.getNodeType() != Node.ELEMENT_NODE || !isLeaf(ch)) {
                continue;
            }
            String t = ch.getTextContent();
            int v = t == null ? -1 : singleValue(t.trim());
            if (v < 0) {
                continue;
            }
            if (names == null) {
                names = new ArrayList<String>();
                groups = new ArrayList<IntList>();
            }
            String name = ch.getNodeName();
            int gi = names.indexOf(name);
            if (gi < 0) {
                names.add(name);
                groups.add(new IntList(Game.PICK));
                gi = names.size() - 1;
            }
            groups.get(gi).add(v);
        }
        if (groups == null) {
            return null;
        }
        for (IntList g : groups) {
            int[] d = exactDraw(g.toArray());
            if (d != null) {
                return d;
            }
        }
        return null;
    }

    private static void appendClean(StringBuilder sb, String text, boolean dayFirst) {
        if (text == null || text.length() == 0) {
            return;
        }
        String t = DateUtil.removeDates(text, dayFirst);
        if (t.indexOf(':') >= 0) {
            t = TIME_TOKEN.matcher(t).replaceAll(" ");
        }
        sb.append(t).append(' ');
    }

    /** A date-shaped but invalid attribute value / leaf child text of {@code e} (e.g. "31.02.2026"), or null. */
    private static String badDateOf(Element e, boolean dayFirst) {
        NamedNodeMap attrs = e.getAttributes();
        int na = attrs == null ? 0 : attrs.getLength();
        for (int i = 0; i < na; i++) {
            String v = attrs.item(i).getNodeValue();
            if (DateUtil.looksLikeDate(v) && DateUtil.normalize(v, dayFirst) == null) {
                return v.trim();
            }
        }
        for (Node ch = e.getFirstChild(); ch != null; ch = ch.getNextSibling()) {
            if (ch.getNodeType() == Node.ELEMENT_NODE && isLeaf(ch)) {
                String v = ch.getTextContent();
                if (DateUtil.looksLikeDate(v) && DateUtil.normalize(v, dayFirst) == null) {
                    return v.trim();
                }
            }
        }
        return null;
    }

    /** Rule (e): the date of the nearest ancestor that has a date-like attribute or child. */
    private static String ancestorDate(Element e, boolean dayFirst, HashMap<Node, String> cache) {
        for (Node p = e.getParentNode(); p != null && p.getNodeType() == Node.ELEMENT_NODE;
             p = p.getParentNode()) {
            String cached = cache.get(p);
            if (cached == null) {
                String v = elementDate((Element) p, dayFirst, false);
                cached = v == null ? "" : v;
                cache.put(p, cached);
            }
            if (cached.length() > 0) {
                return cached;
            }
        }
        return null;
    }

    private static String elementDate(Element e, boolean dayFirst, boolean useOwnText) {
        NamedNodeMap attrs = e.getAttributes();
        int na = attrs == null ? 0 : attrs.getLength();
        // (a) an attribute named like a date
        for (int i = 0; i < na; i++) {
            Node a = attrs.item(i);
            if (isDateName(a.getNodeName())) {
                String iso = DateUtil.normalize(a.getNodeValue(), dayFirst);
                if (iso != null) {
                    return iso;
                }
            }
        }
        // (b) any attribute value that is a date
        for (int i = 0; i < na; i++) {
            String iso = DateUtil.normalize(attrs.item(i).getNodeValue(), dayFirst);
            if (iso != null) {
                return iso;
            }
        }
        // (c) a leaf child element named like a date, then any leaf child that is just a date
        for (Node ch = e.getFirstChild(); ch != null; ch = ch.getNextSibling()) {
            if (ch.getNodeType() == Node.ELEMENT_NODE && isDateName(ch.getNodeName()) && isLeaf(ch)) {
                String v = ch.getTextContent();
                String iso = DateUtil.normalize(v, dayFirst);
                if (iso == null) {
                    iso = DateUtil.findDate(v, dayFirst);
                }
                if (iso != null) {
                    return iso;
                }
            }
        }
        for (Node ch = e.getFirstChild(); ch != null; ch = ch.getNextSibling()) {
            if (ch.getNodeType() == Node.ELEMENT_NODE && isLeaf(ch)) {
                String iso = DateUtil.normalize(ch.getTextContent(), dayFirst);
                if (iso != null) {
                    return iso;
                }
            }
        }
        // (d) a date inside its own text
        if (useOwnText) {
            StringBuilder sb = new StringBuilder();
            for (Node ch = e.getFirstChild(); ch != null; ch = ch.getNextSibling()) {
                short t = ch.getNodeType();
                if (t == Node.TEXT_NODE || t == Node.CDATA_SECTION_NODE) {
                    sb.append(ch.getNodeValue()).append(' ');
                } else if (t == Node.ELEMENT_NODE && isLeaf(ch)) {
                    sb.append(ch.getTextContent()).append(' ');
                }
            }
            return DateUtil.findDate(sb.toString(), dayFirst);
        }
        return null;
    }

    private static boolean isLeaf(Node n) {
        for (Node ch = n.getFirstChild(); ch != null; ch = ch.getNextSibling()) {
            if (ch.getNodeType() == Node.ELEMENT_NODE) {
                return false;
            }
        }
        return true;
    }

    /** date, datum, day, drawdate, draw_date, time, datetime, ... (case-insensitive, prefix ignored). */
    private static boolean isDateName(String name) {
        if (name == null) {
            return false;
        }
        String n = name;
        int colon = n.lastIndexOf(':');
        if (colon >= 0) {
            n = n.substring(colon + 1);
        }
        n = n.toLowerCase(Locale.US);
        return n.contains("date") || n.contains("datum") || n.contains("time")
                || n.equals("day") || n.equals("dan") || n.equals("when");
    }

    private static boolean isDateOrTime(String value, boolean dayFirst) {
        return DateUtil.normalize(value, dayFirst) != null || isTimeLike(value)
                || DateUtil.looksLikeDate(value);
    }

    private static boolean hasDigit(String s) {
        for (int i = 0, len = s.length(); i < len; i++) {
            char ch = s.charAt(i);
            if (ch >= '0' && ch <= '9') {
                return true;
            }
        }
        return false;
    }

    /** Removes a DOCTYPE declaration (DTDs are never processed). */
    private static String stripDoctype(String c) {
        int limit = Math.min(c.length(), 1 << 16);
        int i = -1;
        for (int k = 0; k + 9 <= limit; k++) {
            if (c.charAt(k) == '<' && c.regionMatches(true, k, "<!DOCTYPE", 0, 9)) {
                i = k;
                break;
            }
        }
        if (i < 0) {
            return c;
        }
        int depth = 0;
        for (int k = i + 9; k < c.length(); k++) {
            char ch = c.charAt(k);
            if (ch == '[') {
                depth++;
            } else if (ch == ']') {
                depth--;
            } else if (ch == '>' && depth <= 0) {
                return c.substring(0, i) + c.substring(k + 1);
            }
        }
        return c;
    }

    /** DOM builder with DTDs, external entities and XInclude disabled (each feature best-effort). */
    private static DocumentBuilder newSafeBuilder() throws ParserConfigurationException {
        DocumentBuilderFactory f = DocumentBuilderFactory.newInstance();
        f.setNamespaceAware(false);
        f.setValidating(false);
        try {
            f.setIgnoringComments(true);
        } catch (Throwable ignored) {
            // optional
        }
        try {
            f.setCoalescing(true);
        } catch (Throwable ignored) {
            // optional
        }
        setFeature(f, "http://apache.org/xml/features/disallow-doctype-decl", true);
        setFeature(f, "http://xml.org/sax/features/external-general-entities", false);
        setFeature(f, "http://xml.org/sax/features/external-parameter-entities", false);
        setFeature(f, "http://apache.org/xml/features/nonvalidating/load-external-dtd", false);
        setFeature(f, XMLConstants.FEATURE_SECURE_PROCESSING, true);
        try {
            f.setXIncludeAware(false);
        } catch (Throwable ignored) {
            // not supported by every parser (e.g. older Android ones)
        }
        DocumentBuilder b = f.newDocumentBuilder();
        b.setEntityResolver(new EntityResolver() {
            @Override
            public InputSource resolveEntity(String publicId, String systemId) {
                return new InputSource(new StringReader(""));   // never fetch anything
            }
        });
        b.setErrorHandler(new ErrorHandler() {
            @Override
            public void warning(SAXParseException exception) {
                // ignore
            }

            @Override
            public void error(SAXParseException exception) throws SAXException {
                throw exception;
            }

            @Override
            public void fatalError(SAXParseException exception) throws SAXException {
                throw exception;
            }
        });
        return b;
    }

    private static void setFeature(DocumentBuilderFactory f, String feature, boolean value) {
        try {
            f.setFeature(feature, value);
        } catch (Throwable ignored) {
            // feature unknown to this parser implementation
        }
    }

    // =================================================================== shared helpers

    private static int firstNonBlank(String s) {
        for (int i = 0, len = s.length(); i < len; i++) {
            char ch = s.charAt(i);
            if (ch != '\uFEFF' && !Character.isWhitespace(ch) && ch != '\u0000') {
                return i;
            }
        }
        return -1;
    }

    private static String describe(Throwable e) {
        String m = e.getMessage();
        if (m == null || m.trim().length() == 0) {
            m = e.getClass().getSimpleName();
        }
        m = m.replace('\n', ' ').trim();
        return m.length() > 200 ? m.substring(0, 200) + "..." : m;
    }

    /** Minimal growable int array. */
    static final class IntList {
        private int[] a;
        private int n;

        IntList(int capacity) {
            a = new int[Math.max(4, capacity)];
        }

        void add(int v) {
            if (n == a.length) {
                a = Arrays.copyOf(a, n * 2);
            }
            a[n++] = v;
        }

        int get(int i) {
            return a[i];
        }

        int removeLast() {
            return a[--n];
        }

        int size() {
            return n;
        }

        int[] toArray() {
            return Arrays.copyOf(a, n);
        }
    }
}
