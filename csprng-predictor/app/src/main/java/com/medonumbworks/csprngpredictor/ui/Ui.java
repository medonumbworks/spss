package com.medonumbworks.csprngpredictor.ui;

import android.app.Activity;
import android.app.ActionBar;
import android.app.Dialog;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.util.TypedValue;
import android.widget.Toast;

import com.medonumbworks.csprngpredictor.R;
import com.medonumbworks.csprngpredictor.Repository;
import com.medonumbworks.csprngpredictor.core.Game;
import com.medonumbworks.csprngpredictor.core.ModelParams;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Collection;
import java.util.Locale;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Callable;

/** Small helpers shared by the activities. */
final class Ui {

    private Ui() {
    }

    /** True when results may still be shown in {@code a}. */
    static boolean alive(Activity a) {
        return a != null && !a.isFinishing() && !a.isDestroyed();
    }

    static int dp(Context c, float dp) {
        return Math.round(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, dp,
                c.getResources().getDisplayMetrics()));
    }

    /** Shows the action bar's up arrow (handled as "finish" by the activities). */
    static void enableUp(Activity a) {
        ActionBar bar = a.getActionBar();
        if (bar != null) {
            bar.setDisplayHomeAsUpEnabled(true);
        }
    }

    static void toast(Context c, CharSequence text) {
        Toast.makeText(c, text, Toast.LENGTH_SHORT).show();
    }

    static void toastLong(Context c, CharSequence text) {
        Toast.makeText(c, text, Toast.LENGTH_LONG).show();
    }

    // ------------------------------------------------------------------ numbers

    static String join(int[] nums, String sep) {
        StringBuilder sb = new StringBuilder();
        if (nums != null) {
            for (int i = 0; i < nums.length; i++) {
                if (i > 0) {
                    sb.append(sep);
                }
                sb.append(nums[i]);
            }
        }
        return sb.toString();
    }

    static String join(Collection<Integer> nums, String sep) {
        StringBuilder sb = new StringBuilder();
        if (nums != null) {
            for (Integer n : nums) {
                if (sb.length() > 0) {
                    sb.append(sep);
                }
                sb.append(n);
            }
        }
        return sb.toString();
    }

    static Set<Integer> toSet(int[] nums) {
        TreeSet<Integer> s = new TreeSet<Integer>();
        if (nums != null) {
            for (int n : nums) {
                if (Game.isValidNumber(n)) {
                    s.add(n);
                }
            }
        }
        return s;
    }

    static int[] toArray(Collection<Integer> nums) {
        if (nums == null) {
            return new int[0];
        }
        int[] out = new int[nums.size()];
        int i = 0;
        for (Integer n : nums) {
            out[i++] = n.intValue();
        }
        return out;
    }

    /** {@code fraction} (0..1) as a percentage with the given decimals, e.g. "4.3%". */
    static String percent(double fraction, int decimals) {
        if (Double.isNaN(fraction) || Double.isInfinite(fraction)) {
            fraction = 0.0;
        }
        return String.format(Locale.US, "%." + decimals + "f%%", fraction * 100.0);
    }

    static String fmt2(double v) {
        return String.format(Locale.US, "%.2f", v);
    }

    /** Plain number: "15", "0.5", "0.02" (no trailing zeros, Locale independent). */
    static String number(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            return "0";
        }
        if (v == Math.rint(v) && Math.abs(v) < 1e15) {
            return Long.toString((long) v);
        }
        String s = String.format(Locale.US, "%.6f", v);
        int end = s.length();
        while (end > 0 && s.charAt(end - 1) == '0') {
            end--;
        }
        if (end > 0 && s.charAt(end - 1) == '.') {
            end--;
        }
        return s.substring(0, end);
    }

    /** The learning parameters that auto-tune changes, on one line. */
    static String tuneSummary(Context c, ModelParams p) {
        return c.getString(R.string.tune_summary, number(p.eta), number(p.halfLife), number(p.biasRate));
    }

    // ------------------------------------------------------------------ errors

    /** A short user-facing description of a background failure. */
    static String describe(Context c, Throwable t) {
        if (t instanceof OutOfMemoryError) {
            return c.getString(R.string.error_memory);
        }
        if (t instanceof SecurityException) {
            return c.getString(R.string.error_permission);
        }
        String m = t == null ? null : t.getMessage();
        if (m == null || m.trim().length() == 0) {
            m = t == null ? "?" : t.getClass().getSimpleName();
        }
        return m;
    }

    // ------------------------------------------------------------------ dialogs

    /** Keeps track of the dialog an activity shows, so it can be dismissed in onDestroy. */
    static final class Dialogs {
        private Dialog current;

        void show(Dialog d) {
            dismiss();
            current = d;
            d.show();
        }

        void dismiss() {
            if (current != null) {
                try {
                    if (current.isShowing()) {
                        current.dismiss();
                    }
                } catch (RuntimeException ignored) {
                    // window already gone
                }
                current = null;
            }
        }
    }

    // ------------------------------------------------------------------ Storage Access Framework

    /** Asks the system file chooser where to save a new text file. */
    static void startCreateDocument(Activity a, String type, String fileName, int requestCode) {
        Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType(type);
        i.putExtra(Intent.EXTRA_TITLE, fileName);
        try {
            a.startActivityForResult(i, requestCode);
        } catch (ActivityNotFoundException e) {
            toastLong(a, a.getString(R.string.no_file_picker));
        } catch (SecurityException e) {
            toastLong(a, a.getString(R.string.no_file_picker));
        }
    }

    /** Writes UTF-8 text to a SAF document (truncating it). */
    static void writeText(Context c, Uri uri, String text) throws IOException {
        OutputStream out = null;
        try {
            out = c.getContentResolver().openOutputStream(uri, "wt");
        } catch (IOException e) {
            out = null;                               // provider without "wt": try plain "w"
        } catch (IllegalArgumentException e) {
            out = null;
        } catch (UnsupportedOperationException e) {
            out = null;
        }
        if (out == null) {
            out = c.getContentResolver().openOutputStream(uri);
        }
        if (out == null) {
            throw new IOException("the file could not be opened for writing");
        }
        try {
            Writer w = new OutputStreamWriter(out, "UTF-8");
            w.write(text);
            w.flush();
        } finally {
            try {
                out.close();
            } catch (IOException ignored) {
                // already flushed
            }
        }
    }

    /** OpenableColumns.DISPLAY_NAME of a content Uri, or its last path segment. */
    static String displayName(Context c, Uri uri) {
        String name = null;
        Cursor cur = null;
        try {
            cur = c.getContentResolver().query(uri, new String[]{OpenableColumns.DISPLAY_NAME},
                    null, null, null);
            if (cur != null && cur.moveToFirst()) {
                int idx = cur.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (idx >= 0 && !cur.isNull(idx)) {
                    name = cur.getString(idx);
                }
            }
        } catch (RuntimeException ignored) {
            // SecurityException / IllegalArgumentException from odd providers: fall back
        } finally {
            if (cur != null) {
                cur.close();
            }
        }
        if (name == null || name.trim().length() == 0) {
            name = uri.getLastPathSegment();
        }
        return name == null ? "" : name;
    }

    /** OpenableColumns.SIZE of a content Uri, or -1 when unknown. */
    static long size(Context c, Uri uri) {
        long size = -1;
        Cursor cur = null;
        try {
            cur = c.getContentResolver().query(uri, new String[]{OpenableColumns.SIZE},
                    null, null, null);
            if (cur != null && cur.moveToFirst()) {
                int idx = cur.getColumnIndex(OpenableColumns.SIZE);
                if (idx >= 0 && !cur.isNull(idx)) {
                    size = cur.getLong(idx);
                }
            }
        } catch (RuntimeException ignored) {
            // unknown
        } finally {
            if (cur != null) {
                cur.close();
            }
        }
        return size;
    }

    /**
     * Background job writing the draws (or the issued log) to {@code uri}; returns the
     * document's display name. Static, so it does not capture an activity.
     */
    static Callable<Object> exportJob(final Context appContext, final Uri uri, final boolean draws) {
        return new Callable<Object>() {
            @Override
            public Object call() throws Exception {
                Repository repo = Repository.get(appContext);
                String text = draws ? repo.exportDrawsCsv() : repo.exportIssuedCsv();
                writeText(appContext, uri, text);
                return displayName(appContext, uri);
            }
        };
    }
}
