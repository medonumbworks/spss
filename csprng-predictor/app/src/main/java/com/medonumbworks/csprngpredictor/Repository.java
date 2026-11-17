package com.medonumbworks.csprngpredictor;

import android.content.Context;
import android.util.Log;

import com.medonumbworks.csprngpredictor.core.Draw;
import com.medonumbworks.csprngpredictor.core.DrawCsv;
import com.medonumbworks.csprngpredictor.core.DrawSet;
import com.medonumbworks.csprngpredictor.core.Game;
import com.medonumbworks.csprngpredictor.core.IssuedLog;
import com.medonumbworks.csprngpredictor.core.IssuedPrediction;
import com.medonumbworks.csprngpredictor.core.ModelParams;
import com.medonumbworks.csprngpredictor.core.Prediction;
import com.medonumbworks.csprngpredictor.core.Replay;
import com.medonumbworks.csprngpredictor.core.ReplayResult;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * Process-wide data store: the draws ({@code files/draws.csv}), the log of issued predictions
 * ({@code files/issued.csv}) and the model parameters (via {@link Prefs}).
 *
 * <p>All methods are synchronous and thread-safe. The UI calls the potentially slow ones
 * (anything that may load, replay, import or write) from a background thread. The expensive
 * walk-forward replay runs outside the data lock, so quick reads from other threads are never
 * blocked by it; its result is cached until the draws or the parameters change.
 *
 * <p>Loading is tolerant: a corrupt or unreadable file yields an empty (or partial) set, a
 * {@code Log.w} and a {@code .bak} copy of the original file - never a crash.
 */
public final class Repository {

    private static final String TAG = "CsprngRepository";

    public static final String DRAWS_FILE = "draws.csv";
    public static final String ISSUED_FILE = "issued.csv";

    /**
     * Safety cap for the replay: only the most recent draws are replayed. A replay keeps one
     * prediction record (about 1.2 KB) per draw in memory, so an unbounded replay of a huge
     * import could exhaust the heap. 20,000 daily draws are more than 50 years of history.
     */
    public static final int MAX_REPLAY_DRAWS = 20000;

    private static Repository instance;

    /** The singleton (created with the application context). */
    public static Repository get(Context context) {
        synchronized (Repository.class) {
            if (instance == null) {
                instance = new Repository(context.getApplicationContext());
            }
            return instance;
        }
    }

    /** Outcome of adding (or replacing) draws. */
    public static final class AddResult {
        /** Draws actually added. */
        public final int added;
        /** Offered draws that already existed (or were repeated in the input). */
        public final int duplicates;
        /** Copies of the issued predictions that were settled by the new draws. */
        public final List<IssuedPrediction> settled;
        /** Human readable problem while writing the files, or null. */
        public final String saveProblem;

        AddResult(int added, int duplicates, List<IssuedPrediction> settled, String saveProblem) {
            this.added = added;
            this.duplicates = duplicates;
            this.settled = Collections.unmodifiableList(settled);
            this.saveProblem = saveProblem;
        }
    }

    private final Context app;
    private final Prefs prefs;
    private final File dir;
    /** Serialises replay computations (taken before, never while holding, {@code this}). */
    private final Object replayLock = new Object();

    // ---- guarded by "this"
    private DrawSet draws;
    private IssuedLog issued;
    private ModelParams params;
    private int evaluateLastN;
    private ReplayResult cachedReplay;
    private long version;
    private String saveProblem;

    private Repository(Context appContext) {
        this.app = appContext;
        this.prefs = new Prefs(appContext);
        this.dir = appContext.getFilesDir();
    }

    // =================================================================== data access

    /** The live draw set (loaded from disk on first access). Do not modify it directly. */
    public synchronized DrawSet draws() {
        ensureLoaded();
        return draws;
    }

    /** The live issued-prediction log. Do not modify it directly. */
    public synchronized IssuedLog issued() {
        ensureLoaded();
        return issued;
    }

    /** Copies of all issued predictions in issue order (safe to read on any thread). */
    public synchronized List<IssuedPrediction> issuedSnapshot() {
        ensureLoaded();
        List<IssuedPrediction> all = issued.all();
        List<IssuedPrediction> out = new ArrayList<IssuedPrediction>(all.size());
        for (IssuedPrediction p : all) {
            IssuedPrediction c = copyOf(p);
            if (c != null) {
                out.add(c);
            }
        }
        return out;
    }

    /** A copy of {@code issued().lastSettled()}, or null. */
    public synchronized IssuedPrediction lastSettledCopy() {
        ensureLoaded();
        return copyOf(issued.lastSettled());
    }

    /** A copy of the current parameters. */
    public synchronized ModelParams params() {
        ensureLoaded();
        return params.copy();
    }

    /** Persists a clamped copy; invalidates the cached replay when a model setting changed. */
    public synchronized void setParams(ModelParams p) {
        ensureLoaded();
        ModelParams q = p == null ? new ModelParams() : p.copy();
        q.clamp();
        boolean modelChanged = !sameModelSettings(q, params);
        params = q;
        prefs.saveParams(q);
        if (modelChanged) {
            invalidateLocked();
        }
    }

    public synchronized int evaluateLastN() {
        ensureLoaded();
        return evaluateLastN;
    }

    public synchronized void setEvaluateLastN(int n) {
        ensureLoaded();
        int v = Prefs.clampEvaluateLastN(n);
        prefs.setEvaluateLastN(v);
        if (v != evaluateLastN) {
            evaluateLastN = v;
            invalidateLocked();
        }
    }

    /** The Prefs instance used by the repository. */
    public Prefs prefs() {
        return prefs;
    }

    /** The chronological draws the replay uses (the most recent {@link #MAX_REPLAY_DRAWS}). */
    public synchronized List<Draw> replayDraws() {
        ensureLoaded();
        return replayWindow(draws.all());
    }

    /** True when there are more draws than {@link #MAX_REPLAY_DRAWS}. */
    public synchronized boolean isReplayCapped() {
        ensureLoaded();
        return draws.size() > MAX_REPLAY_DRAWS;
    }

    // =================================================================== replay / prediction

    /**
     * Walk-forward replay over the draws with the current parameters (cached; recomputed
     * lazily after any draw / parameter change). May take a while: call off the main thread.
     */
    public ReplayResult replay() {
        synchronized (replayLock) {
            List<Draw> snapshot;
            ModelParams p;
            int evalN;
            long v;
            synchronized (this) {
                ensureLoaded();
                if (cachedReplay != null) {
                    return cachedReplay;
                }
                snapshot = replayWindow(draws.all());
                p = params.copy();
                evalN = evaluateLastN;
                v = version;
            }
            ReplayResult r = Replay.run(snapshot, p, evalN);
            synchronized (this) {
                if (v == version) {
                    cachedReplay = r;
                }
            }
            return r;
        }
    }

    /** {@code replay().next}; also logs it as issued (see {@link #currentReplay()}). */
    public Prediction currentPrediction() {
        return currentReplay().next;
    }

    /**
     * The replay for the current data, whose {@code next} prediction is about to be shown:
     * it is logged in the issued log unless a pending entry with the same basis
     * (basedOnDate / basedOnCount) exists. Nothing is logged while there are no draws (that
     * "prediction" would just be the numbers 1..11 and is never shown).
     */
    public ReplayResult currentReplay() {
        ReplayResult r = null;
        for (int attempt = 0; attempt < 4; attempt++) {
            long v;
            synchronized (this) {
                ensureLoaded();
                v = version;
            }
            r = replay();
            synchronized (this) {
                // The data may have changed while the replay ran: then compute again, so the
                // logged prediction always matches the data it claims to be based on.
                if (v == version || attempt == 3) {
                    logIssuedLocked(r.next);
                    return r;
                }
            }
        }
        return r;
    }

    // =================================================================== changes

    /** Adds the draws, persists, invalidates the replay and settles issued predictions. */
    public int addDraws(Collection<Draw> ds) {
        return addDrawsDetailed(ds).added;
    }

    /** Like {@link #addDraws} but also reports duplicates, settled predictions and save problems. */
    public synchronized AddResult addDrawsDetailed(Collection<Draw> ds) {
        ensureLoaded();
        int offered = 0;
        if (ds != null) {
            for (Draw d : ds) {
                if (d != null) {
                    offered++;
                }
            }
        }
        int added = draws.addAll(ds);
        String problem = null;
        List<IssuedPrediction> settled = new ArrayList<IssuedPrediction>();
        if (added > 0) {
            invalidateLocked();
            saveDrawsLocked();
            settled = settleLocked();
            problem = takeProblemLocked();
        }
        return new AddResult(added, offered - added, settled, problem);
    }

    /** Adds one draw; false if an equal draw exists. */
    public boolean addDraw(Draw d) {
        if (d == null) {
            return false;
        }
        return addDrawsDetailed(Collections.singletonList(d)).added > 0;
    }

    /**
     * Removes {@code old} (e.g. the draws on the same date) and adds {@code d} in one step,
     * with one write; settles issued predictions.
     */
    public synchronized AddResult replaceDraws(Collection<Draw> old, Draw d) {
        ensureLoaded();
        boolean removed = false;
        if (old != null) {
            for (Draw o : old) {
                if (o != null && !o.equals(d) && draws.remove(o)) {
                    removed = true;
                }
            }
        }
        boolean added = d != null && draws.add(d);
        String problem = null;
        List<IssuedPrediction> settled = new ArrayList<IssuedPrediction>();
        if (removed || added) {
            invalidateLocked();
            saveDrawsLocked();
            if (added) {
                settled = settleLocked();
            }
            problem = takeProblemLocked();
        }
        return new AddResult(added ? 1 : 0, (d != null && !added) ? 1 : 0, settled, problem);
    }

    public synchronized boolean removeDraw(Draw d) {
        ensureLoaded();
        if (!draws.remove(d)) {
            return false;
        }
        invalidateLocked();
        saveDrawsLocked();
        return true;
    }

    public synchronized void clearDraws() {
        ensureLoaded();
        if (draws.isEmpty()) {
            return;
        }
        draws.clear();
        invalidateLocked();
        saveDrawsLocked();
    }

    /** issued().settle(draws()) + persist; returns the number settled. */
    public synchronized int settleIssued() {
        ensureLoaded();
        int n = issued.settle(draws);
        if (n > 0) {
            saveIssuedLocked();
        }
        return n;
    }

    public synchronized void clearIssued() {
        ensureLoaded();
        issued.clear();
        saveIssuedLocked();
    }

    public synchronized String exportDrawsCsv() {
        ensureLoaded();
        return DrawCsv.write(draws.all());
    }

    public synchronized String exportIssuedCsv() {
        ensureLoaded();
        return issued.toCsv();
    }

    /** Persists everything (atomic writes: tmp file, then rename). */
    public synchronized void save() {
        ensureLoaded();
        saveDrawsLocked();
        saveIssuedLocked();
        prefs.saveParams(params);
        prefs.setEvaluateLastN(evaluateLastN);
    }

    /** The last problem while writing a file (then cleared), or null. */
    public synchronized String takeSaveProblem() {
        return takeProblemLocked();
    }

    // =================================================================== internals

    private void ensureLoaded() {
        if (draws != null) {
            return;
        }
        params = prefs.loadParams();
        evaluateLastN = prefs.evaluateLastN();
        draws = loadDraws();
        issued = loadIssued();
    }

    private void invalidateLocked() {
        version++;
        cachedReplay = null;
    }

    private String takeProblemLocked() {
        String p = saveProblem;
        saveProblem = null;
        return p;
    }

    private void logIssuedLocked(Prediction next) {
        if (next == null || next.basedOnCount <= 0 || !Game.isValidDraw(next.numbers)) {
            return;
        }
        if (issued.pendingFor(next.basedOnDate, next.basedOnCount) != null) {
            return;
        }
        try {
            issued.add(IssuedPrediction.of(next, null));
            saveIssuedLocked();
        } catch (IllegalArgumentException e) {
            Log.w(TAG, "Prediction not logged: " + e.getMessage());
        }
    }

    /** Settles pending predictions; returns copies of the ones that were settled now. */
    private List<IssuedPrediction> settleLocked() {
        List<IssuedPrediction> pendingBefore = new ArrayList<IssuedPrediction>();
        for (IssuedPrediction p : issued.all()) {
            if (p.isPending()) {
                pendingBefore.add(p);
            }
        }
        List<IssuedPrediction> out = new ArrayList<IssuedPrediction>();
        if (pendingBefore.isEmpty()) {
            return out;
        }
        int n = issued.settle(draws);
        if (n > 0) {
            for (IssuedPrediction p : pendingBefore) {
                if (!p.isPending()) {
                    IssuedPrediction c = copyOf(p);
                    if (c != null) {
                        out.add(c);
                    }
                }
            }
            saveIssuedLocked();
        }
        return out;
    }

    private static List<Draw> replayWindow(List<Draw> all) {
        if (all.size() <= MAX_REPLAY_DRAWS) {
            return all;
        }
        return new ArrayList<Draw>(all.subList(all.size() - MAX_REPLAY_DRAWS, all.size()));
    }

    /** True when the two parameter sets give the same model (dayFirst does not matter). */
    private static boolean sameModelSettings(ModelParams a, ModelParams b) {
        if (a == null || b == null) {
            return a == b;
        }
        ModelParams x = a.copy();
        ModelParams y = b.copy();
        x.dayFirst = true;
        y.dayFirst = true;
        return x.equals(y);
    }

    static IssuedPrediction copyOf(IssuedPrediction p) {
        if (p == null) {
            return null;
        }
        try {
            IssuedPrediction c = new IssuedPrediction(p.issuedAt, p.basedOnDate, p.basedOnCount,
                    p.targetDate, p.numbers);
            c.actualDate = p.actualDate;
            c.actual = p.actual == null ? null : p.actual.clone();
            c.hits = p.hits;
            return c;
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    // ---- loading

    private DrawSet loadDraws() {
        File f = new File(dir, DRAWS_FILE);
        if (!f.exists()) {
            return new DrawSet();
        }
        try {
            String csv = readFile(f);
            List<String> problems = new ArrayList<String>();
            List<Draw> list = DrawCsv.read(csv, problems);
            if (!problems.isEmpty()) {
                Log.w(TAG, DRAWS_FILE + ": " + problems.size() + " unreadable line(s), first: "
                        + problems.get(0) + " - keeping a backup copy");
                backup(f);
            }
            return new DrawSet(list);
        } catch (IOException e) {
            Log.w(TAG, "Could not read " + DRAWS_FILE + "; starting empty", e);
            backup(f);
        } catch (RuntimeException e) {
            Log.w(TAG, "Corrupt " + DRAWS_FILE + "; starting empty", e);
            backup(f);
        } catch (OutOfMemoryError e) {
            Log.w(TAG, DRAWS_FILE + " is too large to load; starting empty", e);
        }
        return new DrawSet();
    }

    private IssuedLog loadIssued() {
        File f = new File(dir, ISSUED_FILE);
        if (!f.exists()) {
            return new IssuedLog();
        }
        try {
            return IssuedLog.fromCsv(readFile(f));
        } catch (IOException e) {
            Log.w(TAG, "Could not read " + ISSUED_FILE + "; starting empty", e);
            backup(f);
        } catch (RuntimeException e) {
            Log.w(TAG, "Corrupt " + ISSUED_FILE + "; starting empty", e);
            backup(f);
        } catch (OutOfMemoryError e) {
            Log.w(TAG, ISSUED_FILE + " is too large to load; starting empty", e);
        }
        return new IssuedLog();
    }

    private static String readFile(File f) throws IOException {
        long len = f.length();
        if (len > Integer.MAX_VALUE / 2) {
            throw new IOException("file too large: " + len + " bytes");
        }
        InputStream in = new FileInputStream(f);
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream((int) Math.max(256L, len + 16L));
            byte[] buf = new byte[64 * 1024];
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
            }
            return out.toString("UTF-8");
        } finally {
            closeQuietly(in);
        }
    }

    /** Copies a file that could not be read completely to "name.bak" (best effort). */
    private static void backup(File f) {
        File bak = new File(f.getPath() + ".bak");
        InputStream in = null;
        OutputStream out = null;
        try {
            in = new FileInputStream(f);
            out = new FileOutputStream(bak);
            byte[] buf = new byte[64 * 1024];
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
            }
        } catch (IOException e) {
            Log.w(TAG, "Backup of " + f.getName() + " failed", e);
        } finally {
            closeQuietly(in);
            closeQuietly(out);
        }
    }

    // ---- saving

    /** Returns a problem description or null. */
    private String saveDrawsLocked() {
        try {
            writeAtomically(new File(dir, DRAWS_FILE), DrawCsv.write(draws.all()));
            return null;
        } catch (IOException e) {
            Log.e(TAG, "Could not save " + DRAWS_FILE, e);
            saveProblem = app.getString(R.string.save_failed_draws, String.valueOf(e.getMessage()));
        } catch (OutOfMemoryError e) {
            Log.e(TAG, "Could not save " + DRAWS_FILE, e);
            saveProblem = app.getString(R.string.save_failed_draws, "out of memory");
        }
        return saveProblem;
    }

    private String saveIssuedLocked() {
        try {
            writeAtomically(new File(dir, ISSUED_FILE), issued.toCsv());
            return null;
        } catch (IOException e) {
            Log.e(TAG, "Could not save " + ISSUED_FILE, e);
            saveProblem = app.getString(R.string.save_failed_issued, String.valueOf(e.getMessage()));
        } catch (OutOfMemoryError e) {
            Log.e(TAG, "Could not save " + ISSUED_FILE, e);
            saveProblem = app.getString(R.string.save_failed_issued, "out of memory");
        }
        return saveProblem;
    }

    /** Writes "name.tmp" (flushed and synced), then renames it over the target. */
    private static void writeAtomically(File target, String content) throws IOException {
        File parent = target.getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
            throw new IOException("cannot create " + parent);
        }
        File tmp = new File(target.getPath() + ".tmp");
        FileOutputStream fos = new FileOutputStream(tmp);
        boolean ok = false;
        try {
            Writer w = new OutputStreamWriter(new BufferedOutputStream(fos, 64 * 1024), "UTF-8");
            w.write(content);
            w.flush();
            fos.getFD().sync();
            w.close();
            ok = true;
        } finally {
            if (!ok) {
                closeQuietly(fos);
                if (!tmp.delete()) {
                    Log.w(TAG, "Could not delete " + tmp);
                }
            }
        }
        if (!tmp.renameTo(target)) {
            // Some file systems refuse to rename over an existing file.
            if (target.exists() && !target.delete()) {
                throw new IOException("cannot replace " + target.getName());
            }
            if (!tmp.renameTo(target)) {
                throw new IOException("cannot rename " + tmp.getName() + " to " + target.getName());
            }
        }
    }

    private static void closeQuietly(java.io.Closeable c) {
        if (c != null) {
            try {
                c.close();
            } catch (IOException ignored) {
                // nothing useful to do
            }
        }
    }
}
