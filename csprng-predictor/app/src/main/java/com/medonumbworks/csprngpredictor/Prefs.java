package com.medonumbworks.csprngpredictor;

import android.content.Context;
import android.content.SharedPreferences;

import com.medonumbworks.csprngpredictor.core.ModelParams;

/**
 * SharedPreferences wrapper (file "settings"). The model parameters are stored as the
 * {@link ModelParams#serialize()} string under the key "params".
 */
public final class Prefs {

    public static final String FILE = "settings";
    public static final String KEY_PARAMS = "params";
    public static final String KEY_EVALUATE_LAST_N = "evaluateLastN";

    /** Default window for the backtest statistics and auto-tune. */
    public static final int DEFAULT_EVALUATE_LAST_N = 200;
    /** Largest accepted "evaluate last N" value; 0 means "all records". */
    public static final int MAX_EVALUATE_LAST_N = 1000000;

    private final SharedPreferences sp;

    public Prefs(Context context) {
        sp = context.getApplicationContext().getSharedPreferences(FILE, Context.MODE_PRIVATE);
    }

    /** The saved parameters (defaults when nothing or garbage is stored); never null. */
    public ModelParams loadParams() {
        String s = null;
        try {
            s = sp.getString(KEY_PARAMS, null);
        } catch (ClassCastException ignored) {
            // a value of another type under our key: fall back to defaults
        }
        return ModelParams.parse(s);
    }

    /** Persists a clamped copy of {@code p}. */
    public void saveParams(ModelParams p) {
        ModelParams q = p == null ? new ModelParams() : p.copy();
        q.clamp();
        sp.edit().putString(KEY_PARAMS, q.serialize()).apply();
    }

    /** Mirrors {@code loadParams().dayFirst}. */
    public boolean dayFirst() {
        return loadParams().dayFirst;
    }

    /** Changes only the day-first flag of the stored parameters. */
    public void setDayFirst(boolean dayFirst) {
        ModelParams p = loadParams();
        p.dayFirst = dayFirst;
        saveParams(p);
    }

    /** Window for the stats / auto-tune (default 200, 0 = all records). */
    public int evaluateLastN() {
        int v = DEFAULT_EVALUATE_LAST_N;
        try {
            v = sp.getInt(KEY_EVALUATE_LAST_N, DEFAULT_EVALUATE_LAST_N);
        } catch (ClassCastException ignored) {
            // keep the default
        }
        return clampEvaluateLastN(v);
    }

    public void setEvaluateLastN(int n) {
        sp.edit().putInt(KEY_EVALUATE_LAST_N, clampEvaluateLastN(n)).apply();
    }

    /** Forces {@code n} into 0..{@link #MAX_EVALUATE_LAST_N}. */
    public static int clampEvaluateLastN(int n) {
        return n < 0 ? 0 : (n > MAX_EVALUATE_LAST_N ? MAX_EVALUATE_LAST_N : n);
    }
}
