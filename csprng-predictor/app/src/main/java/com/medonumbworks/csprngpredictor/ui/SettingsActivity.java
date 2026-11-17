package com.medonumbworks.csprngpredictor.ui;

import android.app.Activity;
import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;

import com.medonumbworks.csprngpredictor.Prefs;
import com.medonumbworks.csprngpredictor.R;
import com.medonumbworks.csprngpredictor.Repository;
import com.medonumbworks.csprngpredictor.core.ModelParams;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * Plain form for the {@link ModelParams} fields and "evaluate last N". Save validates the
 * input, clamps it to the valid ranges and shows the clamped values.
 */
public final class SettingsActivity extends Activity implements Worker.Host {

    private static final String STATE_LOADED = "loaded";
    private static final String STATE_STATUS = "status";

    private static final int JOB_LOAD = 1;
    private static final int JOB_SAVE = 2;

    private Worker worker;

    private EditText editEta;
    private EditText editShare;
    private EditText editBiasRate;
    private EditText editBiasDecay;
    private EditText editShortWindow;
    private EditText editLongWindow;
    private EditText editHalfLife;
    private EditText editEvalN;
    private Switch switchDayFirst;
    private TextView status;
    private Button btnSave;
    private Button btnRestore;

    private boolean loaded;

    /** Loaded / saved values. */
    static final class Values {
        ModelParams params;
        int evaluateLastN;
        String message;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);
        Ui.enableUp(this);
        worker = Worker.restore(getLastNonConfigurationInstance());

        editEta = (EditText) findViewById(R.id.editEta);
        editShare = (EditText) findViewById(R.id.editShare);
        editBiasRate = (EditText) findViewById(R.id.editBiasRate);
        editBiasDecay = (EditText) findViewById(R.id.editBiasDecay);
        editShortWindow = (EditText) findViewById(R.id.editShortWindow);
        editLongWindow = (EditText) findViewById(R.id.editLongWindow);
        editHalfLife = (EditText) findViewById(R.id.editHalfLife);
        editEvalN = (EditText) findViewById(R.id.editEvalN);
        switchDayFirst = (Switch) findViewById(R.id.switchDayFirstSetting);
        status = (TextView) findViewById(R.id.settingsStatus);
        btnSave = (Button) findViewById(R.id.btnSettingsSave);
        btnRestore = (Button) findViewById(R.id.btnSettingsRestore);

        btnSave.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                save();
            }
        });
        btnRestore.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ModelParams d = new ModelParams();
                fill(d, Prefs.DEFAULT_EVALUATE_LAST_N);
                startSave(d, Prefs.DEFAULT_EVALUATE_LAST_N, getString(R.string.settings_defaults_restored));
            }
        });

        if (savedInstanceState != null) {
            // The framework restores the fields the user may have edited.
            loaded = savedInstanceState.getBoolean(STATE_LOADED, false);
            setStatus(savedInstanceState.getString(STATE_STATUS));
        }
        if (!loaded && !worker.isRunning(JOB_LOAD)) {
            worker.run(JOB_LOAD, loadJob(Repository.get(this)));
        }
        updateEnabled();
    }

    @Override
    protected void onStart() {
        super.onStart();
        worker.attach(this, this);
        updateEnabled();
    }

    @Override
    protected void onStop() {
        worker.detach(this);
        super.onStop();
    }

    @Override
    public Object onRetainNonConfigurationInstance() {
        return worker;
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        out.putBoolean(STATE_LOADED, loaded);
        if (status.getVisibility() == View.VISIBLE) {
            out.putString(STATE_STATUS, status.getText().toString());
        }
    }

    @Override
    protected void onDestroy() {
        if (isFinishing()) {
            worker.abandon();
        }
        super.onDestroy();
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    // =================================================================== jobs

    private static Callable<Object> loadJob(final Repository repo) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                Values v = new Values();
                v.params = repo.params();
                v.evaluateLastN = repo.evaluateLastN();
                return v;
            }
        };
    }

    private static Callable<Object> saveJob(final Repository repo, final ModelParams p,
                                            final int evaluateLastN, final String message) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                repo.setParams(p);
                repo.setEvaluateLastN(evaluateLastN);
                Values v = new Values();
                v.params = repo.params();
                v.evaluateLastN = repo.evaluateLastN();
                v.message = message;
                return v;
            }
        };
    }

    @Override
    public void onJobDone(int job, Object result, Throwable error) {
        if (error != null || !(result instanceof Values)) {
            setStatus(getString(R.string.error_generic, Ui.describe(this, error)));
            loaded = true;                            // allow editing defaults anyway
            updateEnabled();
            return;
        }
        Values v = (Values) result;
        if (job == JOB_LOAD) {
            loaded = true;
            fill(v.params, v.evaluateLastN);
        } else if (job == JOB_SAVE) {
            fill(v.params, v.evaluateLastN);          // shows the clamped values
            setStatus(v.message);
        }
        updateEnabled();
    }

    // =================================================================== form

    private void fill(ModelParams p, int evaluateLastN) {
        setField(editEta, Ui.number(p.eta));
        setField(editShare, Ui.number(p.share));
        setField(editBiasRate, Ui.number(p.biasRate));
        setField(editBiasDecay, Ui.number(p.biasDecay));
        setField(editShortWindow, Integer.toString(p.shortWindow));
        setField(editLongWindow, Integer.toString(p.longWindow));
        setField(editHalfLife, Ui.number(p.halfLife));
        setField(editEvalN, Integer.toString(evaluateLastN));
        switchDayFirst.setChecked(p.dayFirst);
    }

    private static void setField(EditText e, String text) {
        e.setText(text);
        e.setError(null);
    }

    private void updateEnabled() {
        boolean saving = worker.isRunning(JOB_SAVE);
        btnSave.setEnabled(loaded && !saving);
        btnRestore.setEnabled(loaded && !saving);
    }

    private void setStatus(String text) {
        if (text == null || text.length() == 0) {
            status.setVisibility(View.GONE);
        } else {
            status.setText(text);
            status.setVisibility(View.VISIBLE);
        }
    }

    private void save() {
        boolean ok = true;
        double eta = readDouble(editEta);
        double share = readDouble(editShare);
        double biasRate = readDouble(editBiasRate);
        double biasDecay = readDouble(editBiasDecay);
        long shortWindow = readLong(editShortWindow);
        long longWindow = readLong(editLongWindow);
        double halfLife = readDouble(editHalfLife);
        long evalN = readLong(editEvalN);
        if (Double.isNaN(eta) || Double.isNaN(share) || Double.isNaN(biasRate)
                || Double.isNaN(biasDecay) || Double.isNaN(halfLife)
                || shortWindow == Long.MIN_VALUE || longWindow == Long.MIN_VALUE
                || evalN == Long.MIN_VALUE) {
            ok = false;
        }
        if (!ok) {
            setStatus(getString(R.string.settings_fix_errors));
            return;
        }
        ModelParams p = new ModelParams();
        p.eta = eta;
        p.share = share;
        p.biasRate = biasRate;
        p.biasDecay = biasDecay;
        p.shortWindow = saturate(shortWindow);
        p.longWindow = saturate(longWindow);
        p.halfLife = halfLife;
        p.dayFirst = switchDayFirst.isChecked();
        ModelParams raw = p.copy();
        p.clamp();
        int n = Prefs.clampEvaluateLastN(saturate(evalN));

        List<String> adjusted = new ArrayList<String>();
        note(adjusted, R.string.label_eta, raw.eta, p.eta);
        note(adjusted, R.string.label_share, raw.share, p.share);
        note(adjusted, R.string.label_bias_rate, raw.biasRate, p.biasRate);
        note(adjusted, R.string.label_bias_decay, raw.biasDecay, p.biasDecay);
        note(adjusted, R.string.label_short_window, raw.shortWindow, p.shortWindow);
        note(adjusted, R.string.label_long_window, raw.longWindow, p.longWindow);
        note(adjusted, R.string.label_half_life, raw.halfLife, p.halfLife);
        note(adjusted, R.string.label_eval_n, saturate(evalN), n);
        String message;
        if (adjusted.isEmpty()) {
            message = getString(R.string.settings_saved);
        } else {
            StringBuilder sb = new StringBuilder();
            for (String a : adjusted) {
                if (sb.length() > 0) {
                    sb.append("; ");
                }
                sb.append(a);
            }
            message = getString(R.string.settings_adjusted, sb.toString());
        }
        startSave(p, n, message);
    }

    private void startSave(ModelParams p, int evaluateLastN, String message) {
        if (worker.isRunning(JOB_SAVE)) {
            return;
        }
        worker.run(JOB_SAVE, saveJob(Repository.get(this), p, evaluateLastN, message));
        updateEnabled();
    }

    private void note(List<String> out, int labelRes, double raw, double clamped) {
        if (Double.compare(raw, clamped) != 0 && !(raw == 0.0 && clamped == 0.0)) {
            out.add(getString(labelRes) + " → " + Ui.number(clamped));
        }
    }

    /** The field as a number ("," accepted as decimal separator); NaN + field error if invalid. */
    private double readDouble(EditText e) {
        String s = e.getText().toString().trim().replace(',', '.');
        try {
            double v = Double.parseDouble(s);
            if (!Double.isNaN(v) && !Double.isInfinite(v)) {
                e.setError(null);
                return v;
            }
        } catch (NumberFormatException ignored) {
            // fall through
        }
        e.setError(getString(R.string.error_not_number));
        return Double.NaN;
    }

    /** The field as a whole number; Long.MIN_VALUE + field error if invalid. */
    private long readLong(EditText e) {
        String s = e.getText().toString().trim();
        try {
            long v = Long.parseLong(s);
            e.setError(null);
            return v;
        } catch (NumberFormatException ex) {
            boolean digitsOnly = s.length() > 0;
            for (int i = 0; i < s.length(); i++) {
                if (!Character.isDigit(s.charAt(i))) {
                    digitsOnly = false;
                    break;
                }
            }
            if (digitsOnly) {
                e.setError(null);
                return Long.MAX_VALUE;                // too many digits: clamped later
            }
        }
        e.setError(getString(R.string.error_whole_number));
        return Long.MIN_VALUE;
    }

    private static int saturate(long v) {
        return v > Integer.MAX_VALUE ? Integer.MAX_VALUE : (v < Integer.MIN_VALUE ? Integer.MIN_VALUE : (int) v);
    }
}
