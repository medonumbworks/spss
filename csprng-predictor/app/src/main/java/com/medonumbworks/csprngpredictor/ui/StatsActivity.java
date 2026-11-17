package com.medonumbworks.csprngpredictor.ui;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.medonumbworks.csprngpredictor.R;
import com.medonumbworks.csprngpredictor.Repository;
import com.medonumbworks.csprngpredictor.core.AutoTune;
import com.medonumbworks.csprngpredictor.core.DateUtil;
import com.medonumbworks.csprngpredictor.core.Draw;
import com.medonumbworks.csprngpredictor.core.Game;
import com.medonumbworks.csprngpredictor.core.IssuedPrediction;
import com.medonumbworks.csprngpredictor.core.Model;
import com.medonumbworks.csprngpredictor.core.ModelParams;
import com.medonumbworks.csprngpredictor.core.ReplayResult;
import com.medonumbworks.csprngpredictor.core.Stats;
import com.medonumbworks.csprngpredictor.core.Strategies;
import com.medonumbworks.csprngpredictor.core.Strategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;

/**
 * Accuracy screen: backtest summary + histogram + honesty note, expert weights, the checked
 * (issued) predictions, and the tools (auto-tune, reset, clear log, export).
 */
public final class StatsActivity extends Activity implements Worker.Host {

    private static final int REQ_EXPORT = 1;

    private static final int JOB_LOAD = 1;
    private static final int JOB_TUNE = 2;
    private static final int JOB_CHANGE = 3;   // apply / reset / clear, then reload
    private static final int JOB_EXPORT = 4;

    private static final int ISSUED_ROWS = 30;
    private static final int BAR_WIDTH = 12;

    /** Everything the screen shows, computed in the background. */
    static final class StatsData {
        ReplayResult replay;
        int drawCount;
        int evaluateLastN;
        List<IssuedPrediction> issued;     // copies, issue order
        Stats issuedStats;
        String toast;                      // optional message after a change
    }

    /** Auto-tune result. */
    static final class TuneOutcome {
        boolean tooFew;
        ModelParams base;
        Stats baseStats;
        AutoTune.Candidate best;
    }

    /** Kept across rotation: an auto-tune result whose dialog has not been answered yet. */
    static final class State {
        TuneOutcome pendingTune;
    }

    private Worker worker;
    private State state;
    private final Ui.Dialogs dialogs = new Ui.Dialogs();

    private View progressRow;
    private TextView summary;
    private TextView window;
    private TextView histogramTitle;
    private TextView histogram;
    private TextView honesty;
    private TextView expertNote;
    private LinearLayout expertRows;
    private TextView issuedSummary;
    private LinearLayout issuedRows;
    private Button btnAutoTune;
    private View autoTuneProgress;
    private Button btnReset;
    private Button btnClear;
    private Button btnExport;

    private StatsData data;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_stats);
        Ui.enableUp(this);
        worker = Worker.restore(getLastNonConfigurationInstance());
        Object kept = worker.retained();
        state = kept instanceof State ? (State) kept : new State();
        worker.setRetained(state);

        progressRow = findViewById(R.id.statsProgressRow);
        summary = (TextView) findViewById(R.id.statsSummary);
        window = (TextView) findViewById(R.id.statsWindow);
        histogramTitle = (TextView) findViewById(R.id.statsHistogramTitle);
        histogram = (TextView) findViewById(R.id.statsHistogram);
        honesty = (TextView) findViewById(R.id.statsHonesty);
        expertNote = (TextView) findViewById(R.id.expertNote);
        expertRows = (LinearLayout) findViewById(R.id.expertRows);
        issuedSummary = (TextView) findViewById(R.id.issuedSummary);
        issuedRows = (LinearLayout) findViewById(R.id.issuedRows);
        btnAutoTune = (Button) findViewById(R.id.btnAutoTune);
        autoTuneProgress = findViewById(R.id.autoTuneProgressRow);
        btnReset = (Button) findViewById(R.id.btnResetLearning);
        btnClear = (Button) findViewById(R.id.btnClearIssued);
        btnExport = (Button) findViewById(R.id.btnExportIssued);

        // The honesty note does not depend on the data.
        honesty.setText(getString(R.string.honesty_note) + "\n\n"
                + getString(R.string.random_baseline, Ui.percent(Stats.hypergeometricTail(8), 1)));

        btnAutoTune.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startAutoTune();
            }
        });
        btnReset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                confirmReset();
            }
        });
        btnClear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                confirmClear();
            }
        });
        btnExport.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (data != null && data.issued != null && data.issued.isEmpty()) {
                    Ui.toast(StatsActivity.this, getString(R.string.nothing_to_export_predictions));
                } else {
                    Ui.startCreateDocument(StatsActivity.this, "text/csv",
                            "predictions-" + DateUtil.today() + ".csv", REQ_EXPORT);
                }
            }
        });
        updateBusy();
        if (state.pendingTune != null) {
            showTuneResult(state.pendingTune);
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        worker.attach(this, this);
        updateBusy();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!worker.isRunning(JOB_CHANGE)) {
            load();
        }
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
    protected void onDestroy() {
        dialogs.dismiss();
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

    private void load() {
        worker.run(JOB_LOAD, loadJob(Repository.get(this)));
        updateBusy();
    }

    static StatsData collect(Repository repo) {
        StatsData d = new StatsData();
        d.replay = repo.replay();
        d.drawCount = repo.draws().size();
        d.evaluateLastN = repo.evaluateLastN();
        d.issued = repo.issuedSnapshot();
        d.issuedStats = repo.issued().stats();
        return d;
    }

    private static Callable<Object> loadJob(final Repository repo) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                return collect(repo);
            }
        };
    }

    private static Callable<Object> tuneJob(final Repository repo) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                TuneOutcome o = new TuneOutcome();
                List<Draw> draws = repo.replayDraws();
                if (draws.size() < AutoTune.MIN_DRAWS) {
                    o.tooFew = true;
                    return o;
                }
                o.base = repo.params();
                o.baseStats = repo.replay().stats;     // same draws, window and evaluation
                o.best = AutoTune.tune(draws, o.base, repo.evaluateLastN());
                return o;
            }
        };
    }

    /** kind: 0 = apply params, 1 = reset learning, 2 = clear issued log. */
    private static Callable<Object> changeJob(final Repository repo, final int kind,
                                              final ModelParams params, final String toast) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                if (kind == 0) {
                    repo.setParams(params);
                } else if (kind == 1) {
                    ModelParams p = new ModelParams();
                    p.dayFirst = repo.params().dayFirst;
                    repo.setParams(p);
                } else {
                    repo.clearIssued();
                }
                StatsData d = collect(repo);
                String problem = repo.takeSaveProblem();
                d.toast = problem != null ? problem : toast;
                return d;
            }
        };
    }

    @Override
    public void onJobDone(int job, Object result, Throwable error) {
        updateBusy();
        if (job == JOB_LOAD || job == JOB_CHANGE) {
            if (error != null || !(result instanceof StatsData)) {
                showError(Ui.describe(this, error));
                return;
            }
            data = (StatsData) result;
            bind(data);
            if (data.toast != null) {
                Ui.toast(this, data.toast);
                data.toast = null;
            }
        } else if (job == JOB_TUNE) {
            if (error != null || !(result instanceof TuneOutcome)) {
                showError(Ui.describe(this, error));
                return;
            }
            TuneOutcome o = (TuneOutcome) result;
            if (o.tooFew) {
                showError(getString(R.string.auto_tune_too_few, AutoTune.MIN_DRAWS));
                return;
            }
            state.pendingTune = o;
            showTuneResult(o);
        } else if (job == JOB_EXPORT) {
            if (error != null) {
                Ui.toastLong(this, getString(R.string.export_failed, Ui.describe(this, error)));
            } else {
                Ui.toast(this, getString(R.string.export_done, String.valueOf(result)));
            }
        }
    }

    private void updateBusy() {
        boolean tuning = worker.isRunning(JOB_TUNE);
        boolean changing = worker.isRunning(JOB_CHANGE);
        boolean loading = worker.isRunning(JOB_LOAD);
        progressRow.setVisibility(loading || changing ? View.VISIBLE : View.GONE);
        autoTuneProgress.setVisibility(tuning ? View.VISIBLE : View.GONE);
        btnAutoTune.setEnabled(!tuning && !changing);
        btnReset.setEnabled(!tuning && !changing);
        btnClear.setEnabled(!changing);
    }

    // =================================================================== binding

    private void bind(StatsData d) {
        Stats s = d.replay.stats;

        // 1. Backtest
        summary.setText(s.summaryText());
        int records = d.replay.records.size();
        if (d.evaluateLastN > 0 && records > d.evaluateLastN) {
            window.setText(getString(R.string.stats_window_last, d.evaluateLastN));
        } else {
            window.setText(R.string.stats_window_all);
        }
        if (s.n == 0) {
            histogramTitle.setVisibility(View.GONE);
            histogram.setVisibility(View.GONE);
        } else {
            histogramTitle.setVisibility(View.VISIBLE);
            histogram.setVisibility(View.VISIBLE);
            histogram.setText(histogramText(s));
        }

        // 2. Experts
        bindExperts(d.replay.model);

        // 3. Checked predictions
        bindIssued(d);
    }

    /** "5 hits ████████     31%  random 29.0%" lines for 0..11 hits. */
    private static String histogramText(Stats s) {
        int max = 0;
        for (int c : s.histogram) {
            max = Math.max(max, c);
        }
        StringBuilder sb = new StringBuilder();
        for (int k = 0; k <= Game.PICK; k++) {
            int c = k < s.histogram.length ? s.histogram[k] : 0;
            int bar = max > 0 ? (int) Math.round(BAR_WIDTH * (double) c / max) : 0;
            if (c > 0 && bar == 0) {
                bar = 1;                              // make non-empty buckets visible
            }
            StringBuilder b = new StringBuilder(BAR_WIDTH);
            for (int i = 0; i < BAR_WIDTH; i++) {
                b.append(i < bar ? '█' : ' ');
            }
            if (k > 0) {
                sb.append('\n');
            }
            sb.append(String.format(Locale.US, "%2d hits %s %3.0f%%  random %4.1f%%",
                    k, b.toString(), 100.0 * s.fraction(k), 100.0 * Stats.hypergeometricPmf(k)));
        }
        return sb.toString();
    }

    private void bindExperts(Model model) {
        expertRows.removeAllViews();
        final double[] w = model.weights();
        double[] mean = model.expertMeanHits();
        List<Strategy> strategies = model.strategies();
        List<Integer> order = new ArrayList<Integer>();
        for (int i = 0; i < w.length && i < strategies.size(); i++) {
            order.add(Integer.valueOf(i));
        }
        Collections.sort(order, new Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                int c = Double.compare(w[b.intValue()], w[a.intValue()]);
                return c != 0 ? c : a.compareTo(b);
            }
        });
        if (model.observed() == 0) {
            expertNote.setText(R.string.experts_none);
            expertNote.setVisibility(View.VISIBLE);
        } else if (!order.isEmpty()
                && Strategies.UNIFORM.equals(strategies.get(order.get(0).intValue()).id())) {
            expertNote.setText(R.string.uniform_top_note);
            expertNote.setVisibility(View.VISIBLE);
        } else {
            expertNote.setVisibility(View.GONE);
        }
        LayoutInflater inflater = LayoutInflater.from(this);
        for (int i = 0; i < order.size(); i++) {
            int e = order.get(i).intValue();
            Strategy st = strategies.get(e);
            if (i > 0) {
                expertRows.addView(divider());
            }
            View row = inflater.inflate(R.layout.row_expert, expertRows, false);
            ((TextView) row.findViewById(R.id.expertName)).setText(st.name() + " (" + st.id() + ")");
            ((TextView) row.findViewById(R.id.expertWeight)).setText(Ui.percent(w[e], 1));
            TextView meanView = (TextView) row.findViewById(R.id.expertMean);
            if (model.observed() > 0 && e < mean.length) {
                meanView.setText(getString(R.string.expert_weight_line, Ui.fmt2(mean[e])));
                meanView.setVisibility(View.VISIBLE);
            } else {
                meanView.setVisibility(View.GONE);
            }
            ((TextView) row.findViewById(R.id.expertDesc)).setText(st.description());
            expertRows.addView(row);
        }
    }

    private void bindIssued(StatsData d) {
        issuedRows.removeAllViews();
        StringBuilder head = new StringBuilder();
        if (d.issuedStats.n == 0) {
            head.append(getString(R.string.checked_none));
        } else {
            head.append(d.issuedStats.summaryText());
        }
        int pending = 0;
        for (IssuedPrediction p : d.issued) {
            if (p.isPending()) {
                pending++;
            }
        }
        if (pending > 0) {
            head.append("\n").append(getString(R.string.checked_pending, pending));
        }
        if (d.issued.size() > ISSUED_ROWS) {
            head.append("\n").append(getString(R.string.checked_showing, ISSUED_ROWS, d.issued.size()));
        }
        issuedSummary.setText(head.toString());

        LayoutInflater inflater = LayoutInflater.from(this);
        int shown = 0;
        for (int i = d.issued.size() - 1; i >= 0 && shown < ISSUED_ROWS; i--, shown++) {
            IssuedPrediction p = d.issued.get(i);
            if (shown > 0) {
                issuedRows.addView(divider());
            }
            View row = inflater.inflate(R.layout.row_issued, issuedRows, false);
            ((TextView) row.findViewById(R.id.issuedTarget)).setText(
                    getString(R.string.issued_for, DateUtil.display(p.targetDate)));
            TextView hits = (TextView) row.findViewById(R.id.issuedHits);
            TextView actual = (TextView) row.findViewById(R.id.issuedActual);
            if (p.isPending()) {
                hits.setText(R.string.issued_pending);
                hits.setTextColor(getColor(R.color.text_secondary));
                actual.setVisibility(View.GONE);
            } else {
                hits.setText(getString(R.string.issued_hits, p.hits));
                hits.setTextColor(getColor(p.hits > Game.EXPECTED_HITS ? R.color.text_good : R.color.text_primary));
                actual.setVisibility(View.VISIBLE);
                actual.setText(getString(R.string.issued_actual, DateUtil.display(p.actualDate),
                        Ui.join(p.actual, " ")));
            }
            ((TextView) row.findViewById(R.id.issuedPredicted)).setText(
                    getString(R.string.issued_predicted, Ui.join(p.numbers, " ")));
            ((TextView) row.findViewById(R.id.issuedMeta)).setText(
                    getString(R.string.issued_meta, DateUtil.display(p.issuedAt), p.basedOnCount));
            issuedRows.addView(row);
        }
    }

    private View divider() {
        View v = new View(this);
        v.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                Math.max(1, Ui.dp(this, 1))));
        v.setBackgroundColor(getColor(R.color.divider));
        return v;
    }

    // =================================================================== tools

    private void startAutoTune() {
        if (worker.isRunning(JOB_TUNE)) {
            return;
        }
        if (data != null && data.drawCount < AutoTune.MIN_DRAWS) {
            showError(getString(R.string.auto_tune_too_few, AutoTune.MIN_DRAWS));
            return;
        }
        worker.run(JOB_TUNE, tuneJob(Repository.get(this)));
        updateBusy();
    }

    private void showTuneResult(final TuneOutcome o) {
        Stats best = o.best.stats;
        boolean better = best.meanHits > o.baseStats.meanHits && !o.best.params.equals(o.base);
        AlertDialog.Builder b = new AlertDialog.Builder(this).setTitle(R.string.auto_tune_title);
        if (!better) {
            b.setMessage(getString(R.string.auto_tune_same, Ui.fmt2(o.baseStats.meanHits),
                    Ui.fmt2(best.meanHits), best.n));
            b.setPositiveButton(R.string.ok, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    state.pendingTune = null;
                }
            });
        } else {
            b.setMessage(getString(R.string.auto_tune_message,
                    Ui.tuneSummary(this, o.base), Ui.fmt2(o.baseStats.meanHits),
                    Ui.tuneSummary(this, o.best.params), Ui.fmt2(best.meanHits), best.n));
            b.setPositiveButton(R.string.apply, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    state.pendingTune = null;
                    runChange(0, o.best.params, getString(R.string.auto_tune_applied));
                }
            });
            b.setNegativeButton(R.string.cancel, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    state.pendingTune = null;
                }
            });
        }
        b.setOnCancelListener(new DialogInterface.OnCancelListener() {
            @Override
            public void onCancel(DialogInterface dialog) {
                state.pendingTune = null;
            }
        });
        dialogs.show(b.create());
    }

    private void runChange(int kind, ModelParams params, String toast) {
        worker.run(JOB_CHANGE, changeJob(Repository.get(this), kind, params, toast));
        updateBusy();
    }

    private void confirmReset() {
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(R.string.reset_learning)
                .setMessage(R.string.reset_learning_message)
                .setPositiveButton(R.string.ok, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        runChange(1, null, getString(R.string.reset_done));
                    }
                })
                .setNegativeButton(R.string.cancel, null)
                .create();
        dialogs.show(dlg);
    }

    private void confirmClear() {
        if (data == null || data.issued == null) {
            return;                                   // not loaded yet
        }
        int n = data.issued.size();
        if (n == 0) {
            Ui.toast(this, getString(R.string.checked_none));
            return;
        }
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(R.string.clear_checked)
                .setMessage(getString(R.string.clear_checked_message, n))
                .setPositiveButton(R.string.delete, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        runChange(2, null, getString(R.string.cleared));
                    }
                })
                .setNegativeButton(R.string.cancel, null)
                .create();
        dialogs.show(dlg);
    }

    private void showError(String message) {
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(R.string.error_title)
                .setMessage(message)
                .setPositiveButton(R.string.ok, null)
                .create();
        dialogs.show(dlg);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent result) {
        super.onActivityResult(requestCode, resultCode, result);
        Uri uri = result == null ? null : result.getData();
        if (requestCode == REQ_EXPORT && resultCode == RESULT_OK && uri != null) {
            worker.run(JOB_EXPORT, Ui.exportJob(getApplicationContext(), uri, false));
        }
    }
}
