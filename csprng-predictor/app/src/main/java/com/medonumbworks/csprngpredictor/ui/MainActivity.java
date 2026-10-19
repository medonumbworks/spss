package com.medonumbworks.csprngpredictor.ui;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.medonumbworks.csprngpredictor.R;
import com.medonumbworks.csprngpredictor.Repository;
import com.medonumbworks.csprngpredictor.core.DateUtil;
import com.medonumbworks.csprngpredictor.core.Draw;
import com.medonumbworks.csprngpredictor.core.DrawSet;
import com.medonumbworks.csprngpredictor.core.Game;
import com.medonumbworks.csprngpredictor.core.IssuedPrediction;
import com.medonumbworks.csprngpredictor.core.Prediction;
import com.medonumbworks.csprngpredictor.core.ReplayResult;
import com.medonumbworks.csprngpredictor.core.Stats;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * Launcher screen: next prediction, confidence heat map, last check and model summary.
 * Everything is recomputed in {@link #onResume()} on a background thread.
 */
public final class MainActivity extends Activity implements Worker.Host {

    /** MIME types offered to the file chooser (spec list plus common CSV / text variants). */
    static final String[] IMPORT_MIME_TYPES = {
            "text/csv", "text/comma-separated-values", "text/plain", "text/xml",
            "application/xml", "application/octet-stream",
            "text/*", "text/tab-separated-values", "application/vnd.ms-excel"
    };

    private static final int REQ_PICK_FILE = 1;
    private static final int REQ_EXPORT_DRAWS = 2;
    private static final int REQ_EXPORT_ISSUED = 3;

    private static final int JOB_REFRESH = 1;
    private static final int JOB_EXPORT = 2;

    private Worker worker;
    private final Ui.Dialogs dialogs = new Ui.Dialogs();

    private View progressRow;
    private View cardEmpty;
    private View cardNext;
    private View cardHeat;
    private View cardLast;
    private View cardModel;
    private TextView nextSubtitle;
    private TextView nextNumbers;
    private NumberGridView gridNext;
    private NumberGridView gridHeat;
    private TextView heatList;
    private TextView lastText;
    private NumberGridView gridLast;
    private TextView lastHits;
    private TextView lastLegend;
    private TextView modelBacktest;
    private TextView modelExperts;
    private TextView modelVerdict;

    /** Last data shown (null until the first refresh finished). */
    private MainData data;

    /** Everything the screen shows, computed on a background thread. */
    static final class MainData {
        int drawCount;
        String latestDate = "";
        int issuedCount;
        ReplayResult replay;            // null when there are no draws
        IssuedPrediction lastSettled;   // a copy, or null
        boolean capped;
        String problem;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        worker = Worker.restore(getLastNonConfigurationInstance());

        progressRow = findViewById(R.id.mainProgressRow);
        cardEmpty = findViewById(R.id.cardEmpty);
        cardNext = findViewById(R.id.cardNext);
        cardHeat = findViewById(R.id.cardHeat);
        cardLast = findViewById(R.id.cardLast);
        cardModel = findViewById(R.id.cardModel);
        nextSubtitle = (TextView) findViewById(R.id.nextSubtitle);
        nextNumbers = (TextView) findViewById(R.id.nextNumbers);
        gridNext = (NumberGridView) findViewById(R.id.gridNext);
        gridHeat = (NumberGridView) findViewById(R.id.gridHeat);
        heatList = (TextView) findViewById(R.id.heatList);
        lastText = (TextView) findViewById(R.id.lastText);
        gridLast = (NumberGridView) findViewById(R.id.gridLast);
        lastHits = (TextView) findViewById(R.id.lastHits);
        lastLegend = (TextView) findViewById(R.id.lastLegend);
        modelBacktest = (TextView) findViewById(R.id.modelBacktest);
        modelExperts = (TextView) findViewById(R.id.modelExperts);
        modelVerdict = (TextView) findViewById(R.id.modelVerdict);

        gridNext.setMode(NumberGridView.MODE_DISPLAY);
        gridHeat.setMode(NumberGridView.MODE_HEAT);
        gridLast.setMode(NumberGridView.MODE_DISPLAY);

        ((Button) findViewById(R.id.btnCopy)).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                copyNumbers();
            }
        });
        ((Button) findViewById(R.id.btnAddDraw)).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this, AddDrawActivity.class));
            }
        });
        ((Button) findViewById(R.id.btnImport)).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openFilePicker();
            }
        });
        ((Button) findViewById(R.id.btnStats)).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this, StatsActivity.class));
            }
        });

        // After a rotation show the previous data at once (it is refreshed in onResume), so
        // the ScrollView can restore its scroll position.
        if (worker.retained() instanceof MainData) {
            data = (MainData) worker.retained();
            data.problem = null;
            bind(data);
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        worker.attach(this, this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        refresh();
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

    // =================================================================== refresh

    private void refresh() {
        progressRow.setVisibility(View.VISIBLE);
        worker.run(JOB_REFRESH, refreshJob(Repository.get(this)));
    }

    /** Static: the job must not capture the activity. */
    private static Callable<Object> refreshJob(final Repository repo) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                MainData d = new MainData();
                DrawSet ds = repo.draws();
                d.drawCount = ds.size();
                Draw latest = ds.latest();
                d.latestDate = latest == null ? "" : latest.date;
                if (d.drawCount > 0) {
                    d.replay = repo.currentReplay();   // also logs the shown prediction
                    d.capped = repo.isReplayCapped();
                }
                d.issuedCount = repo.issued().size();
                d.lastSettled = repo.lastSettledCopy();
                d.problem = repo.takeSaveProblem();
                return d;
            }
        };
    }

    @Override
    public void onJobDone(int job, Object result, Throwable error) {
        if (job == JOB_REFRESH) {
            progressRow.setVisibility(View.GONE);
            if (error != null || !(result instanceof MainData)) {
                showError(Ui.describe(this, error));
                return;
            }
            data = (MainData) result;
            worker.setRetained(data);
            bind(data);
        } else if (job == JOB_EXPORT) {
            if (error != null) {
                Ui.toastLong(this, getString(R.string.export_failed, Ui.describe(this, error)));
            } else {
                Ui.toast(this, getString(R.string.export_done, String.valueOf(result)));
            }
        }
    }

    private void bind(MainData d) {
        boolean empty = d.drawCount == 0 || d.replay == null || d.replay.next == null;
        cardEmpty.setVisibility(empty ? View.VISIBLE : View.GONE);
        cardNext.setVisibility(empty ? View.GONE : View.VISIBLE);
        cardHeat.setVisibility(empty ? View.GONE : View.VISIBLE);
        cardLast.setVisibility(empty ? View.GONE : View.VISIBLE);
        cardModel.setVisibility(empty ? View.GONE : View.VISIBLE);
        if (d.problem != null) {
            Ui.toastLong(this, getString(R.string.save_problem, d.problem));
        }
        if (empty) {
            return;
        }
        Prediction next = d.replay.next;

        // Next prediction
        String count = getResources().getQuantityString(R.plurals.draw_count, d.drawCount, d.drawCount);
        nextSubtitle.setText(getString(R.string.next_subtitle, DateUtil.display(next.targetDate),
                count, DateUtil.display(d.latestDate)));
        gridNext.setSelected(Ui.toSet(next.numbers));
        nextNumbers.setText(Ui.join(next.numbers, " · "));

        // Confidence by number
        gridHeat.setScores(next.scores);
        gridHeat.setHighlighted(null);
        heatList.setText(topScores(next));

        // Last check
        IssuedPrediction last = d.lastSettled;
        if (last == null || last.actual == null) {
            lastText.setText(R.string.last_none);
            gridLast.setVisibility(View.GONE);
            lastHits.setVisibility(View.GONE);
            lastLegend.setVisibility(View.GONE);
        } else {
            String date = last.targetDate.length() > 0 ? last.targetDate : last.actualDate;
            StringBuilder sb = new StringBuilder(getString(R.string.last_result,
                    DateUtil.display(date), last.hits));
            if (last.actualDate != null && !last.actualDate.equals(date)) {
                sb.append('\n').append(getString(R.string.last_scored_on, DateUtil.display(last.actualDate)));
            }
            lastText.setText(sb.toString());
            gridLast.setVisibility(View.VISIBLE);
            gridLast.setSelected(Ui.toSet(last.numbers));
            gridLast.setHighlighted(Ui.toSet(last.actual));
            int[] hits = last.hitNumbers();
            lastHits.setVisibility(View.VISIBLE);
            lastHits.setText(hits.length == 0 ? getString(R.string.last_no_hits)
                    : getString(R.string.last_hit_numbers, Ui.join(hits, " · ")));
            lastLegend.setVisibility(View.VISIBLE);
        }

        // Model
        Stats s = d.replay.stats;
        if (s.n == 0) {
            modelBacktest.setText(R.string.model_backtest_none);
        } else {
            modelBacktest.setText(getString(R.string.model_backtest, Ui.fmt2(s.meanHits), s.n));
        }
        modelExperts.setText(getString(R.string.model_experts, topExperts(next, 3)));
        String verdict = s.verdict();
        if (d.capped) {
            verdict = verdict + "\n" + getString(R.string.model_capped, Repository.MAX_REPLAY_DRAWS);
        }
        modelVerdict.setText(verdict);
    }

    /**
     * "7: 6.1% · 3: 5.8% · ..." for the 11 highest scores, each as score / 11 * 100: its share
     * of the total score (the scores sum to 11; all equal = 4.5%). The raw expected-count score
     * is not shown as a chance because a peaky expert can push it above 1.
     */
    private static String topScores(final Prediction p) {
        Integer[] order = new Integer[Game.MAX_NUMBER];
        for (int i = 0; i < order.length; i++) {
            order[i] = Integer.valueOf(i + 1);
        }
        Arrays.sort(order, new Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                int c = Double.compare(p.scoreOf(b.intValue()), p.scoreOf(a.intValue()));
                return c != 0 ? c : a.compareTo(b);
            }
        });
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Game.PICK; i++) {
            if (i > 0) {
                sb.append("  ·  ");
            }
            int n = order[i].intValue();
            sb.append(n).append(": ").append(Ui.percent(p.scoreOf(n) / Game.PICK, 1));
        }
        return sb.toString();
    }

    /** "transition 23% · decay 19% · freq 15%". */
    private static String topExperts(Prediction p, int k) {
        final double[] w = p.expertWeights;
        List<Integer> idx = new ArrayList<Integer>();
        for (int i = 0; i < w.length && i < p.expertIds.length; i++) {
            idx.add(Integer.valueOf(i));
        }
        Collections.sort(idx, new Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                int c = Double.compare(w[b.intValue()], w[a.intValue()]);
                return c != 0 ? c : a.compareTo(b);
            }
        });
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < k && i < idx.size(); i++) {
            int e = idx.get(i).intValue();
            if (i > 0) {
                sb.append(" · ");
            }
            sb.append(p.expertIds[e]).append(' ').append(Ui.percent(w[e], 0));
        }
        return sb.toString();
    }

    private void copyNumbers() {
        if (data == null || data.replay == null || data.replay.next == null) {
            return;
        }
        String text = Ui.join(data.replay.next.numbers, " ");
        ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (cm == null) {
            return;
        }
        try {
            cm.setPrimaryClip(ClipData.newPlainText(getString(R.string.clip_label), text));
        } catch (RuntimeException e) {
            // e.g. SecurityException on some OEM builds: tell the user instead of crashing
            showError(Ui.describe(this, e));
            return;
        }
        // Android 13+ shows its own clipboard confirmation.
        if (Build.VERSION.SDK_INT < 33) {
            Ui.toast(this, getString(R.string.copied_toast, text));
        }
    }

    // =================================================================== menu

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == R.id.menu_import) {
            openFilePicker();
            return true;
        } else if (id == R.id.menu_add_draw) {
            startActivity(new Intent(this, AddDrawActivity.class));
            return true;
        } else if (id == R.id.menu_history) {
            startActivity(new Intent(this, HistoryActivity.class));
            return true;
        } else if (id == R.id.menu_stats) {
            startActivity(new Intent(this, StatsActivity.class));
            return true;
        } else if (id == R.id.menu_settings) {
            startActivity(new Intent(this, SettingsActivity.class));
            return true;
        } else if (id == R.id.menu_export_draws) {
            if (data != null && data.drawCount == 0) {
                Ui.toast(this, getString(R.string.nothing_to_export_draws));
            } else {
                Ui.startCreateDocument(this, "text/csv", "draws-" + DateUtil.today() + ".csv",
                        REQ_EXPORT_DRAWS);
            }
            return true;
        } else if (id == R.id.menu_export_predictions) {
            if (data != null && data.issuedCount == 0) {
                Ui.toast(this, getString(R.string.nothing_to_export_predictions));
            } else {
                Ui.startCreateDocument(this, "text/csv", "predictions-" + DateUtil.today() + ".csv",
                        REQ_EXPORT_ISSUED);
            }
            return true;
        } else if (id == R.id.menu_about) {
            showAbout();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void openFilePicker() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        i.putExtra(Intent.EXTRA_MIME_TYPES, IMPORT_MIME_TYPES);
        try {
            startActivityForResult(i, REQ_PICK_FILE);
            return;
        } catch (ActivityNotFoundException e) {
            // no DocumentsUI: fall back to GET_CONTENT below
        } catch (SecurityException e) {
            // fall back below
        }
        Intent g = new Intent(Intent.ACTION_GET_CONTENT);
        g.addCategory(Intent.CATEGORY_OPENABLE);
        g.setType("*/*");
        g.putExtra(Intent.EXTRA_MIME_TYPES, IMPORT_MIME_TYPES);
        try {
            startActivityForResult(g, REQ_PICK_FILE);
        } catch (ActivityNotFoundException e) {
            Ui.toastLong(this, getString(R.string.no_file_picker));
        } catch (SecurityException e) {
            Ui.toastLong(this, getString(R.string.no_file_picker));
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent result) {
        super.onActivityResult(requestCode, resultCode, result);
        Uri uri = result == null ? null : result.getData();
        if (resultCode != RESULT_OK || uri == null) {
            return;                                   // cancelled picker: nothing to do
        }
        if (requestCode == REQ_PICK_FILE) {
            Intent i = new Intent(this, ImportActivity.class);
            i.setData(uri);
            i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(i);
        } else if (requestCode == REQ_EXPORT_DRAWS || requestCode == REQ_EXPORT_ISSUED) {
            worker.run(JOB_EXPORT, Ui.exportJob(getApplicationContext(), uri,
                    requestCode == REQ_EXPORT_DRAWS));
        }
    }

    // =================================================================== dialogs

    private void showAbout() {
        String version = "";
        try {
            PackageInfo info = getPackageManager().getPackageInfo(getPackageName(), 0);
            version = info.versionName == null ? "" : info.versionName;
        } catch (Exception ignored) {
            // keep ""
        }
        AlertDialog d = new AlertDialog.Builder(this)
                .setTitle(R.string.about_title)
                .setMessage(getString(R.string.about_text, version))
                .setPositiveButton(R.string.ok, null)
                .create();
        dialogs.show(d);
    }

    private void showError(String message) {
        AlertDialog d = new AlertDialog.Builder(this)
                .setTitle(R.string.error_title)
                .setMessage(message)
                .setPositiveButton(R.string.ok, null)
                .create();
        dialogs.show(d);
    }
}
