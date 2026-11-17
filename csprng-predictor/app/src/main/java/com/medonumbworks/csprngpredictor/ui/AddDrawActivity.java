package com.medonumbworks.csprngpredictor.ui;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.DatePickerDialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.DatePicker;
import android.widget.EditText;
import android.widget.TextView;

import com.medonumbworks.csprngpredictor.R;
import com.medonumbworks.csprngpredictor.Repository;
import com.medonumbworks.csprngpredictor.core.DateUtil;
import com.medonumbworks.csprngpredictor.core.Draw;
import com.medonumbworks.csprngpredictor.core.DrawParser;
import com.medonumbworks.csprngpredictor.core.DrawSet;
import com.medonumbworks.csprngpredictor.core.Game;
import com.medonumbworks.csprngpredictor.core.IssuedPrediction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Callable;

/**
 * Manual entry of one draw: date (picker / Today / Yesterday), quick-entry text and a tap grid
 * kept in sync, duplicate and same-date checks, then save + scoring of the pending prediction.
 */
public final class AddDrawActivity extends Activity implements Worker.Host {

    private static final String STATE_DATE = "date";
    private static final String STATE_DATE_PICKED = "datePicked";
    private static final String STATE_SELECTED = "selected";
    private static final String STATE_TEXT = "text";
    private static final String STATE_RESULT = "result";

    private static final int JOB_LOAD = 1;
    private static final int JOB_SAVE = 2;

    private Worker worker;
    private final Ui.Dialogs dialogs = new Ui.Dialogs();

    private Button btnDate;
    private EditText quickEntry;
    private TextView quickError;
    private TextView counter;
    private NumberGridView grid;
    private Button btnSave;
    private Button btnClear;
    private View saveProgressRow;

    /** ISO date of the draw being entered. */
    private String date;
    /** True once the user chose a date (then the loaded default does not override it). */
    private boolean datePicked;
    /** True once the draws are loaded (needed for the duplicate checks). */
    private boolean loaded;
    /** Suppresses the text watcher while the grid rewrites the text. */
    private boolean syncing;
    /** Message of the result dialog once saved (kept across rotation). */
    private String resultMessage;

    /** What the save job returns. */
    static final class SaveOutcome {
        Draw draw;
        Repository.AddResult result;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_draw);
        Ui.enableUp(this);
        worker = Worker.restore(getLastNonConfigurationInstance());

        btnDate = (Button) findViewById(R.id.btnDate);
        quickEntry = (EditText) findViewById(R.id.quickEntry);
        quickError = (TextView) findViewById(R.id.quickError);
        counter = (TextView) findViewById(R.id.selectedCounter);
        grid = (NumberGridView) findViewById(R.id.gridSelect);
        btnSave = (Button) findViewById(R.id.btnSave);
        btnClear = (Button) findViewById(R.id.btnClear);
        saveProgressRow = findViewById(R.id.saveProgressRow);

        grid.setMode(NumberGridView.MODE_SELECT);
        grid.setMaxSelection(Game.PICK);

        String text = "";
        int[] sel = null;
        if (savedInstanceState != null) {
            date = savedInstanceState.getString(STATE_DATE);
            datePicked = savedInstanceState.getBoolean(STATE_DATE_PICKED, false);
            sel = savedInstanceState.getIntArray(STATE_SELECTED);
            text = savedInstanceState.getString(STATE_TEXT, "");
            resultMessage = savedInstanceState.getString(STATE_RESULT);
        }
        if (date == null || !DateUtil.isIso(date)) {
            date = DateUtil.today();
        }

        // The text and the selection are restored by hand (below), in a consistent order.
        quickEntry.setSaveEnabled(false);
        syncing = true;
        quickEntry.setText(text);
        grid.setSelected(Ui.toSet(sel));
        syncing = false;
        updateTextError(text);

        quickEntry.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
            }

            @Override
            public void afterTextChanged(Editable s) {
                if (!syncing) {
                    applyText(s.toString());
                }
            }
        });
        grid.setOnSelectionChangedListener(new NumberGridView.OnSelectionChangedListener() {
            @Override
            public void onSelectionChanged(Set<Integer> selected) {
                syncing = true;
                String t = Ui.join(selected, " ");
                quickEntry.setText(t);
                quickEntry.setSelection(t.length());
                syncing = false;
                quickError.setVisibility(View.GONE);
                updateCounter();
            }
        });
        grid.setOnLimitReachedListener(new NumberGridView.OnLimitReachedListener() {
            @Override
            public void onLimitReached(int maxSelection) {
                Ui.toast(AddDrawActivity.this, getString(R.string.limit_reached));
            }
        });

        btnDate.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickDate();
            }
        });
        findViewById(R.id.btnToday).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setDate(DateUtil.today(), true);
            }
        });
        findViewById(R.id.btnYesterday).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setDate(DateUtil.addDays(DateUtil.today(), -1), true);
            }
        });
        btnClear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                syncing = true;
                quickEntry.setText("");
                grid.setSelected(null);
                syncing = false;
                quickError.setVisibility(View.GONE);
                updateCounter();
            }
        });
        btnSave.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onSaveClicked();
            }
        });

        updateDateButton();
        setBusy(worker.isRunning(JOB_SAVE));          // a save may still run after a rotation
        if (resultMessage != null) {
            showResult(resultMessage);                // saved before a rotation
        } else {
            worker.run(JOB_LOAD, loadJob(Repository.get(this)));
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        worker.attach(this, this);
        updateCounter();
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
        out.putString(STATE_DATE, date);
        out.putBoolean(STATE_DATE_PICKED, datePicked);
        out.putIntArray(STATE_SELECTED, Ui.toArray(grid.getSelectedNumbers()));
        out.putString(STATE_TEXT, quickEntry.getText().toString());
        if (resultMessage != null) {
            out.putString(STATE_RESULT, resultMessage);
        }
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

    /** Loads the draws (off the main thread) and returns the latest date ("" if none). */
    private static Callable<Object> loadJob(final Repository repo) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                Draw latest = repo.draws().latest();
                return latest == null ? "" : latest.date;
            }
        };
    }

    private static Callable<Object> saveJob(final Repository repo, final List<Draw> replace, final Draw d) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                SaveOutcome o = new SaveOutcome();
                o.draw = d;
                if (replace.isEmpty()) {
                    o.result = repo.addDrawsDetailed(Collections.singletonList(d));
                } else {
                    o.result = repo.replaceDraws(replace, d);
                }
                repo.settleIssued();                 // normally a no-op: adding already settled
                return o;
            }
        };
    }

    @Override
    public void onJobDone(int job, Object result, Throwable error) {
        if (job == JOB_LOAD) {
            loaded = true;
            if (error == null && result instanceof String && !datePicked) {
                setDate(defaultDate((String) result), false);
            }
            updateCounter();
        } else if (job == JOB_SAVE) {
            setBusy(false);
            if (error != null || !(result instanceof SaveOutcome)) {
                showError(Ui.describe(this, error));
                return;
            }
            SaveOutcome o = (SaveOutcome) result;
            resultMessage = buildResultMessage(o);
            setResult(RESULT_OK);
            showResult(resultMessage);
        }
    }

    /** Latest draw date + 1 if that is not in the future, else today. */
    private static String defaultDate(String latest) {
        String today = DateUtil.today();
        if (latest == null || !DateUtil.isIso(latest)) {
            return today;
        }
        String next = DateUtil.addDays(latest, 1);
        return DateUtil.compare(next, today) <= 0 ? next : today;
    }

    // =================================================================== date

    private void setDate(String iso, boolean byUser) {
        if (iso == null || !DateUtil.isIso(iso)) {
            return;
        }
        date = iso;
        if (byUser) {
            datePicked = true;
        }
        updateDateButton();
    }

    private void updateDateButton() {
        btnDate.setText(DateUtil.display(date));
    }

    private void pickDate() {
        int y = Integer.parseInt(date.substring(0, 4));
        int m = Integer.parseInt(date.substring(5, 7));
        int d = Integer.parseInt(date.substring(8, 10));
        DatePickerDialog dlg = new DatePickerDialog(this, new DatePickerDialog.OnDateSetListener() {
            @Override
            public void onDateSet(DatePicker view, int year, int month, int dayOfMonth) {
                setDate(String.format(Locale.US, "%04d-%02d-%02d", year, month + 1, dayOfMonth), true);
            }
        }, y, m - 1, d);
        dialogs.show(dlg);
    }

    // =================================================================== numbers

    /** Quick entry changed: select the first 11 distinct valid numbers. */
    private void applyText(String text) {
        int[] nums = DrawParser.extractNumbers(text);
        LinkedHashSet<Integer> distinct = new LinkedHashSet<Integer>();
        for (int n : nums) {
            distinct.add(n);
        }
        TreeSet<Integer> sel = new TreeSet<Integer>();
        for (Integer n : distinct) {
            if (sel.size() >= Game.PICK) {
                break;
            }
            sel.add(n);
        }
        grid.setSelected(sel);
        updateTextError(text);
        updateCounter();
    }

    private void updateTextError(String text) {
        int[] nums = DrawParser.extractNumbers(text);
        LinkedHashSet<Integer> distinct = new LinkedHashSet<Integer>();
        for (int n : nums) {
            distinct.add(n);
        }
        String error = null;
        if (distinct.size() > Game.PICK) {
            error = getString(R.string.error_too_many);
        } else if (hasOutOfRange(text)) {
            error = getString(R.string.error_out_of_range);
        }
        if (error == null) {
            quickError.setVisibility(View.GONE);
        } else {
            quickError.setText(error);
            quickError.setVisibility(View.VISIBLE);
        }
    }

    /** True when the text contains a stand-alone integer outside 1..22 (e.g. "0", "23"). */
    private static boolean hasOutOfRange(String text) {
        int len = text.length();
        int i = 0;
        while (i < len) {
            char c = text.charAt(i);
            if (c >= '0' && c <= '9') {
                int j = i;
                long v = 0;
                while (j < len && text.charAt(j) >= '0' && text.charAt(j) <= '9') {
                    if (v < 1000) {
                        v = v * 10 + (text.charAt(j) - '0');
                    }
                    j++;
                }
                boolean glued = (i > 0 && Character.isLetter(text.charAt(i - 1)))
                        || (j < len && Character.isLetter(text.charAt(j)));
                if (!glued && !Game.isValidNumber((int) v)) {
                    return true;
                }
                i = j;
            } else {
                i++;
            }
        }
        return false;
    }

    private void updateCounter() {
        int n = grid.getSelectedNumbers().size();
        counter.setText(getString(R.string.selected_counter, n));
        counter.setTextColor(getColor(n == Game.PICK ? R.color.text_good : R.color.text_primary));
        boolean busy = worker != null && worker.isRunning(JOB_SAVE);
        btnSave.setEnabled(n == Game.PICK && loaded && !busy && resultMessage == null);
        if (busy) {
            saveProgressRow.setVisibility(View.VISIBLE);
        }
    }

    private void setBusy(boolean busy) {
        saveProgressRow.setVisibility(busy ? View.VISIBLE : View.GONE);
        grid.setEnabled(!busy);
        quickEntry.setEnabled(!busy);
        btnClear.setEnabled(!busy);
        updateCounter();
    }

    // =================================================================== save

    private void onSaveClicked() {
        int[] nums = Ui.toArray(grid.getSelectedNumbers());
        if (nums.length != Game.PICK || !loaded) {
            return;
        }
        final Draw d;
        try {
            d = new Draw(date, nums);
        } catch (IllegalArgumentException e) {
            showError(Ui.describe(this, e));
            return;
        }
        // The draws are loaded (JOB_LOAD finished), so these calls are cheap.
        DrawSet ds = Repository.get(this).draws();
        if (ds.contains(d)) {
            Ui.toast(this, getString(R.string.draw_already_saved));
            return;
        }
        final List<Draw> sameDay = ds.onDate(d.date);
        if (sameDay.isEmpty()) {
            startSave(new ArrayList<Draw>(), d);
            return;
        }
        StringBuilder existing = new StringBuilder();
        for (Draw o : sameDay) {
            if (existing.length() > 0) {
                existing.append("; ");
            }
            existing.append(o.numbersString(" "));
        }
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(R.string.same_date_title)
                .setMessage(getString(R.string.same_date_message, DateUtil.display(d.date),
                        existing.toString()))
                .setPositiveButton(R.string.replace, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        startSave(sameDay, d);
                    }
                })
                .setNeutralButton(R.string.add_another, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        startSave(new ArrayList<Draw>(), d);
                    }
                })
                .setNegativeButton(R.string.cancel, null)
                .create();
        dialogs.show(dlg);
    }

    private void startSave(List<Draw> replace, Draw d) {
        if (worker.isRunning(JOB_SAVE)) {
            return;
        }
        worker.run(JOB_SAVE, saveJob(Repository.get(this), new ArrayList<Draw>(replace), d));
        setBusy(true);
    }

    private String buildResultMessage(SaveOutcome o) {
        StringBuilder sb = new StringBuilder();
        Repository.AddResult r = o.result;
        if (r.added == 0) {
            sb.append(getString(R.string.not_saved_duplicate));
        } else {
            sb.append(getString(R.string.saved_message, DateUtil.display(o.draw.date),
                    o.draw.numbersString(" ")));
        }
        for (IssuedPrediction p : r.settled) {
            String target = p.targetDate.length() > 0 ? p.targetDate : p.actualDate;
            sb.append("\n\n").append(getString(R.string.scored_message, DateUtil.display(target), p.hits));
            int[] hits = p.hitNumbers();
            sb.append('\n').append(hits.length == 0 ? getString(R.string.scored_no_hits)
                    : getString(R.string.scored_hits, Ui.join(hits, " ")));
        }
        if (r.added > 0) {
            sb.append("\n\n").append(getString(R.string.saved_next_hint));
        }
        if (r.saveProblem != null) {
            sb.append("\n\n").append(getString(R.string.save_problem, r.saveProblem));
        }
        return sb.toString();
    }

    private void showResult(String message) {
        updateCounter();
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(R.string.saved_title)
                .setMessage(message)
                .setPositiveButton(R.string.ok, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        finish();
                    }
                })
                .setOnCancelListener(new DialogInterface.OnCancelListener() {
                    @Override
                    public void onCancel(DialogInterface dialog) {
                        finish();
                    }
                })
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
}
