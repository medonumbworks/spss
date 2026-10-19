package com.medonumbworks.csprngpredictor.ui;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.DatePickerDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.DatePicker;
import android.widget.ListView;
import android.widget.RadioGroup;
import android.widget.Switch;
import android.widget.TextView;

import com.medonumbworks.csprngpredictor.R;
import com.medonumbworks.csprngpredictor.Repository;
import com.medonumbworks.csprngpredictor.core.DateUtil;
import com.medonumbworks.csprngpredictor.core.Draw;
import com.medonumbworks.csprngpredictor.core.DrawParser;
import com.medonumbworks.csprngpredictor.core.DrawSet;
import com.medonumbworks.csprngpredictor.core.IssuedPrediction;
import com.medonumbworks.csprngpredictor.core.ParseResult;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;

/**
 * Import preview: reads the picked document (SAF) on a background thread, parses it, shows a
 * summary, warnings and the first 50 draws, lets the user resolve undated rows and ambiguous
 * dates, and imports on confirmation. Nothing is saved before "Import".
 */
public final class ImportActivity extends Activity implements Worker.Host {

    /** Largest accepted file (25 MB). */
    static final int MAX_BYTES = 25 * 1024 * 1024;
    private static final int PREVIEW_MAX = 50;
    private static final int INLINE_WARNINGS = 5;

    private static final int JOB_READ = 1;
    private static final int JOB_IMPORT = 2;

    private static final String STATE_OLDEST_FIRST = "oldestFirst";
    private static final String STATE_END_DATE = "endDate";
    private static final String STATE_DAY_FIRST = "dayFirst";
    private static final String STATE_RESULT = "result";
    private static final String STATE_ERROR = "error";

    /** The file was larger than {@link #MAX_BYTES}. */
    static final class TooLargeException extends IOException {
        private static final long serialVersionUID = 1L;

        TooLargeException() {
            super("file larger than 25 MB");
        }
    }

    /** Parsed file, computed once in the background. */
    static final class ReadResult {
        String name = "";
        boolean dayFirstDefault = true;
        ParseResult primary;          // parsed with dayFirstDefault
        ParseResult alternative;      // parsed with !dayFirstDefault; null when identical

        ParseResult forDayFirst(boolean dayFirst) {
            return (dayFirst == dayFirstDefault || alternative == null) ? primary : alternative;
        }

        boolean ambiguous() {
            return alternative != null;
        }
    }

    /** Screen state kept across rotation together with the worker. */
    static final class State {
        ReadResult read;
        boolean oldestFirst = true;
        String endDate;
        Boolean dayFirst;             // null = the file's default (Settings)
        String resultMessage;         // import finished
        String errorMessage;          // reading failed
    }

    /** What the import job returns. */
    static final class ImportOutcome {
        Repository.AddResult result;
        int extraSettled;
    }

    /** One preview row. */
    static final class PreviewRow {
        final String date;
        final String numbers;
        final boolean assigned;
        final boolean exists;

        PreviewRow(String date, String numbers, boolean assigned, boolean exists) {
            this.date = date;
            this.numbers = numbers;
            this.assigned = assigned;
            this.exists = exists;
        }
    }

    private Worker worker;
    private State state;
    private final Ui.Dialogs dialogs = new Ui.Dialogs();

    private View loading;
    private TextView loadingText;
    private ListView list;
    private PreviewAdapter adapter;
    private Button btnImport;
    private TextView fileName;
    private TextView summary;
    private TextView note;
    private TextView warnings;
    private Button btnAllWarnings;
    private View dayFirstCard;
    private Switch switchDayFirst;
    private View undatedCard;
    private TextView undatedTitle;
    private RadioGroup radioOrder;
    private Button btnEndDate;
    private TextView previewTitle;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_import);
        Ui.enableUp(this);
        worker = Worker.restore(getLastNonConfigurationInstance());
        Object kept = worker.retained();
        state = kept instanceof State ? (State) kept : new State();
        worker.setRetained(state);
        if (savedInstanceState != null && kept == null) {
            // Recreated after the process was killed: the parsed data is gone (the file is
            // read again), but the options and any final message are restored.
            state.oldestFirst = savedInstanceState.getBoolean(STATE_OLDEST_FIRST, true);
            state.endDate = savedInstanceState.getString(STATE_END_DATE);
            if (savedInstanceState.containsKey(STATE_DAY_FIRST)) {
                state.dayFirst = Boolean.valueOf(savedInstanceState.getBoolean(STATE_DAY_FIRST));
            }
            state.resultMessage = savedInstanceState.getString(STATE_RESULT);
            state.errorMessage = savedInstanceState.getString(STATE_ERROR);
        }
        if (state.endDate == null || !DateUtil.isIso(state.endDate)) {
            state.endDate = DateUtil.today();
        }

        loading = findViewById(R.id.importLoading);
        loadingText = (TextView) findViewById(R.id.importLoadingText);
        list = (ListView) findViewById(R.id.importList);
        btnImport = (Button) findViewById(R.id.btnImportConfirm);

        View header = LayoutInflater.from(this).inflate(R.layout.import_header, list, false);
        fileName = (TextView) header.findViewById(R.id.importFileName);
        summary = (TextView) header.findViewById(R.id.importSummary);
        note = (TextView) header.findViewById(R.id.importNote);
        warnings = (TextView) header.findViewById(R.id.importWarnings);
        btnAllWarnings = (Button) header.findViewById(R.id.btnAllWarnings);
        dayFirstCard = header.findViewById(R.id.importDayFirstCard);
        switchDayFirst = (Switch) header.findViewById(R.id.switchDayFirst);
        undatedCard = header.findViewById(R.id.importUndatedCard);
        undatedTitle = (TextView) header.findViewById(R.id.importUndatedTitle);
        radioOrder = (RadioGroup) header.findViewById(R.id.radioOrder);
        btnEndDate = (Button) header.findViewById(R.id.btnEndDate);
        previewTitle = (TextView) header.findViewById(R.id.importPreviewTitle);
        // State of these widgets is managed by hand (they live in a list header).
        switchDayFirst.setSaveEnabled(false);
        radioOrder.setSaveEnabled(false);
        header.findViewById(R.id.radioOldestFirst).setSaveEnabled(false);
        header.findViewById(R.id.radioNewestFirst).setSaveEnabled(false);

        list.addHeaderView(header, null, false);
        adapter = new PreviewAdapter(this);
        list.setAdapter(adapter);

        findViewById(R.id.btnImportCancel).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });
        btnImport.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startImport();
            }
        });
        btnAllWarnings.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showAllWarnings();
            }
        });
        btnEndDate.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickEndDate();
            }
        });

        if (state.resultMessage != null) {
            if (state.read != null) {
                bind();
            }
            showResult(state.resultMessage);
        } else if (state.errorMessage != null) {
            showLoading(R.string.reading_file);
            showFatal(state.errorMessage);
        } else if (worker.isRunning(JOB_IMPORT)) {
            showLoading(R.string.importing);
        } else if (state.read != null) {
            bind();
        } else if (worker.isRunning(JOB_READ)) {
            showLoading(R.string.reading_file);
        } else {
            Uri uri = getIntent() == null ? null : getIntent().getData();
            if (uri == null) {
                showLoading(R.string.reading_file);
                state.errorMessage = getString(R.string.error_no_uri);
                showFatal(state.errorMessage);
            } else {
                showLoading(R.string.reading_file);
                worker.run(JOB_READ, readJob(getApplicationContext(), uri));
            }
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        worker.attach(this, this);
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
        out.putBoolean(STATE_OLDEST_FIRST, state.oldestFirst);
        out.putString(STATE_END_DATE, state.endDate);
        if (state.dayFirst != null) {
            out.putBoolean(STATE_DAY_FIRST, state.dayFirst.booleanValue());
        }
        if (state.resultMessage != null) {
            out.putString(STATE_RESULT, state.resultMessage);
        }
        if (state.errorMessage != null) {
            out.putString(STATE_ERROR, state.errorMessage);
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

    // =================================================================== background jobs

    /** Reads and parses the document. Static: does not capture the activity. */
    private static Callable<Object> readJob(final Context app, final Uri uri) {
        return new Callable<Object>() {
            @Override
            public Object call() throws Exception {
                Repository repo = Repository.get(app);
                repo.draws();                        // load now: the preview checks duplicates
                boolean dayFirst = repo.params().dayFirst;
                String name = Ui.displayName(app, uri);
                long size = Ui.size(app, uri);
                if (size > MAX_BYTES) {
                    throw new TooLargeException();
                }
                String content = readText(app, uri, size);
                ReadResult r = new ReadResult();
                r.name = name;
                r.dayFirstDefault = dayFirst;
                r.primary = DrawParser.parse(content, name, dayFirst);
                // Parse once more with the other day/month order: if anything differs, the
                // file has ambiguous dates and the user gets the "Day-first dates" switch.
                ParseResult other = DrawParser.parse(content, name, !dayFirst);
                if (differs(r.primary, other)) {
                    r.alternative = other;
                }
                return r;
            }
        };
    }

    /** Reads at most {@link #MAX_BYTES}; UTF-8 (UTF-16 when it starts with a UTF-16 BOM). */
    static String readText(Context c, Uri uri, long sizeHint) throws IOException {
        InputStream in = c.getContentResolver().openInputStream(uri);
        if (in == null) {
            throw new IOException("the file could not be opened");
        }
        try {
            int initial = sizeHint > 0 ? (int) Math.min(sizeHint + 1, (long) MAX_BYTES + 1)
                    : 64 * 1024;
            byte[] buf = new byte[Math.max(initial, 1024)];
            int len = 0;
            while (true) {
                if (len == buf.length) {
                    int grown = (int) Math.min((long) buf.length * 2, (long) MAX_BYTES + 1);
                    buf = Arrays.copyOf(buf, grown);
                }
                int n = in.read(buf, len, buf.length - len);
                if (n < 0) {
                    break;
                }
                len += n;
                if (len > MAX_BYTES) {
                    throw new TooLargeException();
                }
            }
            if (len >= 2 && buf[0] == (byte) 0xFF && buf[1] == (byte) 0xFE) {
                return new String(buf, 2, len - 2, "UTF-16LE");
            }
            if (len >= 2 && buf[0] == (byte) 0xFE && buf[1] == (byte) 0xFF) {
                return new String(buf, 2, len - 2, "UTF-16BE");
            }
            if (len >= 3 && buf[0] == (byte) 0xEF && buf[1] == (byte) 0xBB && buf[2] == (byte) 0xBF) {
                return new String(buf, 3, len - 3, "UTF-8");
            }
            return new String(buf, 0, len, "UTF-8");
        } finally {
            try {
                in.close();
            } catch (IOException ignored) {
                // nothing to do
            }
        }
    }

    /** True when the two parses found different draws. */
    private static boolean differs(ParseResult a, ParseResult b) {
        if (a.dated.size() != b.dated.size() || a.undated.size() != b.undated.size()
                || a.skippedLines != b.skippedLines) {
            return true;
        }
        for (int i = 0; i < a.dated.size(); i++) {
            if (!a.dated.get(i).equals(b.dated.get(i))) {
                return true;
            }
        }
        return false;
    }

    private static Callable<Object> importJob(final Repository repo, final ParseResult p,
                                              final boolean oldestFirst, final String endDate) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                List<Draw> all = new ArrayList<Draw>(p.dated.size() + p.undated.size());
                all.addAll(p.dated);
                if (!p.undated.isEmpty()) {
                    all.addAll(DrawParser.assignDates(p.undated, endDate, oldestFirst));
                }
                ImportOutcome o = new ImportOutcome();
                o.result = repo.addDrawsDetailed(all);
                o.extraSettled = repo.settleIssued();  // normally 0: adding already settled
                return o;
            }
        };
    }

    @Override
    public void onJobDone(int job, Object result, Throwable error) {
        if (job == JOB_READ) {
            if (error != null || !(result instanceof ReadResult)) {
                state.errorMessage = readErrorMessage(error);
                showFatal(state.errorMessage);
                return;
            }
            state.read = (ReadResult) result;
            bind();
        } else if (job == JOB_IMPORT) {
            if (error != null || !(result instanceof ImportOutcome)) {
                bind();
                showError(getString(R.string.import_failed, Ui.describe(this, error)));
                return;
            }
            state.resultMessage = buildResultMessage((ImportOutcome) result);
            setResult(RESULT_OK);
            bind();
            showResult(state.resultMessage);
        }
    }

    private String readErrorMessage(Throwable error) {
        if (error instanceof TooLargeException) {
            return getString(R.string.error_too_large);
        }
        if (error instanceof SecurityException) {
            return getString(R.string.error_permission);
        }
        if (error instanceof OutOfMemoryError) {
            return getString(R.string.error_memory);
        }
        return getString(R.string.error_read, Ui.describe(this, error));
    }

    // =================================================================== binding

    private void showLoading(int textRes) {
        loadingText.setText(textRes);
        loading.setVisibility(View.VISIBLE);
        list.setVisibility(View.GONE);
        btnImport.setEnabled(false);
    }

    private boolean currentDayFirst() {
        if (state.dayFirst != null) {
            return state.dayFirst.booleanValue();
        }
        return state.read == null || state.read.dayFirstDefault;
    }

    private ParseResult current() {
        return state.read == null ? null : state.read.forDayFirst(currentDayFirst());
    }

    private void bind() {
        ParseResult p = current();
        if (p == null) {
            return;
        }
        loading.setVisibility(View.GONE);
        list.setVisibility(View.VISIBLE);

        String name = state.read.name;
        fileName.setText(name == null || name.length() == 0 ? getString(R.string.file_unknown_name) : name);
        summary.setText(getString(R.string.import_summary, p.dated.size(), p.undated.size(), p.skippedLines));
        note.setText(p.isEmpty() ? R.string.import_nothing : R.string.import_duplicates_note);

        // Warnings: the first few inline, all of them in a dialog.
        if (p.warnings.isEmpty()) {
            warnings.setText(R.string.warnings_none);
            btnAllWarnings.setVisibility(View.GONE);
        } else {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < p.warnings.size() && i < INLINE_WARNINGS; i++) {
                if (i > 0) {
                    sb.append('\n');
                }
                sb.append(p.warnings.get(i));
            }
            warnings.setText(sb.toString());
            btnAllWarnings.setVisibility(p.warnings.size() > INLINE_WARNINGS ? View.VISIBLE : View.GONE);
        }

        // Day-first switch only when the file's dates can be read both ways.
        boolean ambiguous = state.read.ambiguous();
        dayFirstCard.setVisibility(ambiguous ? View.VISIBLE : View.GONE);
        switchDayFirst.setOnCheckedChangeListener(null);
        switchDayFirst.setChecked(currentDayFirst());
        switchDayFirst.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                state.dayFirst = Boolean.valueOf(isChecked);
                bind();
            }
        });

        // Undated rows
        int undated = p.undated.size();
        undatedCard.setVisibility(undated > 0 ? View.VISIBLE : View.GONE);
        undatedTitle.setText(getString(R.string.undated_title, undated));
        radioOrder.setOnCheckedChangeListener(null);
        radioOrder.check(state.oldestFirst ? R.id.radioOldestFirst : R.id.radioNewestFirst);
        radioOrder.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                state.oldestFirst = checkedId != R.id.radioNewestFirst;
                bind();
            }
        });
        btnEndDate.setText(getString(state.oldestFirst ? R.string.last_row_date : R.string.first_row_date,
                DateUtil.display(state.endDate)));

        // Preview
        List<PreviewRow> rows = buildPreview(p);
        adapter.setRows(rows);
        previewTitle.setText(getString(R.string.preview_title, rows.size(), p.drawCount()));
        previewTitle.setVisibility(rows.isEmpty() ? View.GONE : View.VISIBLE);

        boolean busy = worker.isRunning(JOB_IMPORT);
        btnImport.setEnabled(!p.isEmpty() && !busy && state.resultMessage == null);
    }

    /** First {@link #PREVIEW_MAX} draws: dated ones in file order, then undated with their dates. */
    private List<PreviewRow> buildPreview(ParseResult p) {
        List<PreviewRow> rows = new ArrayList<PreviewRow>();
        DrawSet existing = Repository.get(this).draws();   // loaded by the read job
        for (Draw d : p.dated) {
            if (rows.size() >= PREVIEW_MAX) {
                break;
            }
            rows.add(new PreviewRow(DateUtil.display(d.date), d.numbersString("  "), false,
                    existing.contains(d)));
        }
        int n = p.undated.size();
        for (int k = 0; k < n && rows.size() < PREVIEW_MAX; k++) {
            int[] nums = DrawParser.asDraw(p.undated.get(k));
            if (nums == null) {
                continue;
            }
            // Same positional rule as DrawParser.assignDates.
            int back = state.oldestFirst ? (n - 1 - k) : k;
            String iso;
            boolean exists = false;
            try {
                iso = DateUtil.addDays(state.endDate, -back);
                exists = existing.contains(new Draw(iso, nums));
                iso = DateUtil.display(iso);
            } catch (IllegalArgumentException e) {
                iso = "?";
            }
            rows.add(new PreviewRow(iso, Ui.join(nums, "  "), true, exists));
        }
        return rows;
    }

    // =================================================================== actions

    private void startImport() {
        ParseResult p = current();
        if (p == null || p.isEmpty() || worker.isRunning(JOB_IMPORT) || state.resultMessage != null) {
            return;
        }
        showLoading(R.string.importing);
        worker.run(JOB_IMPORT, importJob(Repository.get(this), p, state.oldestFirst, state.endDate));
    }

    private String buildResultMessage(ImportOutcome o) {
        Repository.AddResult r = o.result;
        StringBuilder sb = new StringBuilder(getString(R.string.import_result, r.added, r.duplicates));
        if (!r.settled.isEmpty()) {
            sb.append("\n\n").append(getString(R.string.import_settled, r.settled.size()));
            int shown = 0;
            for (IssuedPrediction ip : r.settled) {
                if (shown++ >= 10) {
                    break;
                }
                String target = ip.targetDate.length() > 0 ? ip.targetDate : ip.actualDate;
                sb.append('\n').append(getString(R.string.import_settled_line,
                        DateUtil.display(target), ip.hits));
            }
        }
        if (r.saveProblem != null) {
            sb.append("\n\n").append(getString(R.string.save_problem, r.saveProblem));
        }
        return sb.toString();
    }

    private void pickEndDate() {
        String d = state.endDate;
        int y = Integer.parseInt(d.substring(0, 4));
        int m = Integer.parseInt(d.substring(5, 7));
        int day = Integer.parseInt(d.substring(8, 10));
        DatePickerDialog dlg = new DatePickerDialog(this, new DatePickerDialog.OnDateSetListener() {
            @Override
            public void onDateSet(DatePicker view, int year, int month, int dayOfMonth) {
                state.endDate = String.format(Locale.US, "%04d-%02d-%02d", year, month + 1, dayOfMonth);
                bind();
            }
        }, y, m - 1, day);
        dialogs.show(dlg);
    }

    private void showAllWarnings() {
        ParseResult p = current();
        if (p == null) {
            return;
        }
        StringBuilder sb = new StringBuilder();
        for (String w : p.warnings) {
            if (sb.length() > 0) {
                sb.append('\n');
            }
            sb.append(w);
        }
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(R.string.warnings_title)
                .setMessage(sb.toString())
                .setPositiveButton(R.string.close, null)
                .create();
        dialogs.show(dlg);
    }

    private void showResult(String message) {
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(R.string.import_result_title)
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

    /** Reading failed: explain and leave the screen. */
    private void showFatal(String message) {
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(R.string.error_title)
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

    // =================================================================== adapter

    /** Preview rows: "date" (bold) and the numbers, plus a note for assigned / existing rows. */
    static final class PreviewAdapter extends BaseAdapter {
        private final LayoutInflater inflater;
        private final String assignedNote;
        private final String existsNote;
        private List<PreviewRow> rows = new ArrayList<PreviewRow>();

        PreviewAdapter(Context c) {
            inflater = LayoutInflater.from(c);
            assignedNote = c.getString(R.string.preview_assigned);
            existsNote = c.getString(R.string.already_saved);
        }

        void setRows(List<PreviewRow> r) {
            rows = r == null ? new ArrayList<PreviewRow>() : r;
            notifyDataSetChanged();
        }

        @Override
        public int getCount() {
            return rows.size();
        }

        @Override
        public Object getItem(int position) {
            return rows.get(position);
        }

        @Override
        public long getItemId(int position) {
            return position;
        }

        @Override
        public boolean areAllItemsEnabled() {
            return false;
        }

        @Override
        public boolean isEnabled(int position) {
            return false;                             // preview only
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            View v = convertView;
            if (v == null) {
                v = inflater.inflate(R.layout.row_draw, parent, false);
            }
            PreviewRow r = rows.get(position);
            ((TextView) v.findViewById(R.id.rowDate)).setText(r.date);
            ((TextView) v.findViewById(R.id.rowNumbers)).setText(r.numbers);
            TextView noteView = (TextView) v.findViewById(R.id.rowNote);
            String n = null;
            if (r.assigned && r.exists) {
                n = assignedNote + " · " + existsNote;
            } else if (r.assigned) {
                n = assignedNote;
            } else if (r.exists) {
                n = existsNote;
            }
            noteView.setVisibility(n == null ? View.GONE : View.VISIBLE);
            noteView.setText(n == null ? "" : n);
            return v;
        }
    }
}
