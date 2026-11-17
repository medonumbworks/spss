package com.medonumbworks.csprngpredictor.ui;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.ListView;
import android.widget.TextView;

import com.medonumbworks.csprngpredictor.R;
import com.medonumbworks.csprngpredictor.Repository;
import com.medonumbworks.csprngpredictor.core.DateUtil;
import com.medonumbworks.csprngpredictor.core.Draw;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

/** All draws, newest first; view / delete single draws, delete all, export CSV. */
public final class HistoryActivity extends Activity implements Worker.Host {

    private static final int REQ_EXPORT = 1;

    private static final int JOB_LOAD = 1;
    private static final int JOB_DELETE = 2;
    private static final int JOB_DELETE_ALL = 3;
    private static final int JOB_EXPORT = 4;

    private Worker worker;
    private final Ui.Dialogs dialogs = new Ui.Dialogs();

    private TextView header;
    private View progress;
    private ListView list;
    private TextView empty;
    private DrawAdapter adapter;
    /** Number of draws shown (-1 until loaded). */
    private int count = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_history);
        Ui.enableUp(this);
        worker = Worker.restore(getLastNonConfigurationInstance());

        header = (TextView) findViewById(R.id.historyHeader);
        progress = findViewById(R.id.historyProgress);
        list = (ListView) findViewById(R.id.historyList);
        empty = (TextView) findViewById(R.id.historyEmpty);
        adapter = new DrawAdapter(this);
        list.setAdapter(adapter);

        list.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Draw d = adapter.getDraw(position);
                if (d != null) {
                    showDraw(d);
                }
            }
        });
        list.setOnItemLongClickListener(new AdapterView.OnItemLongClickListener() {
            @Override
            public boolean onItemLongClick(AdapterView<?> parent, View view, int position, long id) {
                Draw d = adapter.getDraw(position);
                if (d != null) {
                    confirmDelete(d);
                }
                return true;
            }
        });
    }

    @Override
    protected void onStart() {
        super.onStart();
        worker.attach(this, this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        load();
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

    // =================================================================== jobs

    private void load() {
        progress.setVisibility(View.VISIBLE);
        worker.run(JOB_LOAD, loadJob(Repository.get(this)));
    }

    private static Callable<Object> loadJob(final Repository repo) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                return repo.draws().newestFirst();
            }
        };
    }

    private static Callable<Object> deleteJob(final Repository repo, final Draw d) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                boolean removed = repo.removeDraw(d);
                String problem = repo.takeSaveProblem();
                return problem != null ? problem : Boolean.valueOf(removed);
            }
        };
    }

    private static Callable<Object> deleteAllJob(final Repository repo) {
        return new Callable<Object>() {
            @Override
            public Object call() {
                repo.clearDraws();
                String problem = repo.takeSaveProblem();
                return problem != null ? problem : Boolean.TRUE;
            }
        };
    }

    @Override
    @SuppressWarnings("unchecked")
    public void onJobDone(int job, Object result, Throwable error) {
        if (job == JOB_LOAD) {
            progress.setVisibility(worker.isBusy() ? View.VISIBLE : View.GONE);
            if (error != null || !(result instanceof List)) {
                showError(Ui.describe(this, error));
                return;
            }
            bind((List<Draw>) result);
        } else if (job == JOB_DELETE || job == JOB_DELETE_ALL) {
            if (error != null) {
                showError(Ui.describe(this, error));
            } else if (result instanceof String) {
                Ui.toastLong(this, getString(R.string.save_problem, (String) result));
            } else if (Boolean.TRUE.equals(result)) {
                Ui.toast(this, getString(job == JOB_DELETE ? R.string.deleted : R.string.deleted_all));
            }
            load();
        } else if (job == JOB_EXPORT) {
            progress.setVisibility(worker.isBusy() ? View.VISIBLE : View.GONE);
            if (error != null) {
                Ui.toastLong(this, getString(R.string.export_failed, Ui.describe(this, error)));
            } else {
                Ui.toast(this, getString(R.string.export_done, String.valueOf(result)));
            }
        }
    }

    private void bind(List<Draw> newestFirst) {
        adapter.setDraws(newestFirst);
        count = newestFirst.size();
        if (newestFirst.isEmpty()) {
            header.setText(getResources().getQuantityString(R.plurals.draw_count, 0, 0));
            empty.setVisibility(View.VISIBLE);
        } else {
            String first = newestFirst.get(newestFirst.size() - 1).date;
            String last = newestFirst.get(0).date;
            header.setText(getString(R.string.history_header,
                    getResources().getQuantityString(R.plurals.draw_count, count, count),
                    DateUtil.display(first), DateUtil.display(last)));
            empty.setVisibility(View.GONE);
        }
        invalidateOptionsMenu();
    }

    // =================================================================== dialogs

    private void showDraw(final Draw d) {
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(DateUtil.display(d.date))
                .setMessage(d.numbersString("  "))
                .setPositiveButton(R.string.close, null)
                .setNegativeButton(R.string.delete, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        delete(d);
                    }
                })
                .create();
        dialogs.show(dlg);
    }

    private void confirmDelete(final Draw d) {
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(R.string.delete_confirm_title)
                .setMessage(getString(R.string.delete_confirm_message, DateUtil.display(d.date),
                        d.numbersString("  ")))
                .setPositiveButton(R.string.delete, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        delete(d);
                    }
                })
                .setNegativeButton(R.string.cancel, null)
                .create();
        dialogs.show(dlg);
    }

    private void delete(Draw d) {
        progress.setVisibility(View.VISIBLE);
        worker.run(JOB_DELETE, deleteJob(Repository.get(this), d));
    }

    private void confirmDeleteAll() {
        if (count <= 0) {
            return;
        }
        AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle(R.string.delete_all_title)
                .setMessage(getString(R.string.delete_all_message, count))
                .setPositiveButton(R.string.delete, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        progress.setVisibility(View.VISIBLE);
                        worker.run(JOB_DELETE_ALL, deleteAllJob(Repository.get(HistoryActivity.this)));
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

    // =================================================================== menu

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.history_menu, menu);
        return true;
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        MenuItem del = menu.findItem(R.id.menu_history_delete_all);
        if (del != null) {
            del.setEnabled(count > 0);
        }
        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == android.R.id.home) {
            finish();
            return true;
        } else if (id == R.id.menu_history_add) {
            startActivity(new Intent(this, AddDrawActivity.class));
            return true;
        } else if (id == R.id.menu_history_delete_all) {
            confirmDeleteAll();
            return true;
        } else if (id == R.id.menu_history_export) {
            if (count == 0) {
                Ui.toast(this, getString(R.string.nothing_to_export_draws));
            } else {
                Ui.startCreateDocument(this, "text/csv", "draws-" + DateUtil.today() + ".csv", REQ_EXPORT);
            }
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Uri uri = data == null ? null : data.getData();
        if (requestCode == REQ_EXPORT && resultCode == RESULT_OK && uri != null) {
            progress.setVisibility(View.VISIBLE);
            worker.run(JOB_EXPORT, Ui.exportJob(getApplicationContext(), uri, true));
        }
    }

    // =================================================================== adapter

    /** Row: date (bold), numbers below. */
    static final class DrawAdapter extends BaseAdapter {
        private final LayoutInflater inflater;
        private List<Draw> draws = new ArrayList<Draw>();

        DrawAdapter(Context c) {
            inflater = LayoutInflater.from(c);
        }

        void setDraws(List<Draw> d) {
            draws = d == null ? new ArrayList<Draw>() : d;
            notifyDataSetChanged();
        }

        Draw getDraw(int position) {
            return position >= 0 && position < draws.size() ? draws.get(position) : null;
        }

        @Override
        public int getCount() {
            return draws.size();
        }

        @Override
        public Object getItem(int position) {
            return getDraw(position);
        }

        @Override
        public long getItemId(int position) {
            return position;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            View v = convertView;
            ViewHolder h;
            if (v == null) {
                v = inflater.inflate(R.layout.row_draw, parent, false);
                h = new ViewHolder();
                h.date = (TextView) v.findViewById(R.id.rowDate);
                h.numbers = (TextView) v.findViewById(R.id.rowNumbers);
                v.setTag(h);
            } else {
                h = (ViewHolder) v.getTag();
            }
            Draw d = draws.get(position);
            h.date.setText(DateUtil.display(d.date));
            h.numbers.setText(d.numbersString("  "));
            return v;
        }

        static final class ViewHolder {
            TextView date;
            TextView numbers;
        }
    }
}
