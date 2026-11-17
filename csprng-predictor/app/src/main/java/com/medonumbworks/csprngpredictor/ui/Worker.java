package com.medonumbworks.csprngpredictor.ui;

import android.app.Activity;
import android.os.Handler;
import android.os.Looper;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;

/**
 * Runs the background jobs of one screen and hands their results, on the main thread, to the
 * activity instance that is currently attached.
 *
 * <p>An activity keeps its Worker across a configuration change (rotation) through
 * {@code onRetainNonConfigurationInstance()}, so a job started before the rotation reports to
 * the new instance. Results that arrive while no live activity is attached (stopped, or between
 * two instances) are queued and delivered on the next {@link #attach}. When the screen is
 * finished for good, {@link #abandon()} drops late results. Starting a job id again supersedes
 * the previous run of that id: only the latest run's result is delivered.
 *
 * <p>Jobs must not touch views or the activity: create their {@link Callable} in a static
 * method so it does not capture the activity (no leak while the thread finishes).
 */
final class Worker {

    /** Receives job results; called on the main thread while the attached activity is alive. */
    interface Host {
        void onJobDone(int job, Object result, Throwable error);
    }

    private static final Handler MAIN = new Handler(Looper.getMainLooper());

    // All fields are used on the main thread only.
    private Activity activity;
    private Host host;
    private boolean abandoned;
    private final Map<Integer, Integer> generation = new HashMap<Integer, Integer>();
    private final Set<Integer> active = new HashSet<Integer>();
    private final List<Done> queue = new ArrayList<Done>();
    private Object retained;

    private static final class Done {
        final int job;
        final Object result;
        final Throwable error;

        Done(int job, Object result, Throwable error) {
            this.job = job;
            this.result = result;
            this.error = error;
        }
    }

    /** The Worker retained from the previous instance, or a new one. */
    static Worker restore(Object lastNonConfigurationInstance) {
        if (lastNonConfigurationInstance instanceof Worker) {
            Worker w = (Worker) lastNonConfigurationInstance;
            w.abandoned = false;
            return w;
        }
        return new Worker();
    }

    /** Starts {@code work} on a new background thread as job {@code job}. */
    void run(final int job, final Callable<?> work) {
        Integer old = generation.get(job);
        final int gen = old == null ? 1 : old.intValue() + 1;
        generation.put(job, gen);
        active.add(job);
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                Object result = null;
                Throwable error = null;
                try {
                    result = work.call();
                } catch (Throwable e) {
                    // includes OutOfMemoryError on huge inputs: reported, never a crash
                    error = e;
                }
                final Object r = result;
                final Throwable err = error;
                MAIN.post(new Runnable() {
                    @Override
                    public void run() {
                        finished(job, gen, r, err);
                    }
                });
            }
        }, "csprng-job-" + job);
        t.setPriority(Thread.NORM_PRIORITY - 1);
        t.start();
    }

    /** True while the latest run of {@code job} has not reported back. */
    boolean isRunning(int job) {
        return active.contains(job);
    }

    /** True while any job is running. */
    boolean isBusy() {
        return !active.isEmpty();
    }

    void attach(Activity a, Host h) {
        activity = a;
        host = h;
        abandoned = false;
        deliver();
    }

    void detach(Activity a) {
        if (activity == a) {
            activity = null;
            host = null;
        }
    }

    /** The screen is finishing: forget everything, drop late results. */
    void abandon() {
        abandoned = true;
        activity = null;
        host = null;
        queue.clear();
        retained = null;
    }

    /** Extra screen state kept across rotation together with the worker. */
    Object retained() {
        return retained;
    }

    void setRetained(Object o) {
        retained = o;
    }

    private void finished(int job, int gen, Object result, Throwable error) {
        Integer latest = generation.get(job);
        if (latest == null || latest.intValue() != gen) {
            return;                                   // superseded by a newer run
        }
        active.remove(job);
        if (abandoned) {
            return;
        }
        queue.add(new Done(job, result, error));
        deliver();
    }

    private void deliver() {
        while (!queue.isEmpty() && host != null && Ui.alive(activity)) {
            Done d = queue.remove(0);
            host.onJobDone(d.job, d.result, d.error);
        }
    }
}
