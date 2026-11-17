package com.medonumbworks.csprngpredictor.ui;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import com.medonumbworks.csprngpredictor.R;
import com.medonumbworks.csprngpredictor.core.Game;

import java.util.Collections;
import java.util.Set;
import java.util.TreeSet;

/**
 * The 22 numbers as circles in a wrapping grid. Each cell is a child {@link TextView}; the
 * circle backgrounds are {@link GradientDrawable} ovals created in code (no images).
 *
 * <p>Layout: 6 columns on phones; the cell size is derived from the available width in
 * {@link #onMeasure} (min 40dp, max 72dp, 8dp gaps). On wide screens more columns are used so
 * the cells do not exceed 72dp; on very narrow screens fewer, so they stay at least 40dp.
 *
 * <p>Modes: {@link #MODE_DISPLAY} highlights the selected numbers, {@link #MODE_SELECT} lets the
 * user toggle numbers (at most {@link #setMaxSelection} of them) and {@link #MODE_HEAT} fills
 * each circle with an intensity proportional to score / max score. In every mode a
 * "highlighted" number gets an accent ring; selected AND highlighted = green "hit".
 */
public final class NumberGridView extends ViewGroup {

    public static final int MODE_DISPLAY = 0;
    public static final int MODE_SELECT = 1;
    public static final int MODE_HEAT = 2;

    /** Selection changed by a tap (MODE_SELECT). */
    public interface OnSelectionChangedListener {
        void onSelectionChanged(Set<Integer> selected);
    }

    /** A tap tried to select more than the maximum (MODE_SELECT). */
    public interface OnLimitReachedListener {
        void onLimitReached(int maxSelection);
    }

    private static final int COUNT = Game.MAX_NUMBER;
    private static final int PREFERRED_COLUMNS = 6;
    /** Digit size relative to the circle: about 20sp on a typical phone (42dp circles). */
    private static final float TEXT_TO_CELL = 0.47f;

    private final TextView[] cells = new TextView[COUNT];
    private final GradientDrawable[] shapes = new GradientDrawable[COUNT];
    private final TreeSet<Integer> selected = new TreeSet<Integer>();
    private final TreeSet<Integer> highlighted = new TreeSet<Integer>();
    private final double[] scores = new double[COUNT];
    private boolean hasScores;
    private int mode = MODE_DISPLAY;
    private int maxSelection = Game.PICK;
    private OnSelectionChangedListener selectionListener;
    private OnLimitReachedListener limitListener;

    private final int minCell;
    private final int maxCell;
    private final int gap;
    private final int ringWidth;
    private final float maxTextPx;
    private int columns = PREFERRED_COLUMNS;
    private int cellSize;
    private float textPx = -1f;

    private final int colorSelected;
    private final int colorSelectedText;
    private final int colorHit;
    private final int colorHitText;
    private final int colorRing;
    private final int colorUnselected;
    private final int colorUnselectedText;
    private final int colorHeat;
    private final int colorHeatText;

    private final OnClickListener cellClick = new OnClickListener() {
        @Override
        public void onClick(View v) {
            if (mode != MODE_SELECT || !isEnabled()) {
                return;
            }
            Object tag = v.getTag();
            if (!(tag instanceof Integer)) {
                return;
            }
            Integer n = (Integer) tag;
            if (selected.contains(n)) {
                selected.remove(n);
            } else if (selected.size() >= maxSelection) {
                if (limitListener != null) {
                    limitListener.onLimitReached(maxSelection);
                }
                return;
            } else {
                selected.add(n);
            }
            refresh();
            if (selectionListener != null) {
                selectionListener.onSelectionChanged(getSelectedNumbers());
            }
        }
    };

    public NumberGridView(Context context) {
        this(context, null);
    }

    public NumberGridView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public NumberGridView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        minCell = dp(40);
        maxCell = dp(72);
        gap = dp(8);
        ringWidth = dp(3);
        maxTextPx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, 30,
                getResources().getDisplayMetrics());
        cellSize = dp(48);

        colorSelected = context.getColor(R.color.grid_selected);
        colorSelectedText = context.getColor(R.color.grid_selected_text);
        colorHit = context.getColor(R.color.grid_hit);
        colorHitText = context.getColor(R.color.grid_hit_text);
        colorRing = context.getColor(R.color.grid_ring);
        colorUnselected = context.getColor(R.color.grid_unselected);
        colorUnselectedText = context.getColor(R.color.grid_unselected_text);
        colorHeat = context.getColor(R.color.grid_heat);
        colorHeatText = context.getColor(R.color.grid_heat_text);
        int ripple = context.getColor(R.color.grid_ripple);

        for (int i = 0; i < COUNT; i++) {
            TextView tv = new TextView(context);
            tv.setGravity(Gravity.CENTER);
            tv.setIncludeFontPadding(false);
            tv.setTypeface(Typeface.DEFAULT_BOLD);
            tv.setMaxLines(1);
            tv.setText(String.valueOf(i + 1));
            tv.setTag(Integer.valueOf(i + 1));
            GradientDrawable shape = new GradientDrawable();
            shape.setShape(GradientDrawable.OVAL);
            shapes[i] = shape;
            // No mask: the ripple is clipped to the oval content.
            tv.setBackground(new RippleDrawable(ColorStateList.valueOf(ripple), shape, null));
            tv.setOnClickListener(cellClick);
            cells[i] = tv;
            addView(tv);
        }
        applyMode();
        refresh();
    }

    // =================================================================== public API

    /** One of {@link #MODE_DISPLAY}, {@link #MODE_SELECT}, {@link #MODE_HEAT}. */
    public void setMode(int mode) {
        this.mode = (mode == MODE_SELECT || mode == MODE_HEAT) ? mode : MODE_DISPLAY;
        applyMode();
        refresh();
    }

    public int getMode() {
        return mode;
    }

    /** Replaces the selection (values outside 1..22 are ignored; no listener call). */
    public void setSelected(Set<Integer> numbers) {
        selected.clear();
        if (numbers != null) {
            for (Integer n : numbers) {
                if (n != null && Game.isValidNumber(n.intValue())) {
                    if (mode == MODE_SELECT && selected.size() >= maxSelection) {
                        break;
                    }
                    selected.add(n);
                }
            }
        }
        refresh();
    }

    /** A sorted copy of the selected numbers. */
    public Set<Integer> getSelectedNumbers() {
        return new TreeSet<Integer>(selected);
    }

    /** Secondary highlight ring (e.g. the numbers actually drawn). */
    public void setHighlighted(Set<Integer> numbers) {
        highlighted.clear();
        if (numbers != null) {
            for (Integer n : numbers) {
                if (n != null && Game.isValidNumber(n.intValue())) {
                    highlighted.add(n);
                }
            }
        }
        refresh();
    }

    public Set<Integer> getHighlighted() {
        return Collections.unmodifiableSet(new TreeSet<Integer>(highlighted));
    }

    /** Heat-mode scores, index i = number i + 1 (null clears). */
    public void setScores(double[] scores22) {
        hasScores = false;
        for (int i = 0; i < COUNT; i++) {
            double v = scores22 != null && i < scores22.length ? scores22[i] : 0.0;
            if (Double.isNaN(v) || Double.isInfinite(v) || v < 0) {
                v = 0.0;
            }
            scores[i] = v;
            if (v > 0) {
                hasScores = true;
            }
        }
        refresh();
    }

    public void setOnSelectionChangedListener(OnSelectionChangedListener l) {
        selectionListener = l;
    }

    public void setOnLimitReachedListener(OnLimitReachedListener l) {
        limitListener = l;
    }

    /** Maximum number of selected numbers in MODE_SELECT (0..22, default 11). */
    public void setMaxSelection(int max) {
        maxSelection = Math.max(0, Math.min(COUNT, max));
        while (selected.size() > maxSelection) {
            selected.pollLast();
        }
        refresh();
    }

    public int getMaxSelection() {
        return maxSelection;
    }

    @Override
    public void setEnabled(boolean enabled) {
        super.setEnabled(enabled);
        if (cells[COUNT - 1] == null) {
            return;                                   // still constructing
        }
        for (TextView tv : cells) {
            tv.setEnabled(enabled);
        }
        applyMode();
        refresh();
    }

    // =================================================================== drawing state

    private void applyMode() {
        boolean clickable = mode == MODE_SELECT && isEnabled();
        for (TextView tv : cells) {
            tv.setClickable(clickable);
            tv.setFocusable(clickable);
        }
    }

    private void refresh() {
        double max = 0.0;
        if (mode == MODE_HEAT && hasScores) {
            for (double s : scores) {
                max = Math.max(max, s);
            }
        }
        Context c = getContext();
        for (int i = 0; i < COUNT; i++) {
            Integer n = Integer.valueOf(i + 1);
            boolean sel = selected.contains(n);
            boolean hi = highlighted.contains(n);
            int fill;
            int text;
            int stroke = Color.TRANSPARENT;
            String cd;
            if (mode == MODE_HEAT) {
                double t = max > 0 ? scores[i] / max : 0.0;
                t = Math.max(0.0, Math.min(1.0, t));
                fill = blend(colorUnselected, colorHeat, (float) t);
                text = t >= 0.6 ? colorHeatText : colorUnselectedText;
                if (hi) {
                    stroke = colorRing;
                }
                cd = c.getString(R.string.grid_cd_heat, i + 1, (int) Math.round(t * 100));
            } else if (sel && hi) {
                fill = colorHit;
                text = colorHitText;
                cd = c.getString(R.string.grid_cd_hit, i + 1);
            } else if (sel) {
                fill = colorSelected;
                text = colorSelectedText;
                cd = c.getString(R.string.grid_cd_selected, i + 1);
            } else if (hi) {
                fill = colorUnselected;
                text = colorUnselectedText;
                stroke = colorRing;
                cd = c.getString(R.string.grid_cd_drawn, i + 1);
            } else {
                fill = colorUnselected;
                text = colorUnselectedText;
                cd = String.valueOf(i + 1);
            }
            GradientDrawable g = shapes[i];
            g.setColor(fill);
            g.setStroke(ringWidth, stroke);
            cells[i].setTextColor(text);
            cells[i].setContentDescription(cd);
            cells[i].setAlpha(isEnabled() ? 1f : 0.6f);
        }
        invalidate();
    }

    /** Linear RGB interpolation from {@code a} (t = 0) to {@code b} (t = 1). */
    private static int blend(int a, int b, float t) {
        float u = 1f - t;
        int r = Math.round(Color.red(a) * u + Color.red(b) * t);
        int g = Math.round(Color.green(a) * u + Color.green(b) * t);
        int bl = Math.round(Color.blue(a) * u + Color.blue(b) * t);
        return Color.rgb(r, g, bl);
    }

    // =================================================================== layout

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        int widthMode = MeasureSpec.getMode(widthMeasureSpec);
        int widthSize = MeasureSpec.getSize(widthMeasureSpec);
        int padH = getPaddingLeft() + getPaddingRight();
        int padV = getPaddingTop() + getPaddingBottom();

        int avail;
        if (widthMode == MeasureSpec.UNSPECIFIED) {
            avail = PREFERRED_COLUMNS * dp(52) + (PREFERRED_COLUMNS - 1) * gap;
        } else {
            avail = Math.max(0, widthSize - padH);
        }
        computeGrid(avail);

        float px = Math.min(cellSize * TEXT_TO_CELL, maxTextPx);
        if (Math.abs(px - textPx) > 0.5f) {
            textPx = px;
            for (TextView tv : cells) {
                tv.setTextSize(TypedValue.COMPLEX_UNIT_PX, px);
            }
        }
        int childSpec = MeasureSpec.makeMeasureSpec(cellSize, MeasureSpec.EXACTLY);
        for (TextView tv : cells) {
            tv.measure(childSpec, childSpec);
        }

        int rows = (COUNT + columns - 1) / columns;
        int contentW = columns * cellSize + (columns - 1) * gap;
        int contentH = rows * cellSize + (rows - 1) * gap;
        int w = widthMode == MeasureSpec.EXACTLY ? widthSize : contentW + padH;
        if (widthMode == MeasureSpec.AT_MOST) {
            w = Math.min(w, widthSize);
        }
        setMeasuredDimension(w, resolveSize(contentH + padV, heightMeasureSpec));
    }

    /** Chooses {@link #columns} and {@link #cellSize} for {@code avail} pixels of width. */
    private void computeGrid(int avail) {
        int maxFit = Math.max(1, (avail + gap) / (minCell + gap));
        int cols = Math.min(PREFERRED_COLUMNS, maxFit);
        int size = (avail - gap * (cols - 1)) / cols;
        if (size > maxCell) {
            // Wide screen: more columns so the circles stay <= 72dp, rows balanced.
            int want = (int) Math.ceil((avail + gap) / (double) (maxCell + gap));
            want = Math.max(cols, Math.min(Math.min(want, COUNT), maxFit));
            int rows = (COUNT + want - 1) / want;
            want = (COUNT + rows - 1) / rows;
            cols = want;
            size = Math.min(maxCell, (avail - gap * (cols - 1)) / cols);
        }
        if (size <= 0) {
            size = Math.max(1, avail);
        }
        columns = cols;
        cellSize = size;
    }

    @Override
    protected void onLayout(boolean changed, int l, int t, int r, int b) {
        int width = r - l;
        int padL = getPaddingLeft();
        int innerW = width - padL - getPaddingRight();
        int contentW = columns * cellSize + (columns - 1) * gap;
        int x0 = padL + Math.max(0, (innerW - contentW) / 2);
        int y0 = getPaddingTop();
        for (int i = 0; i < COUNT; i++) {
            int row = i / columns;
            int col = i % columns;
            int left = x0 + col * (cellSize + gap);
            int top = y0 + row * (cellSize + gap);
            cells[i].layout(left, top, left + cellSize, top + cellSize);
        }
    }

    @Override
    public boolean shouldDelayChildPressedState() {
        return false;
    }

    private int dp(float v) {
        return Math.round(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v,
                getResources().getDisplayMetrics()));
    }
}
