"""Loading and validating historical draw files.

A draw file is described in config.yaml (separator, encoding, date column and format,
the columns of every pool, optional checksum columns). Rows whose numbers are missing
are kept out of the backtest and listed in the validation report; every other
inconsistency (wrong count, duplicates, out-of-range numbers, unsorted dates, wrong
checksums) stops the pipeline.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from methods.common import Game, Pool


class DataValidationError(ValueError):
    pass


def load_draws(path: str | Path, spec: dict, game: Game) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    raw = pd.read_csv(path, sep=spec.get("sep", ","), encoding=spec.get("encoding", "utf-8"), dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    dates = pd.to_datetime(raw[spec["date_column"]].str.strip(), format=spec["date_format"]).dt.date
    label_cols = spec.get("label_columns", [])
    labels = raw[label_cols].astype(str).agg("/".join, axis=1) if label_cols else dates.astype(str)

    errors, missing = [], []
    rows = []
    for i in range(len(raw)):
        rec = {"date": dates.iloc[i], "label": labels.iloc[i]}
        complete = True
        for pool in game.pools:
            cols = spec["pools"][pool.name]
            vals = raw.loc[i, cols]
            if vals.isna().any() or (vals.astype(str).str.strip() == "").any():
                complete = False
                break
            nums = [int(v) for v in vals]
            if pool.kind == "set":
                if len(set(nums)) != pool.pick:
                    errors.append(f"row {i + 1}: {pool.name} has duplicates {nums}")
                if any(not 1 <= n <= pool.size for n in nums):
                    errors.append(f"row {i + 1}: {pool.name} out of range {nums}")
                rec[pool.name] = tuple(sorted(nums))
            else:
                if any(not 0 <= n < pool.size for n in nums):
                    errors.append(f"row {i + 1}: {pool.name} digit out of range {nums}")
                rec[pool.name] = tuple(nums)
        if not complete:
            missing.append({"row": i + 1, "date": dates.iloc[i].isoformat(), "label": labels.iloc[i]})
            continue
        checks = spec.get("checks", {})
        main = rec[game.pools[0].name]
        if "sum_column" in checks and int(raw.loc[i, checks["sum_column"]]) != sum(main):
            errors.append(f"row {i + 1}: sum column {raw.loc[i, checks['sum_column']]} != {sum(main)}")
        if "even_column" in checks and int(raw.loc[i, checks["even_column"]]) != sum(1 for n in main if n % 2 == 0):
            errors.append(f"row {i + 1}: even-count column mismatch")
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df["date"].duplicated().any():
        errors.append(f"duplicate dates: {df.loc[df['date'].duplicated(), 'date'].tolist()}")
    if not df["date"].is_monotonic_increasing:
        errors.append("draw dates are not in chronological order")
    if errors:
        raise DataValidationError("; ".join(errors[:10]))

    gaps = []
    if spec.get("expected_frequency") == "daily":
        all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D").date
        present = set(df["date"])
        gaps = [d.isoformat() for d in all_days if d not in present]
    report = {
        "file": path.name, "rows_in_file": int(len(raw)), "complete_draws": int(len(df)),
        "rows_without_numbers": missing, "first_date": df["date"].min().isoformat(),
        "last_date": df["date"].max().isoformat(), "calendar_gaps": gaps,
        "checks_passed": ["count and range per pool", "no duplicates within a draw", "chronological order",
                          "unique dates"] + [f"{k} matches" for k in spec.get("checks", {})],
    }
    return df.reset_index(drop=True), report


def crosscheck_xlsx(df: pd.DataFrame, xlsx_path: str | Path, spec: dict, pool: Pool) -> dict:
    """Compare the CSV draws with the workbook supplied alongside it."""
    x = pd.read_excel(xlsx_path, sheet_name=spec["sheet"])
    x = x.dropna(subset=spec["columns"])
    x_dates = pd.to_datetime(x[spec["date_column"]]).dt.date
    x_map = {d: tuple(sorted(int(v) for v in row)) for d, row in zip(x_dates, x[spec["columns"]].values)}
    csv_map = dict(zip(df["date"], df[pool.name]))
    mismatches = [d.isoformat() for d in csv_map if x_map.get(d) != csv_map[d]]
    only_xlsx = [d.isoformat() for d in x_map if d not in csv_map]
    return {"file": Path(xlsx_path).name, "sheet": spec["sheet"], "draws_in_xlsx": len(x_map),
            "identical": not mismatches and not only_xlsx, "mismatched_dates": mismatches, "only_in_xlsx": only_xlsx}


def membership_matrix(df: pd.DataFrame, pool: Pool) -> np.ndarray:
    """(draws, N) 0/1 matrix: M[t, n-1] = 1 if number n was drawn in draw t."""
    m = np.zeros((len(df), pool.size), dtype=np.int8)
    for t, nums in enumerate(df[pool.name]):
        m[t, [n - 1 for n in nums]] = 1
    return m


def split_index(df: pd.DataFrame, evaluation_start: date) -> int:
    """Index of the first evaluation draw: every earlier draw is training data."""
    idx = np.flatnonzero(np.array([d >= evaluation_start for d in df["date"]]))
    if len(idx) == 0:
        raise DataValidationError(f"no draws on or after {evaluation_start}")
    return int(idx[0])
