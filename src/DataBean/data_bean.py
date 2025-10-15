# data_bean.py
# Convert SunDance data hourly CSVs into:
# - consumers:  [time (1-based), demand]
# - prosumers:  [time (1-based), demand, supply]

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import sg_config as Config

# -----------------------------
# Utils
# -----------------------------


def exist_house(dir_path, index):
    """
    Return the next available 'house_{i}.csv' path in dir_path starting at start_index.
    """
    candidate = dir_path / f"house_{index}.csv"
    if candidate.exists():
        return True
    else:
        return False


def _read_series(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["time", "demand"]).dropna().sort_values("time")
    df["time"] = df["time"].astype(int)
    T = int(df["time"].max())
    arr = np.zeros(T, dtype=np.float32)
    arr[df["time"].values - 1] = df["demand"].astype(float).values
    arr[arr < 0] = 0.0
    return arr

# -----------------------------
# Column name detection helpers
# -----------------------------


def detect_columns(cols):
    cl = {c.lower(): c for c in cols}

    ts_candidates = ['date & time', 'date&time',
                     'datetime', 'timestamp', 'time', 'date_time']
    ts_col = next((cl[c] for c in ts_candidates if c in cl), None)
    if ts_col is None:
        raise ValueError(f"Cannot find timestamp column among {cols}")

    use_candidates = ['use [kW]', 'use[kW]',
                      'use (kW)', 'use(kW)', 'usage [kW]', 'usage[kW]', 'use']
    use_col = next((cl[c.lower()]
                   for c in use_candidates if c.lower() in cl), None)

    gen_candidates = ['gen [kW]', 'gen[kW]',
                      'gen (kW)', 'gen(kW)', 'generation [kW]', 'generation[kW]', 'gen']
    gen_col = next((cl[c.lower()]
                   for c in gen_candidates if c.lower() in cl), None)

    return {'ts': ts_col, 'use': use_col, 'gen': gen_col}

# -----------------------------
# Cleaning and ordering
# -----------------------------


def load_and_clean(csv_path, start_at_midnight):
    df = pd.read_csv(csv_path, low_memory=False)
    colmap = detect_columns(df.columns.tolist())

    ts_col = colmap['ts']
    use_col = colmap['use']
    gen_col = colmap['gen']

    keep_cols = [c for c in [ts_col, use_col, gen_col] if c is not None]
    df = df[keep_cols].copy()

    # Parse timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    # Ensure numeric
    if use_col is not None:
        df[use_col] = pd.to_numeric(df[use_col], errors='coerce')
    if gen_col is not None:
        df[gen_col] = pd.to_numeric(df[gen_col], errors='coerce')

    # Start from first 00:00:00 if requested
    if start_at_midnight:
        mask_midnight = df[ts_col].dt.strftime("%H:%M:%S") == "00:00:00"
        if mask_midnight.any():
            first_idx = mask_midnight.idxmax()
            df = df.loc[first_idx:].reset_index(drop=True)

    return df, colmap

# -----------------------------
# Builders
# -----------------------------


def to_consumer(df, colmap, ceil3):
    ts_col = colmap['ts']
    use_col = colmap['use']

    if use_col is None:
        raise ValueError(
            "Cannot build consumer dataset: 'use' column not found in this file.")

    out = df[[ts_col, use_col]].copy().reset_index(drop=True)
    out['time'] = range(1, len(out) + 1)
    out = out[['time', use_col]].rename(columns={use_col: 'demand'})

    if ceil3:
        out['demand'] = np.ceil(out['demand'] * 1000) / 1000.0
    return out


def to_prosumer(df, colmap, ceil3):
    ts_col = colmap['ts']
    use_col = colmap['use']
    gen_col = colmap['gen']

    if use_col is None or gen_col is None:
        raise ValueError(
            "Cannot build prosumer dataset: 'use' or 'gen' column not found in this file.")

    out = df[[ts_col, use_col, gen_col]].copy().reset_index(drop=True)
    out['time'] = range(1, len(out) + 1)
    out = out[['time', use_col, gen_col]].rename(
        columns={use_col: 'demand', gen_col: 'supply'})

    if ceil3:
        out['demand'] = np.ceil(out['demand'] * 1000) / 1000.0
        out['supply'] = np.ceil(out['supply'] * 1000) / 1000.0
    return out

# -----------------------------
# Main
# -----------------------------


def main():
    n = Config.CONSUMER_NUMBER   # target number of consumers
    m = Config.PROSUMER_NUMBER   # target number of prosumers
    start_at_midnight = True     # cut from the first 00:00:00 timestamp
    # ceil values to 3 decimals (for on-chain conversion)
    ceil3 = True

    # Paths
    raw_dir = Path(Config.DATA_DIR) / "rawdata" / "energydata"
    cons_out = Path(Config.DATA_DIR) / "consumer" / "historicaldata"
    pros_out = Path(Config.DATA_DIR) / "prosumer" / "historicaldata"
    cons_out.mkdir(parents=True, exist_ok=True)
    pros_out.mkdir(parents=True, exist_ok=True)

    # Collect CSV files
    files = sorted([p for p in raw_dir.glob("*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    # Split files into candidate lists
    consumers_files = files[1:10]
    prosumers_files = files[10:30]

    if len(consumers_files) < n:
        print(
            f"[WARN] Only found {len(consumers_files)} candidate files for consumers (target {n}).")
    if len(prosumers_files) < m:
        print(
            f"[WARN] Only found {len(prosumers_files)} candidate files for prosumers (target {m}).")

    # -------------------------
    # Build consumers
    # -------------------------
    for i in range(1, n + 1):

        if exist_house(cons_out, i):
            continue

        for path in consumers_files:

            df, colmap = load_and_clean(
                path, start_at_midnight=start_at_midnight)
            out = to_consumer(df, colmap, ceil3=ceil3)

            if len(out) >= 8760:
                cons_file = cons_out / f"house_{i}.csv"
                out.to_csv(cons_file, index=False, float_format="%.3f")
                print(
                    f"[consumer {i}] {path.name} -> {cons_file.name} ({len(out)} rows)")
                try:
                    path.unlink()
                    consumers_files.remove(path)
                    print(
                        f"[consumer {i}] Deleted source file: {path.name}")
                except Exception as de:
                    print(
                        f"[consumer {i}] WARN: could not delete {path.name}: {de}")
                break

            else:
                print(
                    f"[consumer ?] SKIP {path.name}: only {len(out)} rows (need >= 8760)")

    # -------------------------
    # Build prosumers
    # -------------------------
    for i in range(n + 1, n + m + 1):

        if exist_house(pros_out, i):
            continue

        for path in prosumers_files:

            df, colmap = load_and_clean(
                path, start_at_midnight=start_at_midnight)
            out = to_prosumer(df, colmap, ceil3=ceil3)

            if len(out) >= 8760:
                pros_file = pros_out / f"house_{i}.csv"
                out.to_csv(pros_file, index=False, float_format="%.3f")
                print(
                    f"[prosumer {i}] {path.name} -> {pros_file.name} ({len(out)} rows)")
                try:
                    path.unlink()
                    prosumers_files.remove(path)
                    print(
                        f"[prosumer {i}] Deleted source file: {path.name}")
                except Exception as de:
                    print(
                        f"[prosumer {i}] WARN: could not delete {path.name}: {de}")
                break
            
            else:
                print(
                    f"[prosumer ?] SKIP {path.name}: only {len(out)} rows (need >= 8760)")

# plot the tendency to filter data with a lot of noise
def plot_tendency(house_id, type):

    if type == "consumer":
        cons_out = Path(Config.DATA_DIR) / "consumer" / "historicaldata" / f"house_{house_id}.csv"
        series = _read_series(cons_out)
        start_idx = (Config.TIME_WINDOW_START - 1) *24
        end_idx = (Config.TIME_WINDOW_END - 1) * 24
        value_tendency = series[start_idx:end_idx].astype(np.float32)

    if type == "prosumer":
        pros_out = Path(Config.DATA_DIR) / "prosumer" / "historicaldata" / f"house_{house_id}.csv"
        series = _read_series(pros_out)
        start_idx = (Config.TIME_WINDOW_START - 1) *24
        end_idx = (Config.TIME_WINDOW_END - 1) * 24
        value_tendency = series[start_idx:end_idx].astype(np.float32)
    
    n_hours = value_tendency.shape[0]
    hours = np.arange(n_hours)

    plt.figure(figsize=(12,5))
    plt.plot(hours, value_tendency, label=f"{type} {house_id}")
    plt.title(f"Data Tendency of {type} {house_id}")
    plt.xlabel("Hour")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()



if __name__ == "__main__":
    main()
    for i in range (1, Config.CONSUMER_NUMBER + 1):
        plot_tendency(i, "consumer")
    for i in range (Config.CONSUMER_NUMBER + 1, Config.CONSUMER_NUMBER + Config.PROSUMER_NUMBER + 1):
        plot_tendency(i, "prosumer")
