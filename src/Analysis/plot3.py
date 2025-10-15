# plot3
# Supply overview and real vs predicted demand comparison of the whole system
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sg_config as Config

# ----------------------------
# Configuration & Colors
# ----------------------------
DATA_DIR = Path(Config.DATA_DIR)
DAY_START = Config.TIME_WINDOW_START
DAY_END = Config.TIME_WINDOW_END   # inclusive

# Subdirs
cons_hist_dir = DATA_DIR / "consumer" / "historicaldata"
cons_pred_dir = DATA_DIR / "consumer" / "predictiondata"
cons_real_dir = DATA_DIR / "consumer" / "realdata"

pros_hist_dir = DATA_DIR / "prosumer" / "historicaldata"
pros_pred_dir = DATA_DIR / "prosumer" / "predictiondata"
pros_real_dir = DATA_DIR / "prosumer" / "realdata"

# Time base for real-style "time" -> day mapping (1-based hour index)
T0 = (DAY_START - 1) * 24 + 1

# Bar colors (left subplot) — exact specs
COLOR_TOTAL = "#C0BDD3" ##C0BDD3
COLOR_NATIONAL = "#733E73"
COLOR_PROSUM = "#605A92"

# Line colors (right subplot) — exact specs
COLOR_PRED = "#AC5956"
COLOR_REAL = "#8DA5D2"

# Matplotlib global style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


# ----------------------------
# Utilities
# ----------------------------
def _list_house_files(dir_path: Path) -> List[Path]:
    return sorted([p for p in Path(dir_path).glob("house_*.csv") if p.is_file()])


def _safe_read_cols(path: Path, cols: List[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, usecols=cols)
    except Exception:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df


def sum_daily_real_style(dir_a: Path, dir_b: Path, time_col="time", val_col="demand") -> pd.DataFrame:
    """
    Convert per-hour 'time' (absolute index) to day index aligned to DAY_START.
    Sum across houses and both dirs -> ['day','total'].
    """
    def _one(dir_path: Path) -> pd.DataFrame:
        files = _list_house_files(dir_path)
        out = []
        for f in files:
            df = _safe_read_cols(f, [time_col, val_col])
            df = df[pd.to_numeric(df[time_col], errors="coerce").notna()]
            df = df[pd.to_numeric(df[val_col],  errors="coerce").notna()]
            if df.empty:
                continue
            df[time_col] = df[time_col].astype(int)
            df[val_col] = df[val_col].astype(float)
            df["day"] = ((df[time_col] - T0) // 24) + DAY_START
            out.append(df[["day", val_col]])
        if not out:
            return pd.DataFrame(columns=["day", val_col])
        all_df = pd.concat(out, ignore_index=True)
        agg = all_df.groupby("day", as_index=False)[
            val_col].sum().rename(columns={val_col: "val"})
        return agg

    a = _one(dir_a)
    b = _one(dir_b)
    frame = pd.merge(a, b, on="day", how="outer",
                     suffixes=("_a", "_b")).fillna(0.0)
    frame["total"] = frame.get("val_a", 0.0) + frame.get("val_b", 0.0)
    return frame[["day", "total"]].sort_values("day")


def prosumer_daily_production_from_hist(pros_hist: Path, time_col="time", supply_col="supply") -> pd.DataFrame:
    """
    Sum hourly 'supply' to daily totals for all prosumers -> ['day','prosumer_production'].
    Day mapping: time 1..24 -> DAY_START, etc.
    """
    files = _list_house_files(pros_hist)
    out = []
    for f in files:
        df = _safe_read_cols(f, [time_col, supply_col])
        df = df[pd.to_numeric(df[time_col], errors="coerce").notna()]
        df = df[pd.to_numeric(df[supply_col], errors="coerce").notna()]
        if df.empty:
            continue
        df[time_col] = df[time_col].astype(int)
        df[supply_col] = df[supply_col].astype(float)
        df["day"] = ((df[time_col] - 1) // 24) + DAY_START
        out.append(df[["day", supply_col]])
    if not out:
        return pd.DataFrame(columns=["day", "prosumer_production"])
    all_df = pd.concat(out, ignore_index=True)
    agg = all_df.groupby("day", as_index=False)[supply_col].sum()
    return agg.rename(columns={supply_col: "prosumer_production"}).sort_values("day")


def per_house_daily_totals_prediction(dir_path: Path, day_col="day", val_col="demand") -> pd.DataFrame:
    """
    For uncertainty: per-house predicted daily totals -> ['day','house_id','total_pred_house'].
    """
    files = _list_house_files(dir_path)
    out = []
    for f in files:
        df = _safe_read_cols(f, [day_col, val_col])
        df = df[pd.to_numeric(df[day_col], errors="coerce").notna()]
        df = df[pd.to_numeric(df[val_col], errors="coerce").notna()]
        if df.empty:
            continue
        df[day_col] = df[day_col].astype(int)
        df[val_col] = df[val_col].astype(float)
        house_id = int(f.stem.split("_")[-1])
        g = df.groupby(day_col, as_index=False)[val_col].sum().rename(
            columns={val_col: "total_pred_house"})
        g["house_id"] = house_id
        out.append(g)
    if not out:
        return pd.DataFrame(columns=[day_col, "house_id", "total_pred_house"])
    return pd.concat(out, ignore_index=True)


def per_house_daily_totals_real(dir_path: Path, time_col="time", val_col="demand") -> pd.DataFrame:
    """
    For uncertainty: per-house real daily totals -> ['day','house_id','total_real_house'].
    """
    files = _list_house_files(dir_path)
    out = []
    for f in files:
        df = _safe_read_cols(f, [time_col, val_col])
        df = df[pd.to_numeric(df[time_col], errors="coerce").notna()]
        df = df[pd.to_numeric(df[val_col],  errors="coerce").notna()]
        if df.empty:
            continue
        df[time_col] = df[time_col].astype(int)
        df[val_col] = df[val_col].astype(float)
        df["day"] = ((df[time_col] - T0) // 24) + DAY_START
        house_id = int(f.stem.split("_")[-1])
        g = df.groupby("day", as_index=False)[val_col].sum().rename(
            columns={val_col: "total_real_house"})
        g["house_id"] = house_id
        out.append(g)
    if not out:
        return pd.DataFrame(columns=["day", "house_id", "total_real_house"])
    return pd.concat(out, ignore_index=True)


def restrict_days(df: pd.DataFrame, start_day: int, end_day: int) -> pd.DataFrame:
    """Inclusive window [start_day, end_day]."""
    if df.empty:
        return df
    return df[(df["day"] >= start_day) & (df["day"] <= end_day)].copy()


def add_bar_value_labels(ax, x, heights, fmt="{:.0f}", dy=0.01):
    """
    Add value labels above bars.
    dy is a fraction of the y-range for vertical offset.
    """
    if len(x) == 0:
        return
    yr = ax.get_ylim()
    off = (yr[1] - yr[0]) * dy
    for xi, hi in zip(x, heights):
        ax.text(xi, hi + off, fmt.format(hi),
                ha="center", va="bottom", fontsize=11)


# ----------------------------
# Build daily aggregates
# ----------------------------
# Total predicted consumption per day (using real-style aggregator)
pred_total = sum_daily_real_style(
    cons_pred_dir, pros_pred_dir, time_col="time", val_col="demand")
pred_total = pred_total.rename(columns={"total": "total_predicted"})

# Prosumer daily production (from historical supply)
pros_prod = prosumer_daily_production_from_hist(
    pros_hist_dir, time_col="time", supply_col="supply")

# National = Total predicted - Prosumer
daily_prod = pd.merge(pred_total, pros_prod, on="day", how="left").fillna(0.0)
daily_prod["national_production"] = daily_prod["total_predicted"] - \
    daily_prod["prosumer_production"]
daily_prod["national_production"] = daily_prod["national_production"].clip(
    lower=0.0)

# Total real consumption per day
real_total = sum_daily_real_style(
    cons_real_dir, pros_real_dir, time_col="time", val_col="demand")
real_total = real_total.rename(columns={"total": "total_real"})

# Uncertainty (cross-house std) for Pred & Real
pred_cons_h = per_house_daily_totals_prediction(cons_pred_dir)
pred_pros_h = per_house_daily_totals_prediction(pros_pred_dir)
pred_houses = pd.concat([pred_cons_h, pred_pros_h], ignore_index=True) if (
    not pred_cons_h.empty or not pred_pros_h.empty) else pd.DataFrame(columns=["day", "house_id", "total_pred_house"])
pred_std = pred_houses.groupby("day", as_index=False)["total_pred_house"].std(
    ddof=1).rename(columns={"total_pred_house": "pred_std"}).fillna(0.0)

real_cons_h = per_house_daily_totals_real(cons_real_dir)
real_pros_h = per_house_daily_totals_real(pros_real_dir)
real_houses = pd.concat([real_cons_h, real_pros_h], ignore_index=True) if (
    not real_cons_h.empty or not real_pros_h.empty) else pd.DataFrame(columns=["day", "house_id", "total_real_house"])
real_std = real_houses.groupby("day", as_index=False)["total_real_house"].std(
    ddof=1).rename(columns={"total_real_house": "real_std"}).fillna(0.0)

# Attach std to totals
pred_total = pd.merge(pred_total, pred_std, on="day",
                      how="left").fillna({"pred_std": 0.0})
real_total = pd.merge(real_total, real_std, on="day",
                      how="left").fillna({"real_std": 0.0})

# Restrict to inclusive window
daily_prod_win = restrict_days(daily_prod, DAY_START, DAY_END)
pred_total_win = restrict_days(pred_total, DAY_START, DAY_END)
real_total_win = restrict_days(real_total, DAY_START, DAY_END)

# Combined frame for right subplot (Pred vs Real)
cmp_df = pd.merge(pred_total_win, real_total_win, on="day", how="outer", suffixes=(
    "_pred", "_real")).fillna(0.0).sort_values("day")

# ----------------------------
# Plot
# ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# ---- Left: Grouped bars (Total, National, Prosumer) ----
if not daily_prod_win.empty:
    days = daily_prod_win["day"].to_numpy()
    total_prod = daily_prod_win["total_predicted"].to_numpy()
    nat_prod = daily_prod_win["national_production"].to_numpy()
    pros_prod = daily_prod_win["prosumer_production"].to_numpy()

    x = np.arange(len(days))
    bar_w = 0.28

    ax1.bar(x - bar_w, total_prod, width=bar_w,
            color=COLOR_TOTAL,    label="Total Supply")
    ax1.bar(x,          nat_prod, width=bar_w,
            color=COLOR_NATIONAL, label="National Grid Supply")
    ax1.bar(x + bar_w,  pros_prod, width=bar_w,
            color=COLOR_PROSUM,   label="Prosumer Supply")

    # X-ticks as D<day>
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d}" for d in days], rotation=0)

    ax1.set_xlabel("Day")
    ax1.set_ylabel("Power (kWh per day)")
    ax1.set_title(f"Supply (Days {DAY_START}–{DAY_END})")

    # # Optional value labels (comment out if too cluttered)
    # add_bar_value_labels(ax1, x - bar_w, total_prod, fmt="{:.0f}")
    # add_bar_value_labels(ax1, x,         nat_prod,   fmt="{:.0f}")
    # add_bar_value_labels(ax1, x + bar_w, pros_prod,  fmt="{:.0f}")

    ax1.set_ylim(0, 1000)
    # ax1.grid(True, linestyle="--", alpha=0.4)
    # ax1.grid(axis="y", visible=False)
    ax1.legend(loc="upper left")

# ---- Right: Predicted vs Real with uncertainty & shading ----
if not cmp_df.empty:
    d = cmp_df["day"].to_numpy()
    y_pred = cmp_df["total_predicted"].to_numpy()
    y_real = cmp_df["total_real"].to_numpy()
    s_pred = cmp_df["pred_std"].to_numpy(
    ) if "pred_std" in cmp_df else np.zeros_like(y_pred)
    s_real = cmp_df["real_std"].to_numpy(
    ) if "real_std" in cmp_df else np.zeros_like(y_real)

    # --- NEW: metrics block (place right after difference shading, before setting labels/legend) ---
    diff = y_pred - y_real

    # Mean Absolute Error (kWh/day)
    mae = float(np.mean(np.abs(diff)))

    # Mean Absolute Percentage Error (%), safe for zeros
    denom = np.where(y_real == 0, 1.0, y_real)
    mape = float(np.mean(np.abs(diff) / denom) * 100.0)

    # Total absolute error as a percentage of total real (%)
    total_abs_error = float(np.sum(np.abs(diff)))
    total_real = float(np.sum(y_real))
    total_err_pct = float(total_abs_error / total_real *
                          100.0) if total_real > 0 else np.nan

    # Signed bias as percentage of total real (%): positive = overprediction
    bias_pct = float(np.sum(diff) / total_real *
                     100.0) if total_real > 0 else np.nan

    # Compose a compact metrics panel
    metrics_text = (
        f"MAPE = {mape:.2f}%\n"
        f"Total Error = {total_err_pct:.2f}%\n"
        f"Bias = {bias_pct:+.2f}%"
    )

    # Render at bottom-left in axis coordinates with a soft box
    ax2.text(
        0.015, 0.03, metrics_text,
        transform=ax2.transAxes,
        fontsize=12,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="gray", alpha=0.85)
    )

    # Pred line + ±1σ band
    ax2.plot(d, y_pred, label="Predicted Demand",
             color=COLOR_PRED, linewidth=2.0)
    ax2.fill_between(d, y_pred - s_pred, y_pred + s_pred,
                     color=COLOR_PRED, alpha=0.15)

    # Real line + error bars
    ax2.plot(d, y_real, label="Real Demand",
             color=COLOR_REAL, linestyle="--", linewidth=2.0)
    ax2.errorbar(d, y_real, yerr=s_real, fmt="o", ms=4,
                 capsize=3, lw=1.2, color=COLOR_REAL, alpha=0.9)

    # Difference shading
    ax2.fill_between(d, y_real, y_pred, where=(
        y_real >= y_pred), alpha=0.10, color=COLOR_REAL)
    ax2.fill_between(d, y_real, y_pred, where=(
        y_real < y_pred), alpha=0.10, color=COLOR_PRED)

    ax2.set_ylim(0, 1000)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Power (kWh per day)")
    ax2.set_title(
        f"Predicted vs Real Demand (Days {DAY_START}–{DAY_END})")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="upper left")

plt.tight_layout()
plt.show()
