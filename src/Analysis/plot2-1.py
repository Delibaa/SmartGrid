# plot2-1
# Consumers + Prosumers: historical vs predicted vs real, both cropped to the Config window
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sg_config as Config

# -----------------
# Config & paths
# -----------------
data_root = Config.DATA_DIR

cons_hist_dir = data_root / "consumer"  / "historicaldata"
cons_real_dir = data_root / "consumer"  / "realdata"
cons_pred_dir = data_root / "consumer"  / "predictiondata"


# cons_hist_dir = Path("")
# cons_real_dir = Path("")
# cons_pred_dir = Path("")

pros_hist_dir = data_root / "prosumer"  / "historicaldata"
pros_real_dir = data_root / "prosumer"  / "realdata"
pros_pred_dir = data_root / "prosumer"  / "predictiondata"

# pros_hist_dir = Path("")
# pros_real_dir = Path("")
# pros_pred_dir = Path("")

# Window in absolute hours (inclusive)
WIN_START = (Config.TIME_WINDOW_START - 1) * 24 + 1   # 1-based hour index start
WIN_END   = (Config.TIME_WINDOW_END   - 1) * 24       # 1-based hour index end

# Choose plotting time base: "window" or "real"
PLOT_TIME_BASE = "window"  # or "real"

def read_totals_by_time_from_dirs(directories, value_col="demand", time_col="time") -> pd.DataFrame:
    """
    Sum `value_col` across all house_*.csv in `directories`, aligned by `time_col`.
    Returns DataFrame with columns: [time_col, f"total_{value_col}"] (time sorted).
    """
    series_list = []
    for d in map(Path, directories):
        if not d.exists():
            continue
        for f in sorted(d.glob("house_*.csv")):
            try:
                df = pd.read_csv(f, usecols=[time_col, value_col])
            except Exception:
                continue
            df = df[pd.to_numeric(df[time_col], errors="coerce").notna()]
            df = df[pd.to_numeric(df[value_col], errors="coerce").notna()]
            s = df.set_index(time_col)[value_col].astype(float)
            series_list.append(s)

    if not series_list:
        return pd.DataFrame(columns=[time_col, f"total_{value_col}"])

    total = None
    for s in series_list:
        total = s if total is None else total.add(s, fill_value=0.0)
    total = total.sort_index()
    return total.reset_index().rename(columns={value_col: f"total_{value_col}"})


def crop_to_window(df: pd.DataFrame, time_col="time", start=WIN_START, end=WIN_END) -> pd.DataFrame:
    """Keep rows with time in [start, end] inclusive."""
    if df.empty:
        return df
    df = df.copy()
    df[time_col] = df[time_col].astype(int)
    return df[(df[time_col] >= start) & (df[time_col] <= end)].sort_values(time_col)


# -----------------
# Read & crop
# -----------------

# Historical total (consumers + prosumers), then crop to window
hist_tot = read_totals_by_time_from_dirs(
    [cons_hist_dir, pros_hist_dir], value_col="demand", time_col="time"
)
if not hist_tot.empty:
    hist_tot = crop_to_window(hist_tot, time_col="time", start=WIN_START, end=WIN_END)
    hist_tot = hist_tot.rename(columns={"time": "time", "total_demand": "total_hist_demand"})

# Real total (consumers + prosumers), treat start as window start, crop to window
real_tot = read_totals_by_time_from_dirs(
    [cons_real_dir, pros_real_dir], value_col="demand", time_col="time"
)
if not real_tot.empty:
    real_tot = crop_to_window(real_tot, time_col="time", start=WIN_START, end=WIN_END)
    real_tot = real_tot.rename(columns={"time": "time", "total_demand": "total_real_demand"})

pred_tot = read_totals_by_time_from_dirs(
    [cons_pred_dir, pros_pred_dir], value_col="demand", time_col="time"
)
if not pred_tot.empty:
    pred_tot = crop_to_window(pred_tot, time_col="time", start=WIN_START, end=WIN_END)
    pred_tot = pred_tot.rename(columns={"time": "time", "total_demand": "total_pred_demand"})

# -----------------
# Align on the same axis
# -----------------
if not hist_tot.empty and not real_tot.empty and not pred_tot.empty:
    frame = pd.merge(hist_tot, real_tot, on="time", how="outer")
    frame = pd.merge(frame, pred_tot, on="time", how="outer")
elif not hist_tot.empty:
    frame = hist_tot.copy()
    frame["total_real_demand"] = 0.0
elif not real_tot.empty:
    frame = real_tot.copy()
    frame["total_hist_demand"] = 0.0
elif not pred_tot.empty:
    frame = pred_tot.copy()
    frame["total_pred_demand"] = 0.0
else:
    frame = pd.DataFrame(columns=["time", "total_hist_demand", "total_real_demand", "total_pred_demand"])

# Fill missing with zeros, sort
for c in ["total_hist_demand", "total_real_demand", "total_pred_demand"]:
    if c in frame:
        frame[c] = frame[c].fillna(0.0)
frame = frame.sort_values("time")

# -----------------
# Choose plotting x-axis
# -----------------
if PLOT_TIME_BASE == "real" and not real_tot.empty:
    # Use only real times present in the cropped real dataset
    frame_plot = frame.merge(real_tot[["time"]].drop_duplicates(), on="time", how="inner")
else:
    # Use full window axis (i.e., all times within [WIN_START, WIN_END] present after merge)
    frame_plot = frame.copy()

# -----------------
# Plot
# -----------------

hist_color   = "#7e7d7e"
pred_color = "#5c95cf"
real_color = "#ee7b15"

# settings
plt.figure(figsize=(12, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 18 
plt.rcParams['axes.labelsize'] = 18 
plt.rcParams['legend.fontsize'] = 14 
plt.rcParams['xtick.labelsize'] = 18 
plt.rcParams['ytick.labelsize'] = 18 

if not frame_plot.empty:
    plt.plot(
        frame_plot["time"], frame_plot["total_hist_demand"],
        label="Total Demand without BTA",
        linestyle=":", 
        color=hist_color,
        linewidth=1.6
    )
    plt.plot(
        frame_plot["time"], frame_plot["total_pred_demand"],
        label="Total Predictive Demand from EMSs",
        linestyle="--",
        color=pred_color,
        linewidth=1.6
    )
    plt.plot(
        frame_plot["time"], frame_plot["total_real_demand"],
        label="Total Demand with BTA",
        color = real_color,
        linewidth=1.6
    )
    # Visual diff shading
    if {"total_hist_demand", "total_real_demand", "total_pred_demand"}.issubset(frame_plot.columns):
        plt.fill_between(
            frame_plot["time"],
            frame_plot["total_real_demand"],
            frame_plot["total_hist_demand"],
            where=(frame_plot["total_hist_demand"] >= frame_plot["total_pred_demand"]),
            alpha=0.12
        )
        plt.fill_between(
            frame_plot["time"],
            frame_plot["total_real_demand"],
            frame_plot["total_hist_demand"],
            where=(frame_plot["total_hist_demand"] < frame_plot["total_pred_demand"]),
            alpha=0.12
        )
        
plt.ylim(0, 50)
plt.xlabel("Time (hour index)")
plt.ylabel("Power (kWh)")
plt.title(f"Total Consumption With and Without BTA — Window [{WIN_START}, {WIN_END}]")
plt.grid(True, linestyle="--", alpha=0.6)
plt.grid(axis='y', visible=False)
plt.legend()
plt.tight_layout()
plt.show()



