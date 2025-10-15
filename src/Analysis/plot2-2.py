# plot2-2
# single house: historical vs predicted vs real, both cropped to the Config window
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sg_config as Config  # uses Config.DATA_DIR, TIME_WINDOW_START, TIME_WINDOW_END

# ----------------------------
# USER SETTINGS
# ----------------------------
HOUSE_ID = 1              # which house to plot
KIND     = "consumer"     # "consumer" or "prosumer"

# Optional: show/hide which series
SHOW_HIST = True
SHOW_PRED = True
SHOW_REAL = True

# Optional y-limit (None to auto)
Y_LIM = None  # e.g., (0, 50)

# settings
plt.figure(figsize=(12, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 18 
plt.rcParams['axes.labelsize'] = 18 
plt.rcParams['legend.fontsize'] = 14 
plt.rcParams['xtick.labelsize'] = 18 
plt.rcParams['ytick.labelsize'] = 18 

# Colors
COLOR_HIST = "#7e7d7e"
COLOR_PRED = "#5c95cf"
COLOR_REAL = "#ee7b15"


# ----------------------------
# Helpers
# ----------------------------
def _file(kind: str, bucket: str, house_id: int) -> Path:
    """
    kind: "consumer" | "prosumer"
    bucket: "historicaldata" | "predictiondata" | "realdata"
    """
    return Path(Config.DATA_DIR) / kind / bucket / f"house_{house_id}.csv"


def _read_one(csv_path: Path, time_col: str = "time", val_col: str = "demand") -> pd.DataFrame:
    """
    Return DataFrame with columns ["time","demand"], sorted, numeric only.
    If file missing or bad, return empty DF with same columns.
    """
    if not csv_path.exists():
        return pd.DataFrame(columns=[time_col, val_col])

    try:
        df = pd.read_csv(csv_path, usecols=[time_col, val_col])
    except Exception:
        return pd.DataFrame(columns=[time_col, val_col])

    # clean
    df = df[pd.to_numeric(df[time_col], errors="coerce").notna()]
    df = df[pd.to_numeric(df[val_col],  errors="coerce").notna()]
    df[time_col] = df[time_col].astype(int)
    df[val_col]  = df[val_col].astype(float)
    df = df.sort_values(time_col).reset_index(drop=True)
    return df[[time_col, val_col]]


def _crop_window(df: pd.DataFrame,
                 start_day: int,
                 end_day: int,
                 time_col: str = "time") -> pd.DataFrame:
    """
    Keep rows whose `time` is within [WIN_START, WIN_END], inclusive.
    """
    if df.empty:
        return df
    win_start = (start_day - 1) * 24 + 1   # 1-based hour index start
    win_end   = (end_day   - 1) * 24       # 1-based hour index end
    df = df[(df[time_col] >= win_start) & (df[time_col] <= win_end)].copy()
    return df.sort_values(time_col).reset_index(drop=True)


def _merge_on_time(hist: pd.DataFrame,
                   pred: pd.DataFrame,
                   real: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join three frames on 'time'. Result columns:
    ['time', 'hist', 'pred', 'real'] with NaNs filled to 0.
    """
    frame = pd.DataFrame({"time": pd.Series(dtype=int)})

    if not hist.empty:
        h = hist.rename(columns={"demand": "hist"})
        frame = h if frame.empty else frame.merge(h, on="time", how="outer")

    if not pred.empty:
        p = pred.rename(columns={"demand": "pred"})
        frame = p if frame.empty else frame.merge(p, on="time", how="outer")

    if not real.empty:
        r = real.rename(columns={"demand": "real"})
        frame = r if frame.empty else frame.merge(r, on="time", how="outer")

    if frame.empty:
        return pd.DataFrame(columns=["time", "hist", "pred", "real"])

    for col in ["hist", "pred", "real"]:
        if col in frame:
            frame[col] = frame[col].fillna(0.0)
    return frame.sort_values("time").reset_index(drop=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.size == 0 or y_pred.size == 0 or y_true.shape != y_pred.shape:
        return {"MAE": np.nan, "MAPE_%": np.nan}
    mae = float(np.mean(np.abs(y_pred - y_true)))
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)
    return {"MAE": mae, "MAPE_%": mape}


# ----------------------------
# Main plotting
# ----------------------------
def main(house_id: int = HOUSE_ID, kind: str = KIND):
    # Load three series
    hist_df = _read_one(_file(kind, "historicaldata", house_id))
    pred_df = _read_one(_file(kind, "predictiondata",  house_id))
    real_df = _read_one(_file(kind, "realdata",        house_id))

    # Crop by Config window
    hist_df = _crop_window(hist_df, Config.TIME_WINDOW_START, Config.TIME_WINDOW_END)
    pred_df = _crop_window(pred_df, Config.TIME_WINDOW_START, Config.TIME_WINDOW_END)
    real_df = _crop_window(real_df, Config.TIME_WINDOW_START, Config.TIME_WINDOW_END)

    # Merge
    frame = _merge_on_time(hist_df, pred_df, real_df)

    # Plot
    plt.figure(figsize=(12, 6))

    # Lines
    if SHOW_HIST and "hist" in frame:
        plt.plot(frame["time"], frame["hist"], label="Demand without BTA", linestyle=":", color=COLOR_HIST, linewidth=2)
    if SHOW_PRED and "pred" in frame:
        plt.plot(frame["time"], frame["pred"], label="Predictive Demand of EMS1",  linestyle="--", color=COLOR_PRED, linewidth=2)
    if SHOW_REAL and "real" in frame:
        plt.plot(frame["time"], frame["real"], label="Demand with BTA",  color=COLOR_REAL, linewidth=2)
    plt.fill_between(
        frame["time"],
        frame["real"],
        frame["hist"],
        where=(frame["real"] >= frame["hist"]),
        alpha=0.15,
        interpolate=True
    )

    plt.fill_between(
        frame["time"],
        frame["real"],
        frame["hist"],
        where=(frame["real"] < frame["hist"]),
        alpha=0.15,
        interpolate=True
    ) 


    # Metrics (only if both present)
    subtitle = []
    if "real" in frame and "pred" in frame and SHOW_PRED and SHOW_REAL:
        m_pred = _metrics(frame["real"].to_numpy(), frame["pred"].to_numpy())
        subtitle.append(f"Pred vs Real — MAE={m_pred['MAE']:.3f}, MAPE={m_pred['MAPE_%']:.1f}%")
    if "real" in frame and "hist" in frame and SHOW_HIST and SHOW_REAL:
        m_hist = _metrics(frame["real"].to_numpy(), frame["hist"].to_numpy())
        subtitle.append(f"Hist vs Real — MAE={m_hist['MAE']:.3f}, MAPE={m_hist['MAPE_%']:.1f}%")

    # Title & axes
    win_start = (Config.TIME_WINDOW_START - 1) * 24 + 1
    win_end   = (Config.TIME_WINDOW_END   - 1) * 24
    plt.title(f"{kind.capitalize()} house_{house_id} with and without BTA — Window [{win_start}, {win_end}]")
    plt.xlabel("Time (hour index)")
    plt.ylabel("Power (kWh)")

    # Optional y-limit
    if Y_LIM is not None:
        plt.ylim(*Y_LIM)

    # Style
    plt.ylim(0, 20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.grid(axis='y', visible=False)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
