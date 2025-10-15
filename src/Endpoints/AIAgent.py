# AIAgent.py 
# forecast agent and SC logic
from pathlib import Path
from typing import Optional
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sg_config as Config

# ---- CONFIG ----
MODEL_DIR = Config.MODEL_DIR
DEFAULT_SHAVE = 0.3
# ---------------


def _read_series(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["time", "demand"]).dropna().sort_values("time")
    df["time"] = df["time"].astype(int)
    T = int(df["time"].max())
    arr = np.zeros(T, dtype=np.float32)
    arr[df["time"].values - 1] = df["demand"].astype(float).values
    arr[arr < 0] = 0.0
    return arr


def peak_shaving(sequence_24: np.ndarray, shave_ratio: float) -> np.ndarray:
    """
    Iteratively shave peaks: if any single-hour demand exceeds 15% of the day's total,
    reduce that point by `shave_ratio` (clipped to [0.1, 0.3]) and redistribute the
    shaved amount evenly to hours not shaved in that iteration. Repeat until no point
    exceeds 15% or max iterations reached. Returns a 24-length array.

    Energy is conserved by construction. Inputs are not modified in-place.
    """
    y = np.array(sequence_24, dtype=np.float32).copy()
    assert y.shape[0] == 24, "sequence_24 must have 24 values."
    if shave_ratio is None or shave_ratio <= 0:
        return y

    total = float(np.sum(y))
    if total <= 0:
        return y

    threshold = 0.09 * total  # 9% of daily total
    max_iter = 20
    eps = 1e-9

    for _ in range(max_iter):
        # find indices above the 9% cap
        shave_idx = np.where(y > threshold + eps)[0]
        if shave_idx.size == 0:
            break  # no more violations

        # compute shaved total for this iteration
        cuts = y[shave_idx] * shave_ratio
        shaved_total = float(np.sum(cuts))

        # apply shaving
        y[shave_idx] -= cuts

        # redistribute to the others (not shaved this round)
        keep_idx = np.array([i for i in range(24) if i not in shave_idx], dtype=int)
        if keep_idx.size > 0 and shaved_total > 0:
            add_each = shaved_total / keep_idx.size
            y[keep_idx] += add_each
        else:
            # nothing to redistribute to (e.g., all points were above threshold) -> stop
            break

        # (total remains the same; threshold stays 15% of original total)

    # numerical safety
    y[y < 0] = 0.0
    return y

# predict 24h for a specific house/day
def consumption_prediction(
    csv_path: str,
    house_id: int,
    day: int,
    shave_ratio: Optional[float] = None
) -> np.ndarray:
    """
    Predict 24 hours for a specific house/day (1-based).
    Uses ONLY that house's preceding `time_step` hours as history, then rolls forward 24 steps.

    Args:
        csv_path: path to house_{house_id}.csv (columns: time,demand)
        house_id: numeric id, will load MODEL_DIR/house_{house_id}/ artifacts
        day: 1-based day index to forecast (predict hours [(day-1)*24 ... day*24-1])
        shave_ratio: None/<=0 -> no shaving; else 0.1~0.3

    Returns:
        np.ndarray shape (24,)
    """
    # --- load artifacts for this house ---
    model_dir = MODEL_DIR / f"house_{house_id}"   # FIX: ensure 'house_' prefix
    meta_path = model_dir / "meta.json"
    scaler_path = model_dir / "scaler.pkl"
    model_path = model_dir / "model.keras"

    if not meta_path.exists() or not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Artifacts missing for house_{house_id} in {model_dir}")

    meta = json.loads(meta_path.read_text())
    time_step = int(meta["time_step"])
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)

    # --- load series & prepare initial history ---
    series = _read_series(Path(csv_path))
    need_hours = (day - 1) * 24  # history up to the start of target day
    if need_hours < time_step:
        raise ValueError(f"day={day} requires at least time_step={time_step} hours of prior history.")
    if len(series) < need_hours:
        raise ValueError(f"{csv_path}: only {len(series)} hours; need >= {need_hours} for day={day}.")

    hist = series[need_hours - time_step : need_hours].astype(np.float32)  # (time_step,)
    y_pred = []

    # --- roll 24 steps ---
    for _ in range(24):
        x = hist[-time_step:]                                          # (time_step,)
        x_sc = scaler.transform(x.reshape(-1, 1)).reshape(1, time_step, 1)
        y_sc = model.predict(x_sc, verbose=0)                           # (1,1)
        y_hat = scaler.inverse_transform(y_sc.reshape(-1, 1))[0, 0]     # scalar
        y_pred.append(float(y_hat))
        hist = np.append(hist, y_hat)                                   # roll

    y_pred = np.array(y_pred, dtype=np.float32)

    # --- optional peak shaving ---
    if shave_ratio is None:
        shave_ratio = DEFAULT_SHAVE

    y_pred_after_shaving = peak_shaving(y_pred, shave_ratio=float(shave_ratio))

    # return y_pred, y_pred_after_shaving
    return y_pred_after_shaving

# for EMS evaluation
def predict_range(csv_path: str, house_id: int, start_day: int, end_day: int, shave_ratio: Optional[float] = None, chain: bool = False):
    # predict consecutive days
    assert end_day >= start_day, "end_day must be >= start_day"
    num_days = end_day - start_day

    # Get actual series once
    series = _read_series(Path(csv_path))
    start_idx = (start_day - 1) * 24
    end_idx   = (end_day - 1) * 24
    if end_idx > len(series):
        raise ValueError(f"Series length {len(series)} < needed {end_idx} hours for end_day={end_day}.")
    y_true_concat = series[start_idx:end_idx].astype(np.float32)

    if not chain:
        # --- Independent per-day mode: reuse your existing per-day function ---
        preds = []
        preds_after_shaving = []
        for d in range(start_day, end_day):
            y_d, y_d_s = consumption_prediction(csv_path, house_id, d, shave_ratio)
            if y_d.shape[0] != 24:
                raise ValueError(f"Predicted length for day {d} is {y_d.shape[0]}, expected 24.")
            if y_d_s.shape[0] != 24:
                raise ValueError(f"Shaved predicted length for day {d} is {y_d_s.shape[0]}, expected 24.")
            preds.append(y_d.astype(np.float32))
            preds_after_shaving.append(y_d_s.astype(np.float32))

        y_pred_concat = np.concatenate(preds, axis=0)
        y_pred_after_shaving_concat = np.concatenate(preds_after_shaving, axis=0)
        return y_pred_concat, y_true_concat, y_pred_after_shaving_concat

    # --- Chained multi-day roll (uses raw model; no day-by-day resets) ---
    # Load artifacts to roll hour-by-hour:
    model_dir = MODEL_DIR / f"house_{house_id}"
    meta = json.loads((model_dir / "meta.json").read_text())
    time_step = int(meta["time_step"])
    scaler = joblib.load(model_dir / "scaler.pkl")
    model  = load_model(model_dir / "model.keras")

    # Need at least `time_step` hours of history BEFORE the first target hour
    need_hours = (start_day - 1) * 24
    if need_hours < time_step:
        raise ValueError(f"start_day={start_day} requires at least time_step={time_step} hours of prior history.")
    if len(series) < need_hours:
        raise ValueError(f"{csv_path}: only {len(series)} hours; need >= {need_hours}.")

    # Build initial history ending right before start_day
    hist = series[need_hours - time_step : need_hours].astype(np.float32)  # (time_step,)
    total_steps = 24 * num_days
    y_pred_roll = np.empty(total_steps, dtype=np.float32)

    # Roll forward hour-by-hour
    for t in range(total_steps):
        x = hist[-time_step:]
        x_sc = scaler.transform(x.reshape(-1,1)).reshape(1, time_step, 1)
        y_sc = model.predict(x_sc, verbose=0)            # (1,1)
        y_hat = scaler.inverse_transform(y_sc.reshape(-1,1))[0,0]
        y_pred_roll[t] = y_hat
        hist = np.append(hist, y_hat)                    # feed raw prediction (no shaving) into history

    # Apply peak shaving per day on the OUTPUT ONLY (do not affect history)
    if shave_ratio is not None and shave_ratio > 0:
        shaved = []
        for i in range(num_days):
            day_slice = slice(24*i, 24*(i+1))
            shaved.append(peak_shaving(y_pred_roll[day_slice], shave_ratio))
        y_pred_concat = np.concatenate(shaved, axis=0)
    else:
        y_pred_concat = y_pred_roll

    return y_pred_concat, y_true_concat

# plot the prediction accuracy
def plot_range(
    csv_path: str,
    house_id: int,
    start_day: int,
    end_day: int,
    shave_ratio: Optional[float] = None,
    chain: bool = False,
):
    """
    Plot concatenated predictions vs. actuals for [start_day, end_day].

    - If chain=False: per-day independent predictions (history-only per day).
    - If chain=True : true multi-day rolling; predictions feed the next step’s history.

    Shows overall MAE/MAPE across the whole range, and day boundary lines.
    """
    plt.rcParams['font.family'] = 'Times New Roman'

    # colors
    actual_color = "#a00000"  
    pred_color   = "#d8a6a6"
    pred_after_shaving_color = "#027fd8" 

    y_pred, y_true, y_pred_after_shaving = predict_range(
        csv_path=csv_path,
        house_id=house_id,
        start_day=start_day,
        end_day=end_day,
        shave_ratio=shave_ratio,
        chain=chain,
    )

    if house_id <= 5:
        type_label = "consumer"
    else:
        type_label = "prosumer"

    assert y_pred.shape == y_true.shape
    assert y_pred_after_shaving.shape == y_pred.shape
    n_hours = y_true.shape[0]
    num_days = end_day - start_day
    hours = np.arange(n_hours)

    mae = float(np.mean(np.abs(y_pred - y_true)))
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)

    plt.figure(figsize=(12, 5))

    plt.plot(hours, y_true, label=f"Actual (D{start_day}–D{end_day - 1})", 
             linewidth=2.4, color=actual_color)
    plt.plot(hours, y_pred, label=f"Predicted (D{start_day}–D{end_day - 1})", 
             linestyle="--", linewidth=2.4, color=pred_color)
    plt.plot(hours, y_pred_after_shaving, label=f"Predicted After Shaving (D{start_day}–D{end_day - 1})",
             linestyle="-.", linewidth=2.4, color=pred_after_shaving_color)
    plt.fill_between(
        hours, y_pred, y_pred_after_shaving,
        where=(y_pred > y_pred_after_shaving),
        color="lightblue", alpha=0.4, interpolate=True
    )

    # Day boundaries
    for i in range(1, num_days):
        x = 24 * i
        plt.axvline(x=x, color="gray", linestyle=":", alpha=0.7)

    # axis
    tick_positions = [24*i for i in range(num_days)]
    tick_labels = [f"D{start_day + i}" for i in range(num_days)]
    plt.xticks(tick_positions, tick_labels, fontsize=14)
    plt.yticks(fontsize=14)


    # labels & title
    plt.xlabel("Day boundaries", fontsize=16)
    plt.ylabel("Demand", fontsize=16)
    plt.title(
        f"{type_label}_{house_id}: Days {start_day}–{end_day - 1} "
        f"| MAE={mae:.3f}  MAPE={mape:.1f}%", fontsize=18
    )

    plt.grid(False)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_root_c = Path(Config.DATA_DIR) / "consumer" / "historicaldata"
    data_root_p = Path(Config.DATA_DIR) / "prosumer" / "historicaldata"
    
    for i in range(1, 6):
        csv = data_root_c / f"house_{i}.csv"
        plot_range(str(csv), i, Config.TIME_WINDOW_START, Config.TIME_WINDOW_END, DEFAULT_SHAVE, False)
    
    for i in range(6, 16):
        csv = data_root_p / f"house_{i}.csv"
        plot_range(str(csv), i, Config.TIME_WINDOW_START, Config.TIME_WINDOW_END, DEFAULT_SHAVE, False)


