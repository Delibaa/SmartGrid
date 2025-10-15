# score.py
# Choose next-day leader by comparing real vs predicted consumption accuracy,
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import sg_config as Config

logger = Config.get_logger(os.path.splitext(os.path.basename(__file__))[0])

# helpers
def _day_slice(day):
    if day < 1:
        raise ValueError("day must be >= 1")
    start = (day - Config.TIME_WINDOW_START - 1) * 24
    end = start + 24
    return start, end

# Load a CSV with columns ['time','demand'] and return 24 values for a given day.
def _load_24h(path, day):

    if not path.exists():
        logger.warning("Missing file: %s", path)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error("Failed to read %s: %s", path, e)
        return None

    if "demand" not in df.columns:
        logger.error("File %s missing 'demand' column", path)
        return None

    # Ensure sorted by time if column exists
    if "time" in df.columns:
        df = df.sort_values("time", kind="stable").reset_index(drop=True)

    start, end = _day_slice(day)
    if len(df) < end:
        logger.warning("File %s has only %d rows, needs at least %d for day %d",
                       path, len(df), end, day)
        return None

    # Take exactly 24 points
    segment = df.iloc[start:end]["demand"].to_numpy(dtype=float)
    if segment.size != 24 or np.any(pd.isna(segment)):
        logger.warning("File %s day %d slice invalid size/NaNs", path, day)
        return None
    return segment

# calculation, Root Mean Square Error, pay more attention to the max error
def daily_rmse(real24: np.ndarray, pred24: np.ndarray) -> float:
    return float(np.sqrt(np.mean((real24 - pred24) ** 2)))


# main function of calculating
async def calculateLeaderForNextDay(day):
    
    c_real_dir = Path(os.path.join(Config.DATA_DIR, "consumer", "realdata"))
    c_pred_dir = Path(os.path.join(Config.DATA_DIR, "consumer", "predictiondata"))

    evaluated: List[Dict[str, object]] = []
    best_house: Optional[int] = None
    best_err: Optional[float] = None

    for house in range(0, Config.CONSUMER_NUMBER):
        real_path = c_real_dir / f"house_{house + 1}.csv"
        pred_path = c_pred_dir / f"house_{house + 1}.csv"

        real24 = _load_24h(real_path, day)
        pred24 = _load_24h(pred_path, day)

        if real24 is None:
            logger.info(f"Skip house {house + 1}...")
            continue

        err = daily_rmse(real24, pred24)
        real_total = float(np.sum(real24))
        pred_total = float(np.sum(pred24))

        evaluated.append({
            "house": house + 1,
            "error": err,
            "real_total": real_total,
            "pred_total": pred_total,
        })

        if best_err is None or err < best_err:
            best_err = err
            best_house = house + 1
            best_real_total = real_total
            best_pred_total = float(pred_total)
    
    p_real_dir = Path(os.path.join(Config.DATA_DIR, "prosumer", "realdata"))
    p_pred_dir = Path(os.path.join(Config.DATA_DIR, "prosumer", "predictiondata"))

    for house in range(0, Config.PROSUMER_NUMBER):
        p_real_path = p_real_dir / f"house_{Config.CONSUMER_NUMBER + house + 1}.csv"
        p_pred_path = p_pred_dir / f"house_{Config.CONSUMER_NUMBER + house + 1}.csv"

        real24 = _load_24h(p_real_path, day)
        pred24 = _load_24h(p_pred_path, day)

        if real24 is None:
            logger.info(f"Skip house {Config.CONSUMER_NUMBER + house + 1}...")
            continue

        err = daily_rmse(real24, pred24)
        real_total = float(np.sum(real24))
        pred_total = float(np.sum(pred24))

        evaluated.append({
            "house": Config.CONSUMER_NUMBER + house + 1,
            "error": err,
            "real_total": real_total,
            "pred_total": pred_total,
        })

        if best_err is None or err < best_err:
            best_err = err
            best_house = Config.CONSUMER_NUMBER + house + 1
            best_real_total = real_total
            best_pred_total = float(pred_total)

    if best_house is None:
        logger.warning(f"No valid house data found...")
        return {
            "leader": None,
            "min_error": None,
            "evaluated": evaluated,
            "tx": None,
        }

    logger.info(
        f"Leader for day {day}: house_{best_house} | abs(real_total({round(best_real_total , 3)}) - pred_total({round(best_pred_total , 3)})) = {round(best_err , 3)}"
    )

    return {
        "leader": best_house,
        "min_error": round(best_err,3),
        "evaluated": evaluated,
    }
