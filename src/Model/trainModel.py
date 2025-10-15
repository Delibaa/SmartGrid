# trainModel.py
# training the LSTM model for each endpoint based on the historical consumption data
import os, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sg_config as Config
from Model.model import LSTMConfig, build_model

# ---------------- CONFIG ----------------
OUT_DIR = Config.MODEL_DIR

TIME_STEP = 72      # past 7 days
VAL_DAYS  = 7        # last 7 days for validation
BATCH     = 32
EPOCHS    = 50
LR        = 1e-3

LSTM_UNITS   = 50
LSTM_LAYERS  = 4
DROPOUT      = 0.2
SEED         = 42

logger = Config.get_logger(os.path.splitext(os.path.basename(__file__))[0])
# ----------------------------------------

# set random seed for reproducibility
def set_seed(seed=SEED):
    import random, tensorflow as tf
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# load data for one house
def load_house_series(csv_path):
    df = pd.read_csv(csv_path, usecols=["time", "demand"]).dropna().sort_values("time")
    df["time"] = df["time"].astype(int)
    T = int(df["time"].max())
    arr = np.zeros(T, dtype=np.float32)
    arr[df["time"].values - 1] = df["demand"].astype(float).values
    arr[arr < 0] = 0.0
    return arr

# Build sliding windows one-step ahead within [start, end) slice.
def make_windows(series, time_step, start, end):
    s = series[start:end]
    X, y = [], []
    if len(s) < time_step + 1:
        return np.empty((0, time_step, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    for t in range(time_step, len(s)):
        X.append(s[t-time_step:t])
        y.append(s[t])
    X = np.asarray(X, dtype=np.float32)[..., None]  # (N, time_step, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y

# plot training and validation losses
def plot_losses(train_hist, out_dir: Path):
    plt.figure(figsize=(7,5))
    plt.plot(train_hist.history["loss"], label="train")
    if "val_loss" in train_hist.history:
        plt.plot(train_hist.history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.title("Training vs Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.6); plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

# train model for one house
def train_consumption_model(house_id, type):
    set_seed()

    # load path
    out_dir = Config.MODEL_DIR / f"house_{house_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_csv = Config.DATA_DIR / type / "historicaldata" / f"house_{house_id}.csv"
    if not data_csv.exists():
        logger.warning(f"[Skip] file not found: {data_csv}")
        return

    # load data
    s = load_house_series(data_csv)

    # sequencen slice
    min_len = len(s)
    val_hours = VAL_DAYS * 24
    train_end = min_len - (val_hours + 1)
    if train_end <= TIME_STEP + 1:
        logger.warning(f"[Skip] not enough data for house_{house_id} with TIME_STEP={TIME_STEP} and VAL_DAYS={VAL_DAYS}")
        return

    # fit
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(s[:train_end].reshape(-1, 1))

    # make windows for training
    X_tr, y_tr = make_windows(s, TIME_STEP, 0, train_end)
    if X_tr.size == 0:
        logger.warning(f"[Skip] no train windows for house_{house_id}")
        return
    X_tr = scaler.transform(X_tr.reshape(-1, 1)).reshape(-1, TIME_STEP, 1)
    y_tr = scaler.transform(y_tr)

    # make windows for valiadation
    X_va, y_va = make_windows(s, TIME_STEP, train_end, min_len)
    if X_va.size > 0:
        X_va = scaler.transform(X_va.reshape(-1, 1)).reshape(-1, TIME_STEP, 1)
        y_va = scaler.transform(y_va)
        val_data = (X_va, y_va)
        # callbacks = [
        #     EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        #     ModelCheckpoint(filepath=str(out_dir / "best.keras"),
        #                     monitor="val_loss", save_best_only=True, save_weights_only=False),
        # ]
    else:
        val_data = None
        logger.warning(f"[Warn] no val windows for house_{house_id}, training without validation")

    # build and train the model
    cfg = LSTMConfig(time_step=TIME_STEP, units=LSTM_UNITS, num_layers=LSTM_LAYERS, dropout=DROPOUT, lr=LR)
    model = build_model(cfg)
    model.summary()

    history = model.fit(
        X_tr, y_tr,
        validation_data=val_data,
        epochs=EPOCHS,
        batch_size=BATCH,
        shuffle=True,
        verbose=2,
    )

    # save
    model.save(out_dir / "model.keras")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    meta = dict(
        time_step=TIME_STEP,
        lstm_units=LSTM_UNITS,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        lr=LR,
        val_days=VAL_DAYS,
        file=str(data_csv),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # training data
    plot_losses(history, out_dir)
    logger.info(f"[OK] trained model for house_{house_id}, saved to {out_dir}")

# main
def main():

    logger.info(f"Training models for {Config.CONSUMER_NUMBER} consumers and {Config.PROSUMER_NUMBER} prosumers...")
    for i in range(1, Config.CONSUMER_NUMBER + 1):
        train_consumption_model(i, "consumer")
    
    for i in range(Config.CONSUMER_NUMBER + 1, Config.CONSUMER_NUMBER + Config.PROSUMER_NUMBER + 1):
        train_consumption_model(i, "prosumer")
    
    logger.info("  ✔ Training model finished...")

if __name__ == "__main__":
    main()
