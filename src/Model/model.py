# model.py
# the structure of the AI model for prediction
from dataclasses import dataclass
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

@dataclass
class LSTMConfig:
    time_step: int = 72
    units: int = 50
    num_layers: int = 4
    dropout: float = 0.2
    lr: float = 1e-3

# 4 layers LSTM model
def build_model(cfg: LSTMConfig):
    layers = [Input(shape=(cfg.time_step, 1))]
    # 1 layer（return_sequences = True）
    layers += [LSTM(cfg.units, return_sequences=True), Dropout(cfg.dropout)]
    # 2 & 3 layer（return_sequences=True）
    for _ in range(max(1, cfg.num_layers) - 2):
        layers += [LSTM(cfg.units, return_sequences=True)]
        # layers += [Dropout(cfg.dropout)]
    
    # 4 layer
    if cfg.num_layers >= 2:
        layers += [LSTM(cfg.units)]
    # output
    layers += [Dense(1)]

    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate=cfg.lr), loss="mean_squared_error")
    return model

