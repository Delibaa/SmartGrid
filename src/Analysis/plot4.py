# Plots for on-chain market info from EMoutput.csv
# Figure 1: Market activity — tradePrice (left y-axis, CHF/kWh) vs share of daily trades (right y-axis, %)
# Figure 2: Gas costs — total_gas_cost_bids / asks / trade (same axes)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sg_config as Config

# ----------------------------
# Config & paths
# ----------------------------
OUTPUTS_DIR = Path(Config.OUTPUTS_DIR)
em_csv = OUTPUTS_DIR / "EMoutput.csv"

# Day indexing (inclusive). Adjust to your config names if needed.
DAY_START = Config.TIME_WINDOW_START
DAY_END   = Config.TIME_WINDOW_END

# Absolute-time base used by EMoutput.csv:
# time column is ((DAY_START - 1) * 24)-based
T0 = (DAY_START - 1) * 24

# Optional: limit plotted days (inclusive). Set to None to plot all
DAY_WINDOW = (DAY_START, DAY_END)  # or None

# ----------------------------
# Load data
# ----------------------------
usecols = [
    "time", "total_demand", "total_supply", "tradePirces", "total_transactions",
    "total_gas_cost_bids", "total_gas_cost_asks", "total_gas_cost_trade",
    "total_trades_from_Market", "total_transactions_from_Market", "total_transactions_from_NG"
]
df = pd.read_csv(em_csv, usecols=usecols)

# Clean types & sort by time
int_cols = ["time", "total_transactions", "total_trades_from_Market",
            "total_transactions_from_Market", "total_transactions_from_NG"]
for c in int_cols:
    if c in df:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

float_cols = ["total_demand", "total_supply", "tradePirces",
              "total_gas_cost_bids", "total_gas_cost_asks", "total_gas_cost_trade"]
for c in float_cols:
    if c in df:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

df = df.sort_values("time").reset_index(drop=True)

# ----------------------------
# Map absolute time -> day index aligned to DAY_START
# day = floor((time - T0)/24) + DAY_START
# ----------------------------
df["day"] = ((df["time"] - T0) // 24) + DAY_START

# Optional day filter
if DAY_WINDOW is not None:
    a, b = DAY_WINDOW
    df = df[(df["day"] >= a) & (df["day"] <= b)].copy()

# ----------------------------
# Prepare data for Subplot 1 (Market Activity)
# - Left axis: tradePirces (CHF/kWh) by time
# - Right axis: per-time share (%) of the day's total_trades_from_Market
# ----------------------------
# Daily total trades
daily_trades = df.groupby("day", as_index=False)["total_trades_from_Market"].sum()
daily_trades = daily_trades.rename(columns={"total_trades_from_Market": "day_total_trades"})

# Merge daily totals back to each timepoint and compute share
act = df[["time", "day", "tradePirces", "total_trades_from_Market"]].merge(
    daily_trades, on="day", how="left"
)
# Avoid division by zero
act["trade_share_pct"] = np.where(
    act["day_total_trades"] > 0,
    (act["total_trades_from_Market"] / act["day_total_trades"]) * 100.0,
    0.0
)

# ----------------------------
# Prepare data for Subplot 2 (Gas Costs)
# ----------------------------
gas_cols = ["total_gas_cost_bids", "total_gas_cost_asks", "total_gas_cost_trade"]
gas = df[["time"] + gas_cols].copy()

# ----------------------------
# Plot: One figure with two subplots
# ----------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16 
plt.rcParams['axes.labelsize'] = 16 
plt.rcParams['legend.fontsize'] = 12 
plt.rcParams['xtick.labelsize'] = 16 
plt.rcParams['ytick.labelsize'] = 16 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

fig.suptitle("On-chain Market Overview", y=1.02)

color1 = "#C9CACB"
color2 = "#F6CC7F"
color3 = "#C57F7F"


# ----- Subplot 1: Market Activity (scatter + inverted right y-axis + size ~ share) -----
# Filter out zero-price points
act_nonzero = act[act["tradePirces"] > 0].copy()

# Compute marker sizes proportional to share (% in [0,100])
# Feel free to tune these numbers
size_min, size_max = 20, 200
share = act_nonzero["trade_share_pct"].clip(lower=0, upper=100).to_numpy()
sizes = size_min + (size_max - size_min) * (share / 100.0)

# Left axis: price scatter
sc_price = ax1.scatter(
    act_nonzero["time"],
    act_nonzero["tradePirces"],
    s=sizes,
    color="#a4757d",
    alpha=0.55,
    linewidths=0,
    label="Trade Price (CHF/kWh)",
    zorder=2
)

# Optional: rolling mean to show trend
window = 24  # 1-day smoothing
if len(act_nonzero) >= window:
    act_nonzero["price_roll"] = act_nonzero["tradePirces"].rolling(window=window, center=True).mean()
    line_roll, = ax1.plot(
        act_nonzero["time"],
        act_nonzero["price_roll"],
        color="#a4757d",
        linewidth=2.2,
        zorder=3
    )
else:
    line_roll = None

# National Grid baseline
baseline_val = Config.NATIONAL_GRID_PRICE
line_baseline = ax1.axhline(
    y=baseline_val, color="#790102", linestyle=":", linewidth=2,
    zorder=1
)

ax1.set_xlabel("Time (hour index)")
ax1.set_ylabel("Trade Price (CHF/kWh)")

# Right axis: share (%) — invert so the line runs along the top area
ax1r = ax1.twinx()
line_share, = ax1r.plot(
    act["time"], act["trade_share_pct"],
    color="#7e7d7e", linestyle="--", linewidth=2,
    label="Share of Daily Trades (%)",
    zorder=2
)
# ax1r.set_ylabel("Share of Daily Trades (%)")
ax1r.set_ylim(0, 100)
ax1r.invert_yaxis()  # <<< invert the right y-axis

# Adjust left y-limits to accommodate baseline and points
ymin_price, ymax_price = ax1.get_ylim()
ax1.set_ylim(min(ymin_price, baseline_val) * 0.9, ymax_price * 1.15)
ax1.set_ylim(0.05,0.4)

# Build a size legend for share (small/medium/large)
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

legend_handles = []
legend_labels  = []

# scatter (price)
legend_handles.append(mlines.Line2D([], [], color="#a4757d", marker='o', linestyle='None', markersize=8, label="Trade Price (scatter)"))
legend_labels.append("Trade Price (scatter)")

# right-axis share line
legend_handles.append(line_share)
legend_labels.append(line_share.get_label())

# size scale markers (mapped to share %)
for lab, pct in [("Share ~ 25%", 25), ("Share ~ 50%", 50), ("Share ~ 75%", 75)]:
    ms = np.sqrt(size_min + (size_max - size_min) * (pct / 100.0))  # legend markersize uses diameter-ish, use sqrt to look balanced
    legend_handles.append(mlines.Line2D([], [], color="#a4757d", marker='o', linestyle='None', markersize=ms/3, alpha=0.55))

ax1.set_title("Market Activity Dynamics")
ax1.legend(legend_handles, legend_labels, loc="center right")

# Optional cosmetics
ax1.grid(True, linestyle="--", alpha=0.25)


# ----- Subplot 2: Gas Costs -----
colors = [color1, color2, color3]  # classic three-color scheme
labels = ["Gas Cost - Bids", "Gas Cost - Asks", "Gas Cost - Trade"]
gas_cols = ["total_gas_cost_bids", "total_gas_cost_asks", "total_gas_cost_trade"]

for col, color, label, linestyle in zip(gas_cols, colors, labels, [":", "--", "-"]):
    ax2.plot(gas["time"], gas[col], label=label, color=color, linestyle = linestyle, linewidth=2)

ax2.set_xlabel("Time (hour index)")
ax2.set_ylabel("Gas Cost (CHF)")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.grid(axis='y', visible=False)
ax2.set_title("Gas Costs Dynamics")
ax2.legend()

# Expand y-limits for gas costs
ymin_gas, ymax_gas = ax2.get_ylim()
ax2.set_ylim(ymin_gas * 0.9, ymax_gas * 1.5)

plt.tight_layout()
plt.show()


