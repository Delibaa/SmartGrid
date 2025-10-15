# plot1
# total demand of all consumers and total supply/demand of all prosumers
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sg_config as Config

# -----------------
# Config
# -----------------
cons_dir = Path(Config.DATA_DIR) / "consumer" / "historicaldata"
pros_dir = Path(Config.DATA_DIR) / "prosumer" / "historicaldata"

# Set time range (inclusive)
time_range = ((Config.TIME_WINDOW_START - 1) * 24 + 1, (Config.TIME_WINDOW_END - 1) * 24)   # time range

# sum helper
def sum_series_by_time(files, value_col):
    series_list = []
    for f in sorted(files):
        df = pd.read_csv(f, usecols=['time', value_col])
        s = df.set_index('time')[value_col]
        series_list.append(s)
    if not series_list:
        return pd.DataFrame(columns=['time', f'total_{value_col}'])
    total = None
    for s in series_list:
        total = s if total is None else total.add(s, fill_value=0.0)
    total = total.sort_index()
    return total.reset_index().rename(columns={value_col: f'total_{value_col}'})

# -----------------
# Aggregate totals
# -----------------
cons_files = list(cons_dir.glob("house_*.csv"))
pros_files = list(pros_dir.glob("house_*.csv"))

total_cons = sum_series_by_time(cons_files, 'demand')          # -> time, total_demand (consumers)
total_pros_dem = sum_series_by_time(pros_files, 'demand')      # -> time, total_demand (prosumers)
total_pros_sup = sum_series_by_time(pros_files, 'supply')      # -> time, total_supply (prosumers)

total_pros = pd.merge(total_pros_dem, total_pros_sup, on='time', how='outer') \
               .fillna(0.0).sort_values('time')

# -----------------
# Apply time range filter
# -----------------
def apply_time_range(df, time_range):
    if df.empty:
        return df
    start, end = time_range
    return df[(df['time'] >= start) & (df['time'] <= end)]

total_cons = apply_time_range(total_cons, time_range)
total_pros = apply_time_range(total_pros, time_range)

# -----------------
# Build a combined table & compute system totals
# -----------------
# Create a dense time index to ensure proper alignment even if some frames are sparse
x_min, x_max = time_range
frame = pd.DataFrame({'time': range(x_min, x_max + 1)})

# Merge all pieces
frame = frame.merge(total_cons.rename(columns={'total_demand': 'total_consumer_demand'}),
                    on='time', how='left')
frame = frame.merge(total_pros[['time', 'total_demand', 'total_supply']].rename(
                        columns={'total_demand': 'total_prosumer_demand',
                                 'total_supply': 'total_prosumer_supply'}),
                    on='time', how='left')

# Fill missing with 0
for c in ['total_consumer_demand', 'total_prosumer_demand', 'total_prosumer_supply']:
    if c in frame:
        frame[c] = frame[c].fillna(0.0)

# System-level totals
frame['total_system_demand'] = frame['total_consumer_demand'] + frame['total_prosumer_demand']
frame['supply_minus_demand'] = frame['total_prosumer_supply'] - frame['total_system_demand']

# -----------------
# Compute common axis limits (include system totals)
# -----------------
y_candidates = []
if 'total_consumer_demand' in frame:
    y_candidates += [frame['total_consumer_demand'].min(), frame['total_consumer_demand'].max()]
if 'total_prosumer_demand' in frame:
    y_candidates += [frame['total_prosumer_demand'].min(), frame['total_prosumer_demand'].max()]
if 'total_prosumer_supply' in frame:
    y_candidates += [frame['total_prosumer_supply'].min(), frame['total_prosumer_supply'].max()]
if 'total_system_demand' in frame:
    y_candidates += [frame['total_system_demand'].min(), frame['total_system_demand'].max()]

if y_candidates:
    y_min = min(y_candidates)
    y_max = max(y_candidates)
else:
    y_min, y_max = 0.0, 1.0  # safe fallback

# -----------------
# Plot: three panels
#   1) Consumers: total demand
#   2) Prosumers: demand & supply
#   3) System: (cons + pros) demand vs pros supply, plus gap shading
# -----------------

# settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 18 
plt.rcParams['axes.labelsize'] = 18 
plt.rcParams['legend.fontsize'] = 14 
plt.rcParams['xtick.labelsize'] = 18 
plt.rcParams['ytick.labelsize'] = 18 

color1 = "#5c95bf"
color2 = "#bf7f5c"

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
ax1, ax2, ax3 = axes

# 1) Consumers
if frame['total_consumer_demand'].notna().any():
    ax1.plot(frame['time'], frame['total_consumer_demand'], label="Total Consumer Demand", color=color1)
ax1.set_xlabel("Time (hour index)")
ax1.set_ylabel("Power (kWh)")
ax1.set_title(f"Consumers: Total Demand")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.grid(axis='y', visible=False)
ax1.legend()

# 2) Prosumers
if frame[['total_prosumer_demand', 'total_prosumer_supply']].notna().any().any():
    ax2.plot(frame['time'], frame['total_prosumer_demand'], label="Prosumer Demand", linestyle="--", color=color1)
    ax2.plot(frame['time'], frame['total_prosumer_supply'], label="Prosumer Supply", linestyle="-.", color=color2)
ax2.set_xlabel("Time (hour index)")
ax2.set_title(f"Prosumers: Demand & Supply")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.grid(axis='y', visible=False)
ax2.legend()

# 3) System comparison
if frame[['total_system_demand', 'total_prosumer_supply']].notna().any().any():
    ax3.plot(frame['time'], frame['total_system_demand'], label="System Demand", linestyle="--", color=color1)
    ax3.plot(frame['time'], frame['total_prosumer_supply'], label="Prosumer Supply", linestyle="-.", color=color2)


ax3.set_xlabel("Time (hour index)")
ax3.set_title(f"System: Demand vs Prosumer Supply")
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.grid(axis='y', visible=False)
ax3.legend()

# Apply same axis ranges
for ax in (ax1, ax2, ax3):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0,60)

plt.tight_layout()
plt.show()



