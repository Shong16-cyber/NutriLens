"""
Smart Receipt Analyzer -- Step 2: Moving Average Filtering (Convolution)
=========================================================================
SIGNAL PROCESSING CONCEPT:
  The moving average filter is a discrete-time LTI system.
  
  Impulse response:   h[n] = 1/N  for n = 0, 1, ..., N-1
  Output:             y[n] = x[n] * h[n]  (linear convolution)
  
  Expanding:          y[n] = (1/N) SUM x[n-k]  for k = 0 to N-1
  
  This is EXACTLY the convolution operation from TECHIN 513.
  numpy.convolve() computes this directly.

WHY WE NEED IT:
  Our raw signal x[n] (daily calories from grocery purchases) looks like:
    [0, 0, 5128, 0, 0, 2985, 0, 0, 0, 0, 5378, ...]
  
  ~75% of values are 0 (non-shopping days). This is meaningless for
  nutritional analysis -- nobody eats 5000 cal on shopping day and 0 
  on other days. The purchased food is consumed over the following days.
  
  The moving average spreads each purchase across N days, giving us
  an ESTIMATE of actual daily intake:
    y[n] ≈ "what this person likely ate per day this week"

BANDWIDTH-RESOLUTION TRADE-OFF:
  - Small N (e.g., 3): preserves temporal detail, but still noisy
  - Medium N (e.g., 7): good weekly average, our primary choice
  - Large N (e.g., 14): very smooth, but blurs week-to-week changes
  
  We demonstrate all three to show this trade-off in the report.

INPUT:  daily_nutrition.csv (from simulate_purchases.py)
OUTPUT: smoothed_nutrition.csv, comparison plots (PNG)

Usage:
  python sp_moving_average.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


# ============================================================
# LOAD DATA -- the discrete-time signal x[n]
# ============================================================

print("=" * 70)
print("[CHART]  STEP 2: Moving Average Filtering (Convolution)")
print("=" * 70)

df = pd.read_csv("daily_nutrition.csv")
df["date"] = pd.to_datetime(df["date"])
n_days = len(df)

print(f"\n[OK] Loaded daily_nutrition.csv: {n_days} days")
print(f"   Date range: {df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

# Extract the raw signal x[n] for each nutrient
nutrients = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]
raw = {n: df[n].values for n in nutrients}

shopping_days = np.count_nonzero(raw["calories"])
print(f"   Shopping days: {shopping_days}/{n_days} ({100*shopping_days/n_days:.0f}%)")
print(f"   Non-shopping days (zeros): {n_days - shopping_days}")


# ============================================================
# MOVING AVERAGE FILTER -- the convolution operation
# ============================================================

def moving_average_filter(x, N):
    """
    Apply a moving average filter via convolution.
    
    This implements the discrete-time LTI system:
      h[n] = 1/N  for n = 0, 1, ..., N-1
      y[n] = x[n] * h[n]  (convolution)
    
    We use mode='full' to see the complete convolution output,
    then trim to match the input length (causal: no future data).
    
    Parameters:
      x : numpy array -- input signal x[n]
      N : int -- filter window length (number of taps)
    
    Returns:
      y : numpy array -- filtered output y[n], same length as x
    """
    # Define the impulse response (uniform kernel)
    h = np.ones(N) / N
    
    # Convolve -- this is the core SP operation!
    # mode='full' gives output of length len(x) + len(h) - 1
    y_full = np.convolve(x, h, mode='full')
    
    # Keep only the first len(x) samples (causal filter: uses past data only)
    y = y_full[:len(x)]
    
    return y


# Apply filter with three window sizes to demonstrate the trade-off
WINDOWS = [3, 7, 14]
filtered = {}

print(f"\n[FIX] Applying moving average filter with N = {WINDOWS}:")
print(f"   Convolution: y[n] = x[n] * h[n],  h[n] = 1/N for n=0..N-1\n")

for N in WINDOWS:
    filtered[N] = {}
    for nutrient in nutrients:
        filtered[N][nutrient] = moving_average_filter(raw[nutrient], N)
    
    # Show the effect on calories
    cal_raw_mean = np.mean(raw["calories"])
    cal_filt_mean = np.mean(filtered[N]["calories"])
    cal_filt_nonzero = np.count_nonzero(filtered[N]["calories"] > 1)
    
    print(f"   N={N:>2}: Calorie signal -- "
          f"mean={cal_filt_mean:.0f} cal/day, "
          f"non-zero days: {cal_filt_nonzero}/{n_days}, "
          f"max={np.max(filtered[N]['calories']):.0f}")

# Our primary filter: N=7 (weekly average)
PRIMARY_N = 7
print(f"\n   [OK] Primary filter: N={PRIMARY_N} (weekly moving average)")
print(f"      This spreads each grocery purchase across 7 days,")
print(f"      giving an estimate of daily intake for that week.")


# ============================================================
# SNR ANALYSIS -- quantify the smoothing effect
# ============================================================

print(f"\n[SNR] SNR Analysis (Signal-to-Noise Ratio improvement):")
print(f"   SNR = mean² / variance  (higher = smoother signal)")
print(f"   {'Nutrient':<14} {'Raw SNR':>10} {'N=3':>10} {'N=7':>10} {'N=14':>10}")
print(f"   {'-'*54}")

for nutrient in nutrients:
    x = raw[nutrient]
    # Use only non-trivial signal region (after first purchase)
    first_nz = np.argmax(x > 0)
    x_region = x[first_nz:]
    
    raw_snr = np.mean(x_region)**2 / max(np.var(x_region), 1e-10)
    
    snr_vals = [f"{raw_snr:.3f}"]
    for N in WINDOWS:
        y = filtered[N][nutrient][first_nz:]
        filt_snr = np.mean(y)**2 / max(np.var(y), 1e-10)
        snr_vals.append(f"{filt_snr:.3f}")
    
    print(f"   {nutrient:<14} {snr_vals[0]:>10} {snr_vals[1]:>10} {snr_vals[2]:>10} {snr_vals[3]:>10}")

print(f"\n   -> Higher N = higher SNR = smoother signal (less variance)")
print(f"   -> But also less temporal resolution (blurs rapid changes)")


# DRI thresholds (used in all plots and Step 3)
DRI = {
    "calories": {"target": 1800, "low": 1600, "high": 2000},
    "protein_g": {"target": 50, "low": 46, "high": 56},
    "carbs_g": {"target": 275, "low": 225, "high": 325},
    "fat_g": {"target": 78, "low": 65, "high": 97},
    "fiber_g": {"target": 25, "low": 21, "high": 30},
}


# ============================================================
# PLOT 1: Raw vs. Smoothed Calorie Signal (main figure)
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
dates = df["date"].values

# Top: Raw signal with all three filtered versions
ax = axes[0]
ax.bar(dates, raw["calories"], width=0.8, alpha=0.3, color="gray", label="Raw x[n] (purchase days)")
colors = {"3": "#e74c3c", "7": "#2980b9", "14": "#27ae60"}
for N in WINDOWS:
    ax.plot(dates, filtered[N]["calories"], linewidth=2 if N == 7 else 1.2,
            color=colors[str(N)], label=f"MA(N={N})", 
            linestyle="-" if N == 7 else "--")
ax.set_ylabel("Calories", fontsize=12)
ax.set_title("Moving Average Filter: Raw vs. Smoothed Daily Calorie Signal", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)

# Bottom: Zoom into N=7 with interpretation
ax = axes[1]
y7 = filtered[PRIMARY_N]["calories"]
ax.fill_between(dates, y7, alpha=0.3, color="#2980b9")
ax.plot(dates, y7, color="#2980b9", linewidth=2, label=f"Smoothed y[n] (N={PRIMARY_N})")
# Add DRI reference line (will be detailed in Step 3)
ax.axhspan(DRI["calories"]["low"], DRI["calories"]["high"], alpha=0.08, color="red", label=f"DRI range ({DRI['calories']['low']}-{DRI['calories']['high']})")
ax.axhline(y=DRI["calories"]["target"], color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"DRI: {DRI['calories']['target']} cal/day")
ax.set_ylabel("Calories (smoothed)", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.set_title(f"Weekly Moving Average (N={PRIMARY_N}) with DRI Reference", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig("fig_moving_average_calories.png", dpi=150, bbox_inches="tight")
print(f"\n[FIG] Saved: fig_moving_average_calories.png")


# ============================================================
# PLOT 2: All nutrients -- N=7 smoothed with DRI lines
# ============================================================



DRI_LABELS = {
    "calories": "Calories (kcal)",
    "protein_g": "Protein (g)",
    "carbs_g": "Carbohydrates (g)",
    "fat_g": "Fat (g)",
    "fiber_g": "Fiber (g)",
}

fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
axes_flat = axes.flatten()

for i, nutrient in enumerate(nutrients):
    ax = axes_flat[i]
    y = filtered[PRIMARY_N][nutrient]
    
    ax.fill_between(dates, y, alpha=0.25, color="#2980b9")
    ax.plot(dates, y, color="#2980b9", linewidth=1.8, label=f"MA(N={PRIMARY_N})")
    dri_info = DRI[nutrient]
    ax.axhspan(dri_info["low"], dri_info["high"], alpha=0.08, color="red", label=f"DRI range ({dri_info['low']}-{dri_info['high']})")
    ax.axhline(y=dri_info["target"], color="red", linestyle=":", linewidth=1.5, 
               alpha=0.7, label=f"DRI = {dri_info['target']}")
    
    ax.set_ylabel(DRI_LABELS[nutrient], fontsize=10)
    ax.set_title(DRI_LABELS[nutrient], fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

# Hide the 6th (empty) subplot
axes_flat[5].set_visible(False)

fig.suptitle(f"Smoothed Nutrient Signals (Moving Average N={PRIMARY_N}) vs. DRI Thresholds",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("fig_all_nutrients_smoothed.png", dpi=150, bbox_inches="tight")
print(f"[FIG] Saved: fig_all_nutrients_smoothed.png")


# ============================================================
# PLOT 3: Window size comparison (ablation study figure)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, N in enumerate(WINDOWS):
    ax = axes[i]
    ax.bar(dates, raw["calories"], width=0.8, alpha=0.15, color="gray")
    ax.plot(dates, filtered[N]["calories"], color=colors[str(N)], linewidth=2)
    ax.axhspan(DRI["calories"]["low"], DRI["calories"]["high"], alpha=0.08, color="red")
    ax.axhline(y=DRI["calories"]["target"], color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_title(f"N = {N} ({'3-day' if N==3 else '7-day (weekly)' if N==7 else '14-day (biweekly)'})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    if i == 0:
        ax.set_ylabel("Calories", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=9)

fig.suptitle("Ablation Study: Moving Average Window Size (Bandwidth-Resolution Trade-off)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("fig_window_size_ablation.png", dpi=150, bbox_inches="tight")
print(f"[FIG] Saved: fig_window_size_ablation.png")


# ============================================================
# SAVE SMOOTHED DATA -- for Step 3 (threshold detection)
# ============================================================

# Save the N=7 smoothed signals to CSV
smoothed_df = df[["day_index", "date"]].copy()
for nutrient in nutrients:
    smoothed_df[f"{nutrient}_raw"] = raw[nutrient]
    smoothed_df[f"{nutrient}_ma7"] = np.round(filtered[PRIMARY_N][nutrient], 2)

# Also keep category columns for FFT in Step 4
cat_cols = [c for c in df.columns if c.startswith("cat_")]
for c in cat_cols:
    smoothed_df[c] = df[c]

smoothed_df.to_csv("smoothed_nutrition.csv", index=False)
print(f"\n[SAVE] Saved: smoothed_nutrition.csv (input for Steps 3 & 4)")


# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*70}")
print(f"[INFO]  STEP 2 SUMMARY")
print(f"{'='*70}")
print(f"""
  Signal Processing technique: Moving Average Filter (Convolution)
  
  Mathematical formulation:
    h[n] = 1/N  for n = 0, 1, ..., N-1    (impulse response)
    y[n] = x[n] * h[n]                     (linear convolution)
    y[n] = (1/N) SUM x[n-k], k=0..N-1        (expanded form)
  
  Implementation: numpy.convolve(x, h, mode='full')[:len(x)]
  
  Primary window: N = 7 (weekly average)
  Ablation:       N in {{3, 7, 14}} -- demonstrates bandwidth-resolution trade-off
  
  Key result:
    Raw signal:      {shopping_days} non-zero days out of {n_days} (75% zeros)
    After MA(N=7):   {np.count_nonzero(filtered[7]['calories'] > 1)} non-zero days out of {n_days}
    -> Signal is now interpretable as estimated daily intake
  
  Files generated:
    [FIG] fig_moving_average_calories.png  -- Raw vs. smoothed (main figure)
    [FIG] fig_all_nutrients_smoothed.png   -- All 5 nutrients with DRI lines
    [FIG] fig_window_size_ablation.png     -- N=3 vs N=7 vs N=14 comparison
    [>] smoothed_nutrition.csv           -- Input for Steps 3 & 4
  
  TECHIN 513 connection:
    This is a direct application of convolution from discrete-time
    LTI systems. The uniform kernel h[n] = 1/N is the simplest
    lowpass filter -- it attenuates high-frequency fluctuations
    (day-to-day purchase bursts) while preserving the low-frequency
    trend (weekly nutritional pattern).
""")
