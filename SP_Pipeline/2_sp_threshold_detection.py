"""
Smart Receipt Analyzer -- Step 3: DRI Threshold Detection
=========================================================================
SIGNAL PROCESSING CONCEPT:
  Threshold detection (level-crossing detection) is a fundamental SP
  operation: given a signal y[n] and a reference level T, produce a
  binary indicator:
  
      d[n] = 1   if y[n] < T    (deficient)
      d[n] = 0   otherwise       (sufficient)
  
  The cumulative deficiency rate over N samples:
      D = (1/N) SUM d[n]
  
  This tells us: "what fraction of the time is this nutrient below
  the recommended intake?"

WHY FILTERING BEFORE DETECTION MATTERS:
  On raw data:  d[n]=1 on EVERY non-shopping day (75% of days!)
                -> false alarm rate is meaningless
  On smoothed:  d[n]=1 only when the weekly TREND is below DRI
                -> meaningful health insight
  
  This is the same principle as in communications/radar:
  you filter noise BEFORE making threshold decisions to reduce
  false alarm rates. We demonstrate this quantitatively.

DRI THRESHOLDS (Dietary Reference Intakes):
  Based on National Academies of Sciences recommendations for adults.
  We use "adequate intake" or "RDA" values as detection thresholds.

INPUT:  smoothed_nutrition.csv (from sp_moving_average.py)
OUTPUT: threshold detection results, comparison plots, deficiency report

Usage:
  python sp_threshold_detection.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("[FIND]  STEP 3: DRI Threshold Detection")
print("=" * 70)

df = pd.read_csv("smoothed_nutrition.csv")
df["date"] = pd.to_datetime(df["date"])
n_days = len(df)
dates = df["date"].values

print(f"\n[OK] Loaded smoothed_nutrition.csv: {n_days} days")


# ============================================================
# DRI THRESHOLDS -- our detection reference levels T
# ============================================================

DRI = {
    "calories":  {"threshold": 1800, "low": 1600, "high": 2000, "unit": "kcal", "label": "Calories",      "direction": "below"},
    "protein_g": {"threshold": 50,   "low": 46,   "high": 56,   "unit": "g",    "label": "Protein",       "direction": "below"},
    "carbs_g":   {"threshold": 275,  "low": 225,  "high": 325,  "unit": "g",    "label": "Carbohydrates", "direction": "below"},
    "fat_g":     {"threshold": 78,   "low": 65,   "high": 97,   "unit": "g",    "label": "Fat",           "direction": "below"},
    "fiber_g":   {"threshold": 25,   "low": 21,   "high": 30,   "unit": "g",    "label": "Fiber",         "direction": "below"},
}

print(f"\n[INFO] DRI Thresholds (detection levels):")
print(f"   {'Nutrient':<18} {'Threshold':>10}  {'Direction'}")
print(f"   {'-'*45}")
for key, info in DRI.items():
    print(f"   {info['label']:<18} {info['threshold']:>8} {info['unit']}  signal < T -> deficient")


# ============================================================
# THRESHOLD DETECTION FUNCTION
# ============================================================

def threshold_detect(signal, threshold, skip_initial=2):
    """
    Binary threshold detection on a discrete-time signal.
    
    Parameters:
      signal : numpy array -- the input signal y[n]
      threshold : float -- the detection level T
      skip_initial : int -- skip first N samples (filter warm-up period)
    
    Returns:
      d : numpy array -- binary deficiency indicator d[n]
      D : float -- cumulative deficiency rate (fraction below threshold)
      
    Mathematical formulation:
      d[n] = 1  if y[n] < T,  0 otherwise
      D = (1/N_valid) SUM d[n]  for n in valid range
    """
    d = np.zeros(len(signal))
    
    for n in range(len(signal)):
        if n < skip_initial:
            d[n] = 0  # don't count warm-up period
        elif signal[n] < threshold:
            d[n] = 1  # DEFICIENT
        else:
            d[n] = 0  # SUFFICIENT
    
    # Deficiency rate over valid samples
    valid = d[skip_initial:]
    D = np.mean(valid) if len(valid) > 0 else 0
    
    return d, D


# ============================================================
# APPLY DETECTION -- raw vs. filtered comparison
# ============================================================

nutrients = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]

print(f"\n{'='*70}")
print(f"[CHART]  DETECTION RESULTS: Raw vs. Filtered (N=7)")
print(f"{'='*70}")
print(f"\n   WHY THIS MATTERS:")
print(f"   On raw data, every non-shopping day reads 0 -> always 'deficient'")
print(f"   On smoothed data, we detect ACTUAL weekly intake trends\n")

print(f"   {'Nutrient':<16} {'DRI':>6} | {'Raw D':>8}  {'Raw Flag':>10} | {'MA7 D':>8}  {'MA7 Flag':>10}")
print(f"   {'-'*75}")

results = {}

for key in nutrients:
    info = DRI[key]
    T = info["threshold"]
    
    raw_signal = df[f"{key}_raw"].values
    ma7_signal = df[f"{key}_ma7"].values
    
    # Detect on raw signal
    d_raw, D_raw = threshold_detect(raw_signal, T, skip_initial=0)
    
    # Detect on smoothed signal (skip first 2 days for filter warm-up)
    d_ma7, D_ma7 = threshold_detect(ma7_signal, T, skip_initial=2)
    
    # Flag: deficient if D > 0.5 (below threshold more than half the time)
    flag_raw = "[!]  DEFICIENT" if D_raw > 0.5 else "[OK] OK"
    flag_ma7 = "[!]  DEFICIENT" if D_ma7 > 0.5 else "[OK] OK"
    
    results[key] = {
        "info": info,
        "T": T,
        "d_raw": d_raw, "D_raw": D_raw,
        "d_ma7": d_ma7, "D_ma7": D_ma7,
        "raw_signal": raw_signal,
        "ma7_signal": ma7_signal,
        "flag_raw": flag_raw,
        "flag_ma7": flag_ma7,
    }
    
    print(f"   {info['label']:<16} {T:>5}{info['unit'][0]} | "
          f"{D_raw:>7.1%}  {flag_raw:>10} | "
          f"{D_ma7:>7.1%}  {flag_ma7:>10}")


# ============================================================
# KEY INSIGHT: false alarm analysis
# ============================================================

print(f"\n{'-'*70}")
print(f"   KEY INSIGHT -- False Alarm Reduction:")
for key in nutrients:
    r = results[key]
    if r["D_raw"] > r["D_ma7"] + 0.1:
        reduction = r["D_raw"] - r["D_ma7"]
        print(f"   {r['info']['label']:<16} Raw: {r['D_raw']:.0%} deficient -> "
              f"Filtered: {r['D_ma7']:.0%} deficient  "
              f"(v {reduction:.0%} false alarm reduction)")

print(f"\n   The raw signal flags ~75% deficiency for ALL nutrients because")
print(f"   non-shopping days register as 0. After smoothing, deficiency")
print(f"   rates reflect actual weekly intake patterns -- a meaningful metric.")


# ============================================================
# PLOT 1: Threshold detection for each nutrient (main figure)
# ============================================================

fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

colors_ok = "#2ecc71"       # green = above DRI
colors_def = "#e74c3c"      # red = below DRI
colors_signal = "#2980b9"   # blue signal line

for i, key in enumerate(nutrients):
    ax = axes[i]
    r = results[key]
    y = r["ma7_signal"]
    T = r["T"]
    d = r["d_ma7"]
    
    # Plot smoothed signal
    ax.plot(dates, y, color=colors_signal, linewidth=1.8, zorder=3)
    
    # Shade: green where above threshold, red where below
    for n in range(len(dates)):
        if n < 2:
            continue  # skip warm-up
        color = colors_def if d[n] == 1 else colors_ok
        ax.axvspan(dates[max(0,n-1)], dates[n], alpha=0.15, color=color, linewidth=0)
    
    # DRI threshold line
    ax.axhline(y=T, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(dates[-1], T, f"  DRI={T}{r['info']['unit']}", 
            va="center", fontsize=9, color="red", fontweight="bold")
    
    # Labels
    ax.set_ylabel(f"{r['info']['label']} ({r['info']['unit']})", fontsize=11)
    
    # Deficiency rate annotation
    D_pct = f"{r['D_ma7']:.0%}"
    flag = "DEFICIENT" if r["D_ma7"] > 0.5 else "OK"
    flag_color = colors_def if r["D_ma7"] > 0.5 else colors_ok
    ax.text(0.02, 0.92, f"Below DRI: {D_pct} of days -> {flag}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=flag_color, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.grid(True, alpha=0.2)
    ax.set_xlim(dates[0], dates[-1])

axes[0].set_title("DRI Threshold Detection on Smoothed Signals (N=7)",
                   fontsize=14, fontweight="bold")
axes[-1].set_xlabel("Date", fontsize=12)

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=9)

# Legend
ok_patch = mpatches.Patch(color=colors_ok, alpha=0.3, label="Above DRI (sufficient)")
def_patch = mpatches.Patch(color=colors_def, alpha=0.3, label="Below DRI (deficient)")
axes[0].legend(handles=[ok_patch, def_patch], loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("fig_threshold_detection.png", dpi=150, bbox_inches="tight")
print(f"\n[FIG] Saved: fig_threshold_detection.png")


# ============================================================
# PLOT 2: Raw vs. Filtered detection comparison (ablation)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: Raw detection (mostly red -- false alarms)
ax = axes[0]
raw_cal = results["calories"]["raw_signal"]
d_raw = results["calories"]["d_raw"]

ax.bar(dates, raw_cal, width=0.8, color="gray", alpha=0.4)
for n in range(len(dates)):
    color = colors_def if d_raw[n] == 1 else colors_ok
    ax.axvspan(dates[max(0,n-1)], dates[n], alpha=0.12, color=color, linewidth=0)
ax.axhline(y=2000, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
ax.set_title(f"Baseline 3: Raw Signal (NO filtering)\nDeficiency rate: {results['calories']['D_raw']:.0%}",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Calories", fontsize=11)
ax.set_xlabel("Date", fontsize=10)
ax.text(0.02, 0.85, f"Almost ALL days flagged\nas deficient (false alarms\nfrom 0-valued non-shopping days)",
        transform=ax.transAxes, fontsize=9, color=colors_def,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.grid(True, alpha=0.2)

# Right: Filtered detection (meaningful)
ax = axes[1]
ma7_cal = results["calories"]["ma7_signal"]
d_ma7 = results["calories"]["d_ma7"]

ax.fill_between(dates, ma7_cal, alpha=0.3, color=colors_signal)
ax.plot(dates, ma7_cal, color=colors_signal, linewidth=1.8)
for n in range(2, len(dates)):
    color = colors_def if d_ma7[n] == 1 else colors_ok
    ax.axvspan(dates[n-1], dates[n], alpha=0.12, color=color, linewidth=0)
ax.axhline(y=2000, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
ax.set_title(f"Our Method: Smoothed Signal (MA N=7)\nDeficiency rate: {results['calories']['D_ma7']:.0%}",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Calories (smoothed)", fontsize=11)
ax.set_xlabel("Date", fontsize=10)
ax.text(0.02, 0.85, f"Meaningful detection:\nflags weeks where actual\nestimated intake < DRI",
        transform=ax.transAxes, fontsize=9, color=colors_signal,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.grid(True, alpha=0.2)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=9)

fig.suptitle("Baseline 3 Comparison: Why Filtering Before Threshold Detection Matters",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("fig_raw_vs_filtered_detection.png", dpi=150, bbox_inches="tight")
print(f"[FIG] Saved: fig_raw_vs_filtered_detection.png")


# ============================================================
# PLOT 3: Deficiency summary bar chart (for the report)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

labels = [DRI[k]["label"] for k in nutrients]
D_raw_vals = [results[k]["D_raw"] * 100 for k in nutrients]
D_ma7_vals = [results[k]["D_ma7"] * 100 for k in nutrients]

x = np.arange(len(labels))
width = 0.35

bars_raw = ax.bar(x - width/2, D_raw_vals, width, label="Raw (no filtering)",
                  color="gray", alpha=0.6, edgecolor="black", linewidth=0.5)
bars_ma7 = ax.bar(x + width/2, D_ma7_vals, width, label="Smoothed (MA N=7)",
                  color="#2980b9", alpha=0.8, edgecolor="black", linewidth=0.5)

# 50% threshold line
ax.axhline(y=50, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(len(labels)-0.5, 52, "Flag threshold (50%)", fontsize=9, color="red")

# Value labels on bars
for bar in bars_raw:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%", 
            ha="center", va="bottom", fontsize=9, color="gray")
for bar in bars_ma7:
    h = bar.get_height()
    color = colors_def if h > 50 else colors_ok
    ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)

ax.set_ylabel("Deficiency Rate (%)", fontsize=12)
ax.set_title("Nutrient Deficiency Rates: Raw vs. Smoothed Detection", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("fig_deficiency_summary.png", dpi=150, bbox_inches="tight")
print(f"[FIG] Saved: fig_deficiency_summary.png")


# ============================================================
# NUTRITION REPORT CARD -- text output
# ============================================================

print(f"\n{'='*70}")
print(f"[HEALTH]  NUTRITION REPORT CARD (based on smoothed MA7 signals)")
print(f"{'='*70}")

deficient_nutrients = []
ok_nutrients = []

for key in nutrients:
    r = results[key]
    D = r["D_ma7"]
    info = r["info"]
    avg_intake = np.mean(r["ma7_signal"][2:])  # skip warm-up
    T = r["T"]
    gap = T - avg_intake
    
    status = "[!]  BELOW DRI" if D > 0.5 else "[OK] MEETS DRI"
    
    print(f"\n   {info['label']}")
    print(f"   {'-'*40}")
    print(f"   DRI Target:      {T} {info['unit']}/day")
    print(f"   Avg Intake (MA7): {avg_intake:.1f} {info['unit']}/day")
    print(f"   Deficiency Rate: {D:.0%}")
    print(f"   Status:          {status}")
    if gap > 0:
        print(f"   Gap:             {gap:.1f} {info['unit']}/day below target")
    
    if D > 0.5:
        deficient_nutrients.append(info['label'])
    else:
        ok_nutrients.append(info['label'])

print(f"\n{'-'*70}")
print(f"   [!] Areas for Improvement:  {', '.join(deficient_nutrients) if deficient_nutrients else 'None!'}")
print(f"   [OK] Meeting Guidelines:     {', '.join(ok_nutrients) if ok_nutrients else 'None'}")
print(f"{'-'*70}")


# ============================================================
# SAVE DETECTION RESULTS -- for potential downstream use
# ============================================================

detection_df = df[["day_index", "date"]].copy()
for key in nutrients:
    r = results[key]
    detection_df[f"{key}_ma7"] = r["ma7_signal"]
    detection_df[f"{key}_deficient"] = r["d_ma7"].astype(int)

detection_df.to_csv("threshold_detection_results.csv", index=False)
print(f"\n[SAVE] Saved: threshold_detection_results.csv")


# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*70}")
print(f"[INFO]  STEP 3 SUMMARY")
print(f"{'='*70}")
print(f"""
  Signal Processing technique: Threshold Detection (Level-Crossing)
  
  Mathematical formulation:
    d[n] = 1  if y[n] < T,   0 otherwise    (binary indicator)
    D = (1/N) SUM d[n]                         (cumulative deficiency rate)
    Flag as "deficient" if D > 0.5           (below DRI >50% of the time)
  
  Key results:
    Raw detection:      ALL nutrients flagged as deficient (~75%)
                        -> meaningless (false alarms from zero-valued days)
    Filtered detection: meaningful rates based on weekly intake trends
                        -> correctly identifies which nutrients need attention
  
  Deficient (>50%): {', '.join(deficient_nutrients) if deficient_nutrients else 'None'}
  Sufficient:       {', '.join(ok_nutrients) if ok_nutrients else 'None'}
  
  Files generated:
    [FIG] fig_threshold_detection.png       -- Per-nutrient detection (main figure)
    [FIG] fig_raw_vs_filtered_detection.png -- Baseline 3 comparison (ablation)
    [FIG] fig_deficiency_summary.png        -- Bar chart summary
    [>] threshold_detection_results.csv   -- Binary detection results per day
  
  TECHIN 513 connection:
    Threshold detection is a fundamental operation in signal processing
    (used in radar, communications, sensor systems). The key insight is
    that filtering BEFORE detection (our moving average step) dramatically
    reduces false alarm rates -- the same principle as matched filtering
    in detection theory. We demonstrate this quantitatively by comparing
    Baseline 3 (raw detection) vs. our method (filtered detection).
""")
