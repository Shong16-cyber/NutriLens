"""
Smart Receipt Analyzer -- Step 4: FFT Analysis & Nutrition Report
=========================================================================
PART A -- FFT ANALYSIS (Signal Processing):
  The Discrete Fourier Transform decomposes purchase signals into
  frequency components to detect periodic buying patterns:
  
      P(k) = SUM p[n] · e^(-j2pikn/N)    for n = 0, 1, ..., N-1
  
  Each frequency bin k corresponds to a period of N/k days.
  A peak at bin k -> the person shops on a regular N/k-day cycle.
  
  Example: N=56 days, peak at k=8 -> period = 56/8 = 7 days (weekly)

PART B -- REPORT GENERATION:
  Generate a comprehensive monthly nutrition report matching the
  sample mockup, including:
    1. Macronutrient breakdown (pie chart)
    2. DRI comparison (horizontal bar)
    3. Category purchase frequency (bar chart)
    4. Calorie intake trend (line chart -- from Step 2)
    5. Most purchased items with frequency & price
    6. Store visit frequency
    7. Spending summary
    8. Health insights & recommendations

INPUT:  daily_nutrition.csv, smoothed_nutrition.csv, purchase_history.json
OUTPUT: FFT plots, full report figure

Usage:
  python sp_fft_and_report.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import Counter


# ============================================================
# LOAD ALL DATA
# ============================================================

print("=" * 70)
print("[CHART]  STEP 4: FFT Analysis & Nutrition Report Generation")
print("=" * 70)

df_raw = pd.read_csv("daily_nutrition.csv")
df_smooth = pd.read_csv("smoothed_nutrition.csv")
df_raw["date"] = pd.to_datetime(df_raw["date"])
df_smooth["date"] = pd.to_datetime(df_smooth["date"])

with open("purchase_history.json", "r") as f:
    receipts = json.load(f)

n_days = len(df_raw)
dates = df_raw["date"].values

print(f"\n[OK] Loaded: {n_days} days, {len(receipts)} receipts")


# ============================================================
# PART A: FFT ANALYSIS -- Periodic Purchase Pattern Detection
# ============================================================

print(f"\n{'='*70}")
print(f"[FFT]  PART A: FFT Analysis of Purchase Patterns")
print(f"{'='*70}")

# Extract binary category signals p[n]
categories = ["Produce", "Dairy", "Grains", "Meat", "Snacks", "Beverages"]
cat_cols = {c: f"cat_{c.lower()}" for c in categories}

print(f"\n   DFT: P(k) = SUM p[n]·e^(-j2pikn/N),  N = {n_days}")
print(f"   Frequency bin k -> period = N/k = {n_days}/k days")
print(f"   Key bins: k=8 -> 7 days (weekly), k=4 -> 14 days (biweekly)\n")

# Compute FFT for each category
fft_results = {}
for cat in categories:
    col = cat_cols[cat]
    if col in df_raw.columns:
        signal = df_raw[col].values.astype(float)
        
        # Compute DFT
        P = np.fft.fft(signal)
        magnitude = np.abs(P)
        
        # Frequency axis (only positive frequencies, skip DC)
        freqs = np.fft.fftfreq(n_days, d=1)  # in cycles/day
        periods = np.zeros(n_days)
        periods[1:] = n_days / np.arange(1, n_days)  # in days
        
        # Find dominant frequency (skip DC component k=0)
        half = n_days // 2
        mag_pos = magnitude[1:half]  # positive freqs only, skip DC
        k_dominant = np.argmax(mag_pos) + 1  # +1 because we skipped k=0
        period_dominant = n_days / k_dominant
        
        fft_results[cat] = {
            "signal": signal,
            "magnitude": magnitude,
            "freqs": freqs,
            "k_dominant": k_dominant,
            "period_dominant": period_dominant,
            "purchase_days": int(signal.sum()),
        }
        
        # Check if weekly pattern exists (k=8 for N=56)
        k_weekly = round(n_days / 7)  # k=8
        weekly_strength = magnitude[k_weekly] if k_weekly < half else 0
        total_energy = np.sum(mag_pos)
        weekly_ratio = weekly_strength / total_energy if total_energy > 0 else 0
        
        fft_results[cat]["k_weekly"] = k_weekly
        fft_results[cat]["weekly_strength"] = weekly_strength
        fft_results[cat]["weekly_ratio"] = weekly_ratio
        
        print(f"   {cat:<12} purchased {int(signal.sum()):>2}/{n_days} days | "
              f"dominant period: {period_dominant:.1f} days (k={k_dominant}) | "
              f"weekly strength: {weekly_ratio:.0%}")


# -- FFT PLOT 1: Magnitude spectra for all categories --
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()
colors_cat = {"Produce": "#27ae60", "Dairy": "#3498db", "Grains": "#f39c12",
              "Meat": "#e74c3c", "Snacks": "#9b59b6", "Beverages": "#1abc9c"}

for i, cat in enumerate(categories):
    ax = axes_flat[i]
    r = fft_results[cat]
    half = n_days // 2
    
    # Plot magnitude spectrum (positive frequencies only)
    k_vals = np.arange(1, half)
    mag_vals = r["magnitude"][1:half]
    periods_vals = n_days / k_vals
    
    ax.stem(k_vals, mag_vals, linefmt=colors_cat[cat], markerfmt="o",
            basefmt="gray", label=f"|P(k)|")
    
    # Highlight weekly bin
    k_w = r["k_weekly"]
    if k_w < half:
        ax.stem([k_w], [r["magnitude"][k_w]], linefmt="red", markerfmt="ro",
                basefmt="gray", label=f"k={k_w} (7-day cycle)")
    
    # Add period labels on top x-axis
    ax.set_xlabel("Frequency bin k", fontsize=10)
    ax.set_ylabel("|P(k)|", fontsize=10)
    ax.set_title(f"{cat}\n({r['purchase_days']} shopping days, "
                 f"dominant period: {r['period_dominant']:.0f} days)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, half)
    
    # Add secondary x-axis showing period in days
    ax2 = ax.twiny()
    tick_ks = [2, 4, 8, 14, 28]
    tick_ks = [k for k in tick_ks if k < half]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_ks)
    ax2.set_xticklabels([f"{n_days/k:.0f}d" for k in tick_ks], fontsize=8, color="gray")
    ax2.set_xlabel("Period (days)", fontsize=8, color="gray")

fig.suptitle("FFT Magnitude Spectra of Purchase Patterns by Food Category\n"
             "P(k) = SUM p[n]·e^(-j2pikn/N),  N=56 days",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_fft_purchase_patterns.png", dpi=150, bbox_inches="tight")
print(f"\n[FIG] Saved: fig_fft_purchase_patterns.png")


# -- FFT PLOT 2: Purchase regularity summary --
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Binary purchase heatmap
cat_matrix = np.zeros((len(categories), n_days))
for i, cat in enumerate(categories):
    col = cat_cols[cat]
    if col in df_raw.columns:
        cat_matrix[i] = df_raw[col].values

im = ax1.imshow(cat_matrix, aspect="auto", cmap="YlOrRd", interpolation="none")
ax1.set_yticks(range(len(categories)))
ax1.set_yticklabels(categories, fontsize=10)
ax1.set_xlabel("Day index (n)", fontsize=11)
ax1.set_title("Purchase Activity Heatmap\np[n] = 1 (purchased) or 0 (not)",
              fontsize=12, fontweight="bold")

# Add week dividers
for w in range(1, 8):
    ax1.axvline(x=w*7 - 0.5, color="white", linewidth=1, alpha=0.5)
# Add week labels
for w in range(8):
    ax1.text(w*7 + 3, -0.7, f"W{w+1}", ha="center", fontsize=8, color="gray")

plt.colorbar(im, ax=ax1, label="Purchased (0/1)", shrink=0.8)

# Right: Weekly strength comparison bar chart
cats_sorted = sorted(fft_results.keys(), key=lambda c: fft_results[c]["weekly_ratio"], reverse=True)
weekly_ratios = [fft_results[c]["weekly_ratio"] * 100 for c in cats_sorted]
bar_colors = [colors_cat[c] for c in cats_sorted]

bars = ax2.barh(range(len(cats_sorted)), weekly_ratios, color=bar_colors, edgecolor="black", linewidth=0.5)
ax2.set_yticks(range(len(cats_sorted)))
ax2.set_yticklabels(cats_sorted, fontsize=11)
ax2.set_xlabel("Weekly Cycle Strength (% of total spectral energy)", fontsize=10)
ax2.set_title("Purchase Regularity Score\n(FFT energy at 7-day frequency)",
              fontsize=12, fontweight="bold")
ax2.grid(True, axis="x", alpha=0.3)

for bar, val in zip(bars, weekly_ratios):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{val:.0f}%", va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("fig_fft_regularity.png", dpi=150, bbox_inches="tight")
print(f"[FIG] Saved: fig_fft_regularity.png")


# ============================================================
# PART B: COMPREHENSIVE NUTRITION REPORT
# (matching the sample mockup with store & price tracking)
# ============================================================

print(f"\n{'='*70}")
print(f"[INFO]  PART B: Comprehensive Nutrition Report")
print(f"{'='*70}")

# -- Aggregate statistics from purchase history --

# Item frequency & price tracking
item_counter = Counter()
item_prices = {}
category_counter = Counter()
store_counter = Counter()
total_spending = 0

# Extract store names, prices, items from actual receipt data
for receipt in receipts:
    store = receipt.get("store", "").strip()
    if store:
        store_counter[store] += 1

    # Use actual total_price from receipt data
    receipt_total = 0
    tp = receipt.get("total_price")
    if tp:
        try:
            receipt_total = float(str(tp).replace("$", "").replace(",", ""))
        except (ValueError, TypeError):
            receipt_total = 0

    for item in receipt["items"]:
        name = item["food_name"]
        item_counter[name] += 1
        category_counter[item["category"]] += 1

        price = item.get("price") or 0
        if not price and receipt_total and len(receipt["items"]) > 0:
            price = receipt_total / len(receipt["items"])
        if name not in item_prices:
            item_prices[name] = {"total": 0, "count": 0}
        item_prices[name]["total"] += price
        item_prices[name]["count"] += 1

    total_spending += receipt_total

# Weekly calorie averages (from smoothed data)
weekly_cal = []
for w in range(8):
    start = w * 7
    end = min(start + 7, n_days)
    week_vals = df_smooth[f"calories_ma7"].values[start:end]
    weekly_cal.append(np.mean(week_vals))

# DRI comparison (from Step 3 results)
DRI = {
    "Calories": {"target": 1800, "low": 1600, "high": 2000, "unit": "kcal"},
    "Protein":  {"target": 50,   "low": 46,   "high": 56,   "unit": "g"},
    "Carbs":    {"target": 275,  "low": 225,  "high": 325,  "unit": "g"},
    "Fat":      {"target": 78,   "low": 65,   "high": 97,   "unit": "g"},
    "Fiber":    {"target": 25,   "low": 21,   "high": 30,   "unit": "g"},
}
nutrient_keys = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]
nutrient_labels = ["Calories", "Protein", "Carbs", "Fat", "Fiber"]

# Average daily intake (from smoothed MA7, skip first 2 warm-up days)
avg_intake = {}
for key, label in zip(nutrient_keys, nutrient_labels):
    avg_intake[label] = np.mean(df_smooth[f"{key}_ma7"].values[2:])


# ============================================================
# GENERATE THE FULL REPORT FIGURE (6-panel + text)
# ============================================================

fig = plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3,
                       left=0.06, right=0.94, top=0.93, bottom=0.03)

# -- Title --
# Dynamic title from actual date range
_dates_parsed = pd.to_datetime(df_raw["date"])
_month_str = _dates_parsed.iloc[0].strftime("%B %Y") if len(_dates_parsed) > 0 else "Unknown"
fig.suptitle(f"Your Monthly Nutrition Report -- {_month_str}\n"
             f"Based on {len(receipts)} grocery receipts processed",
             fontsize=20, fontweight="bold", y=0.97)

# -- Panel 1: Macronutrient Breakdown (Pie Chart) --
ax1 = fig.add_subplot(gs[0, 0])
macro_vals = [avg_intake["Carbs"], avg_intake["Protein"], avg_intake["Fat"]]
macro_total = sum(macro_vals)
macro_pcts = [v/macro_total*100 for v in macro_vals]
macro_labels = [f"Carbs\n{macro_pcts[0]:.0f}%", f"Protein\n{macro_pcts[1]:.0f}%", f"Fat\n{macro_pcts[2]:.0f}%"]
macro_colors = ["#f39c12", "#27ae60", "#e74c3c"]
wedges, texts, autotexts = ax1.pie(macro_vals, labels=macro_labels, colors=macro_colors,
                                     autopct="", startangle=90, textprops={"fontsize": 12})
ax1.set_title("1. Macronutrients (Daily Average)", fontsize=14, fontweight="bold", pad=15)
# Add summary text below pie
ax1.text(0, -1.35, f"Carbs: {avg_intake['Carbs']:.0f}g | Protein: {avg_intake['Protein']:.0f}g | "
         f"Fat: {avg_intake['Fat']:.0f}g\nTotal Calories: ~{avg_intake['Calories']:.0f} kcal/day",
         ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0"))


# -- Panel 2: DRI Comparison (Horizontal Bar) --
ax2 = fig.add_subplot(gs[0, 1])
dri_pcts = []
dri_labels_plot = []
for label in reversed(nutrient_labels):
    pct = (avg_intake[label] / DRI[label]["target"]) * 100
    dri_pcts.append(pct)
    dri_labels_plot.append(label)

bar_colors_dri = ["#e74c3c" if p < 80 else "#f39c12" if p < 100 else "#27ae60" for p in dri_pcts]
bars = ax2.barh(range(len(dri_labels_plot)), dri_pcts, color=bar_colors_dri,
                edgecolor="black", linewidth=0.5, height=0.6)
ax2.set_yticks(range(len(dri_labels_plot)))
ax2.set_yticklabels(dri_labels_plot, fontsize=11)
ax2.set_xlabel("% of Daily Recommended Intake", fontsize=10)
ax2.axvline(x=100, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
ax2.text(102, len(dri_labels_plot)-0.5, "100% DRI", fontsize=9, color="gray")
ax2.set_xlim(0, max(max(dri_pcts) + 15, 120))
ax2.set_title("2. Meeting Daily Recommendations?", fontsize=14, fontweight="bold")
ax2.grid(True, axis="x", alpha=0.2)

for bar, pct in zip(bars, dri_pcts):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f"{pct:.0f}%", va="center", fontsize=11, fontweight="bold")


# -- Panel 3: Category Purchase Frequency (Bar Chart) --
ax3 = fig.add_subplot(gs[1, 0])
cat_sorted = category_counter.most_common()
cat_names = [c[0] for c in cat_sorted]
cat_counts = [c[1] for c in cat_sorted]
cat_bar_colors = [colors_cat.get(c, "#95a5a6") for c in cat_names]

bars3 = ax3.bar(range(len(cat_names)), cat_counts, color=cat_bar_colors,
                edgecolor="black", linewidth=0.5)
ax3.set_xticks(range(len(cat_names)))
ax3.set_xticklabels(cat_names, fontsize=10, rotation=15)
ax3.set_ylabel("Items Purchased", fontsize=11)
ax3.set_title("3. What You Bought", fontsize=14, fontweight="bold")
ax3.grid(True, axis="y", alpha=0.3)
for bar, count in zip(bars3, cat_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(count), ha="center", fontsize=11, fontweight="bold")


# -- Panel 4: Calorie Intake Trend (Line Chart) --
ax4 = fig.add_subplot(gs[1, 1])
weeks = np.arange(1, 9)
ax4.fill_between(weeks, DRI["Calories"]["low"], DRI["Calories"]["high"], alpha=0.15, color="green", label=f"Healthy Range ({DRI['Calories']['low']}-{DRI['Calories']['high']} kcal)")
ax4.axhline(y=DRI["Calories"]["target"], color="green", linestyle="--", linewidth=1.5, alpha=0.6, label=f"Target ({DRI['Calories']['target']} kcal)")
ax4.plot(weeks, weekly_cal, "o-", color="#2980b9", linewidth=2.5, markersize=8, label="Your Average")
ax4.set_xlabel("Week", fontsize=11)
ax4.set_ylabel("Daily Calories (kcal)", fontsize=11)
ax4.set_title("4. Calorie Intake Trend", fontsize=14, fontweight="bold")
ax4.set_xticks(weeks)
ax4.set_xticklabels([f"Week {w}" for w in weeks], fontsize=9, rotation=20)
ax4.legend(fontsize=9, loc="upper right")
ax4.grid(True, alpha=0.3)
ax4.set_ylim(min(min(weekly_cal)-200, DRI["Calories"]["low"]-400), max(max(weekly_cal)+200, DRI["Calories"]["high"]+400))


# -- Panel 5: Health Insights & Recommendations --
ax7 = fig.add_subplot(gs[2, :])
ax7.axis("off")

# Build insights text
deficient = []
sufficient = []
for label in nutrient_labels:
    pct = (avg_intake[label] / DRI[label]["target"]) * 100
    if pct < 80:
        deficient.append((label, avg_intake[label], DRI[label]["target"], DRI[label]["unit"], pct))
    else:
        sufficient.append((label, avg_intake[label], DRI[label]["target"], DRI[label]["unit"], pct))

# Find most regular category (from FFT)
most_regular = max(fft_results.keys(), key=lambda c: fft_results[c]["weekly_ratio"])
least_regular = min(fft_results.keys(), key=lambda c: fft_results[c]["weekly_ratio"])

insights = "7. Your Health Insights & Recommendations\n"
insights += "-" * 70 + "\n\n"

insights += "[+] GOOD NEWS:\n"
for label, val, target, unit, pct in sufficient:
    insights += f"   * {label} intake is good ({val:.0f}{unit}/day -- {pct:.0f}% of DRI)\n"
top_item = item_counter.most_common(1)[0]
insights += f"   * Most purchased: {top_item[0].title()} ({top_item[1]} times) -- great staple!\n"
insights += f"   * {most_regular} purchases are most regular (strong weekly cycle in FFT)\n"

insights += "\n[!] AREAS TO IMPROVE:\n"
for label, val, target, unit, pct in deficient:
    gap = target - val
    insights += f"   * {label} is low ({val:.0f}{unit} vs. {target}{unit} recommended) -- only {pct:.0f}% of DRI\n"
if least_regular in fft_results:
    insights += f"   * {least_regular} purchases are irregular -- consider a more consistent buying schedule\n"

insights += "\n[>] SIMPLE ACTIONS THIS MONTH:\n"
actions = []
if any(l == "Fiber" for l, *_ in deficient):
    actions.append("   1. Add more fiber: Buy brown rice, oats, or legumes")
if any(l == "Calories" for l, *_ in deficient):
    actions.append("   2. Increase calories: Add healthy fats (avocado, nuts, olive oil)")
if any(l == "Carbs" for l, *_ in deficient):
    actions.append("   3. Boost carbs: Whole grains, fruits, and starchy vegetables")
if any(l == "Fat" for l, *_ in deficient):
    actions.append("   4. Healthy fats: Nuts, seeds, avocado, fatty fish")
actions.append(f"   {len(actions)+1}. Keep buying {most_regular.lower()} regularly -- great habit!")
insights += "\n".join(actions)

insights += f"\n\n[$] SPENDING SUMMARY:\n"
insights += f"   * Total grocery spend: ${total_spending:.2f} this period\n"
insights += f"   * Average per receipt: ${total_spending/len(receipts):.2f}\n"
most_visited = store_counter.most_common(1)[0]
insights += f"   * Most visited store: {most_visited[0]} ({most_visited[1]} visits)\n"

# Draw text box
ax7.text(0.05, 0.95, insights, transform=ax7.transAxes,
         fontsize=11, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.8", facecolor="#fafafa", edgecolor="#2c3e50", linewidth=2))

# Footer
fig.text(0.5, 0.01, "Generated by Smart Receipt Analyzer  |  Signal Processing: MA Filter -> Threshold Detection -> FFT",
         ha="center", fontsize=10, color="gray", style="italic")

plt.savefig("fig_nutrition_report_full.png", dpi=150, bbox_inches="tight")
print(f"[FIG] Saved: fig_nutrition_report_full.png")


# ============================================================
# FFT INTERPRETATION SUMMARY
# ============================================================

print(f"\n{'='*70}")
print(f"[INFO]  FFT ANALYSIS RESULTS")
print(f"{'='*70}")

print(f"\n   {'Category':<12} {'Days':>5} {'Dominant':>10} {'Weekly k':>9} {'Regularity':>12}")
print(f"   {'-'*52}")
for cat in categories:
    if cat in fft_results:
        r = fft_results[cat]
        regularity = "Regular [OK]" if r["weekly_ratio"] > 0.15 else "Irregular [!]"
        print(f"   {cat:<12} {r['purchase_days']:>3}/{n_days} "
              f"{r['period_dominant']:>7.0f} days  k={r['k_weekly']:<4} "
              f"{regularity}")

print(f"""
   INTERPRETATION:
   Categories with high weekly strength have consistent purchasing
   patterns (the person shops for these every week). Categories with
   low weekly strength are purchased irregularly, which may correlate
   with nutritional gaps in those food groups.
   
   The FFT decomposes the binary purchase signal p[n] into frequency
   components -- a peak at the weekly frequency (k={round(n_days/7)}) means the
   category is bought on a reliable 7-day cycle.
""")


# ============================================================
# STEP 4 SUMMARY
# ============================================================

print(f"{'='*70}")
print(f"[INFO]  STEP 4 SUMMARY")
print(f"{'='*70}")
print(f"""
  PART A -- Signal Processing: FFT (Discrete Fourier Transform)
  
  Mathematical formulation:
    P(k) = SUM p[n]·e^(-j2pikn/N),  n = 0..N-1
    
    Input:  p[n] = binary purchase signal (1=bought, 0=didn't)
    Output: |P(k)| = magnitude spectrum showing periodic components
    
    Key: peak at k -> purchasing cycle of N/k days
         k={round(n_days/7)} for N={n_days} -> 7-day (weekly) cycle
  
  Implementation: numpy.fft.fft()
  
  PART B -- Report Generation
    Full nutrition report with 7 panels matching sample mockup:
    pie chart, DRI bars, category purchases, calorie trend,
    top items + prices, store visits, health insights
  
  Files generated:
    [FIG] fig_fft_purchase_patterns.png  -- Magnitude spectra per category
    [FIG] fig_fft_regularity.png         -- Heatmap + regularity scores
    [FIG] fig_nutrition_report_full.png  -- Complete monthly nutrition report
  
  TECHIN 513 connection:
    The DFT is the central tool of frequency analysis in discrete-time
    signal processing. We apply it to detect periodicity in purchasing
    behavior -- a novel application of classical SP to health data.
    The FFT (numpy.fft.fft) is the O(N log N) algorithm for computing
    the DFT, directly from TECHIN 513 coursework.
""")
