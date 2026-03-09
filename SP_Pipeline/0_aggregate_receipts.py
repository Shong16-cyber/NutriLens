"""
Smart Receipt Analyzer -- Receipt Aggregator
============================================================
PURPOSE:
  Takes ANY number of processed receipt JSONs (output from
  nutrition_pipeline.py) and produces the two files the SP
  pipeline needs:
    1. daily_nutrition.csv  -- daily nutrient signals x[n]
    2. purchase_history.json -- item-level detail for report

USAGE:
  # Option 1: Pass files as arguments
  python aggregate_receipts.py receipt1.json receipt2.json receipt3.json

  # Option 2: Pass a folder containing all JSONs
  python aggregate_receipts.py ./receipt_outputs/

  # Option 3: Use glob pattern
  python aggregate_receipts.py ./receipts/*.json

REQUIREMENTS:
  Each receipt JSON should have this structure (from nutrition_pipeline.py):
  {
    "items": [
      {
        "original_name": "BNLS CHICKEN BREAST",
        "matched_name": "chicken breast",
        "portion_g": 680,
        "weight_source": "receipt",
        "nutrition": {
          "calories": 1122.0,
          "protein_g": 210.8,
          "carbs_g": 0,
          "fat_g": 24.5,
          "fiber_g": 0,
          "source": "offline",
          "usda_name": "chicken breast"
        },
        "status": "matched"
      },
      ...
    ],
    "totals": {
      "calories": 3500.0,
      "protein": 120.5,
      "carbs": 450.2,
      "fat": 89.3,
      "fiber": 25.1
    }
  }

  OPTIONAL fields (from run_receipt.py header extraction):
    "store_name", "date", "total" (price)

  If "date" is missing, the script will ask you to input it
  or auto-assign dates based on file order.

NOTE:
  Works with any number of receipts (1 to 100+).
  The SP pipeline adapts automatically -- FFT resolution improves
  with more days of data.
"""

import os
import sys
import json
import csv
import glob
import numpy as np
from datetime import datetime, timedelta


# ============================================================
# FOOD CATEGORY MAPPING
# (used to generate binary category signals for FFT)
# ============================================================

CATEGORY_MAP = {
    # Produce
    "banana": "Produce", "apple": "Produce", "avocado": "Produce",
    "strawberries": "Produce", "blueberries": "Produce", "spinach": "Produce",
    "broccoli": "Produce", "tomato": "Produce", "onion": "Produce",
    "potato": "Produce", "carrot": "Produce", "bell pepper": "Produce",
    "orange": "Produce", "lettuce": "Produce", "mushrooms": "Produce",
    "cucumber": "Produce", "asparagus": "Produce", "celery": "Produce",
    "sweet potato": "Produce", "mixed greens": "Produce", "corn": "Produce",
    "zucchini": "Produce", "peach": "Produce", "pear": "Produce",
    "kiwi": "Produce", "cauliflower": "Produce", "watermelon": "Produce",
    "mango": "Produce", "pineapple": "Produce", "lemon": "Produce",
    "lime": "Produce", "grapes": "Produce", "cherries": "Produce",
    "peas": "Produce", "cabbage": "Produce", "eggplant": "Produce",
    "green beans": "Produce", "radish": "Produce", "beet": "Produce",

    # Dairy
    "milk": "Dairy", "egg": "Dairy", "cheese": "Dairy",
    "yogurt": "Dairy", "butter": "Dairy", "sour cream": "Dairy",
    "heavy cream": "Dairy", "cream cheese": "Dairy",

    # Meat & Seafood
    "chicken breast": "Meat", "chicken thigh": "Meat",
    "ground beef": "Meat", "ground turkey": "Meat",
    "salmon": "Meat", "shrimp": "Meat", "tuna": "Meat",
    "tilapia": "Meat", "bacon": "Meat", "ham": "Meat",
    "beef steak": "Meat", "pork chop": "Meat", "sausage": "Meat",

    # Grains
    "bread": "Grains", "rice": "Grains", "pasta": "Grains",
    "oats": "Grains", "flour": "Grains", "tortilla": "Grains",
    "bagel": "Grains", "english muffin": "Grains",

    # Snacks / Processed
    "potato chips plain salted": "Snacks", "corn tortilla chips": "Snacks",
    "cookies chocolate chip": "Snacks", "cookies chocolate sandwich": "Snacks",
    "ice cream premium": "Snacks", "ice cream regular": "Snacks",
    "crackers saltine": "Snacks", "crackers whole wheat": "Snacks",

    # Beverages
    "juice": "Beverages", "water": "Beverages", "coffee": "Beverages",
    "tea": "Beverages", "carbonated cola beverage": "Beverages",
    "sports drink": "Beverages", "energy drink": "Beverages",
}

ALL_CATEGORIES = ["Beverages", "Dairy", "Grains", "Meat", "Produce", "Snacks"]


def guess_category(food_name):
    """Try to match a food name to a category."""
    name_lower = food_name.lower().strip()

    # Exact match
    if name_lower in CATEGORY_MAP:
        return CATEGORY_MAP[name_lower]

    # Partial match (e.g., "organic banana" -> "banana" -> Produce)
    for key, cat in CATEGORY_MAP.items():
        if key in name_lower or name_lower in key:
            return cat

    return "Other"


# ============================================================
# LOAD RECEIPT JSONs
# ============================================================

def collect_json_paths(args):
    """Collect all JSON file paths from command-line arguments."""
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            # Directory: grab all .json files inside
            paths.extend(sorted(glob.glob(os.path.join(arg, "*.json"))))
        elif os.path.isfile(arg) and arg.endswith(".json"):
            paths.append(arg)
        elif "*" in arg or "?" in arg:
            # Glob pattern
            paths.extend(sorted(glob.glob(arg)))
    return paths


def load_receipt(json_path):
    """Load a single receipt JSON and normalize its structure."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract date -- try multiple possible fields
    date_str = None
    for key in ["date", "receipt_date", "purchase_date"]:
        if key in data and data[key]:
            date_str = data[key]
            break

    # Extract store name
    store = data.get("store_name", data.get("store", "Unknown"))

    # Extract total price
    total_price = data.get("total", data.get("total_price", None))
    if total_price:
        try:
            total_price = float(str(total_price).replace("$", "").replace(",", ""))
        except ValueError:
            total_price = None

    # Extract items with nutrition
    items = []
    for item in data.get("items", []):
        if item.get("status") != "matched" or not item.get("nutrition"):
            continue

        n = item["nutrition"]
        food_name = item.get("matched_name", item.get("original_name", "unknown"))
        category = guess_category(food_name)

        items.append({
            "food_name": food_name,
            "original_name": item.get("original_name", food_name),
            "category": category,
            "portion_g": item.get("portion_g", 100),
            "calories": n.get("calories", 0),
            "protein_g": n.get("protein_g", 0),
            "carbs_g": n.get("carbs_g", 0),
            "fat_g": n.get("fat_g", 0),
            "fiber_g": n.get("fiber_g", 0),
            "price": item.get("price", None),
        })

    # Use totals from the JSON if available, otherwise sum items
    totals = data.get("totals", {})
    if not totals or totals.get("calories", 0) == 0:
        totals = {
            "calories": sum(it["calories"] for it in items),
            "protein_g": sum(it["protein_g"] for it in items),
            "carbs_g": sum(it["carbs_g"] for it in items),
            "fat_g": sum(it["fat_g"] for it in items),
            "fiber_g": sum(it["fiber_g"] for it in items),
        }
    # Normalize key names (pipeline uses "protein" not "protein_g" in totals)
    normalized_totals = {
        "calories": totals.get("calories", 0),
        "protein_g": totals.get("protein_g", totals.get("protein", 0)),
        "carbs_g": totals.get("carbs_g", totals.get("carbs", 0)),
        "fat_g": totals.get("fat_g", totals.get("fat", 0)),
        "fiber_g": totals.get("fiber_g", totals.get("fiber", 0)),
    }

    return {
        "source_file": os.path.basename(json_path),
        "date": date_str,
        "store": store,
        "total_price": total_price,
        "items": items,
        "totals": normalized_totals,
    }


# ============================================================
# DATE HANDLING
# ============================================================

def parse_date(date_str):
    """Try multiple date formats."""
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d",
               "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def resolve_dates(receipts):
    """
    Ensure every receipt has a valid date.
    Strategy:
      1. Use date from JSON if parseable
      2. Ask user interactively for missing dates
      3. If non-interactive, auto-assign with 3-5 day gaps
    """
    # First pass: parse known dates
    for r in receipts:
        r["parsed_date"] = parse_date(r["date"]) if r["date"] else None

    # Check how many are missing
    missing = [r for r in receipts if r["parsed_date"] is None]

    if missing:
        print(f"\n[!]  {len(missing)} receipt(s) missing dates:")
        for r in missing:
            print(f"   - {r['source_file']}")

        # Try interactive input
        try:
            print(f"\n   Enter dates (YYYY-MM-DD) or press Enter to auto-assign:\n")
            for r in missing:
                user_input = input(f"   Date for {r['source_file']}: ").strip()
                if user_input:
                    parsed = parse_date(user_input)
                    if parsed:
                        r["parsed_date"] = parsed
                    else:
                        print(f"   Could not parse '{user_input}', will auto-assign")
        except (EOFError, KeyboardInterrupt):
            print("\n   Non-interactive mode, auto-assigning dates...")

    # Auto-assign remaining missing dates
    # Find the earliest known date, or default to today - N*4 days
    known_dates = [r["parsed_date"] for r in receipts if r["parsed_date"]]
    if known_dates:
        base_date = min(known_dates) - timedelta(days=3)
    else:
        base_date = datetime.now() - timedelta(days=len(receipts) * 4)

    still_missing = [r for r in receipts if r["parsed_date"] is None]
    for i, r in enumerate(still_missing):
        r["parsed_date"] = base_date + timedelta(days=i * 4)
        print(f"   Auto-assigned: {r['source_file']} -> {r['parsed_date'].strftime('%Y-%m-%d')}")

    # Sort by date
    receipts.sort(key=lambda r: r["parsed_date"])

    return receipts


# ============================================================
# BUILD DAILY SIGNALS
# ============================================================

def build_daily_signals(receipts):
    """
    Aggregate receipts into daily nutrient signals x[n].
    Returns a DataFrame with one row per day.
    """
    # Determine date range
    dates = [r["parsed_date"] for r in receipts]
    start_date = min(dates)
    end_date = max(dates)
    n_days = (end_date - start_date).days + 1

    # Ensure minimum useful length for FFT (at least 14 days)
    if n_days < 14:
        # Pad to at least 28 days
        n_days = max(28, n_days)
        print(f"   Note: Padded to {n_days} days for meaningful FFT analysis")

    nutrients = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]
    daily = {n: np.zeros(n_days) for n in nutrients}
    cat_daily = {c: np.zeros(n_days) for c in ALL_CATEGORIES}

    for receipt in receipts:
        day_idx = (receipt["parsed_date"] - start_date).days
        if 0 <= day_idx < n_days:
            # Add nutrition totals
            for n in nutrients:
                daily[n][day_idx] += receipt["totals"].get(n, 0)

            # Mark category presence (binary)
            for item in receipt["items"]:
                cat = item.get("category", "Other")
                if cat in cat_daily:
                    cat_daily[cat][day_idx] = 1

    # Build CSV rows
    rows = []
    for d in range(n_days):
        date = start_date + timedelta(days=d)
        row = {
            "day_index": d,
            "date": date.strftime("%Y-%m-%d"),
        }
        for n in nutrients:
            row[n] = round(daily[n][d], 1)
        for c in ALL_CATEGORIES:
            row[f"cat_{c.lower()}"] = int(cat_daily[c][d])
        rows.append(row)

    return rows, n_days, start_date


# ============================================================
# BUILD PURCHASE HISTORY JSON
# ============================================================

def build_purchase_history(receipts):
    """Convert receipts to the format sp_fft_and_report.py expects."""
    history = []
    for i, r in enumerate(receipts):
        entry = {
            "receipt_id": i + 1,
            "date": r["parsed_date"].strftime("%Y-%m-%d"),
            "day_index": (r["parsed_date"] - receipts[0]["parsed_date"]).days,
            "store": r["store"],
            "total_price": r["total_price"],
            "trip_type": "real_receipt",
            "n_items": len(r["items"]),
            "items": r["items"],
            "totals": r["totals"],
        }
        history.append(entry)
    return history


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("[*]  RECEIPT AGGREGATOR -- Build SP Pipeline Input")
    print("=" * 70)

    # Collect JSON files
    if len(sys.argv) < 2:
        print(f"\nUsage:")
        print(f"  python {sys.argv[0]} receipt1.json receipt2.json ...")
        print(f"  python {sys.argv[0]} ./receipt_folder/")
        print(f"  python {sys.argv[0]} ./receipts/*.json")
        print(f"\nNo files provided. Looking for *output*.json in current directory...")
        paths = sorted(glob.glob("*output*.json")) + sorted(glob.glob("*receipt*.json"))
        paths = [p for p in paths if p != "purchase_history.json"]  # exclude our own output
        if not paths:
            print("No receipt JSONs found. Please provide file paths.")
            sys.exit(1)
    else:
        paths = collect_json_paths(sys.argv[1:])

    if not paths:
        print("No JSON files found!")
        sys.exit(1)

    # Remove duplicates, preserve order
    seen = set()
    unique_paths = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            unique_paths.append(p)
    paths = unique_paths

    print(f"\n[>] Found {len(paths)} receipt JSON(s):")
    for p in paths:
        print(f"   - {p}")

    # Load all receipts
    receipts = []
    for p in paths:
        try:
            r = load_receipt(p)
            receipts.append(r)
            n_items = len(r["items"])
            cal = r["totals"]["calories"]
            print(f"   [OK] {r['source_file']}: {n_items} items, {cal:.0f} cal"
                  f"{', store=' + r['store'] if r['store'] != 'Unknown' else ''}"
                  f"{', date=' + r['date'] if r['date'] else ', no date'}")
        except Exception as e:
            print(f"   ❌ {p}: Failed to load -- {e}")

    if not receipts:
        print("\nNo valid receipts loaded!")
        sys.exit(1)

    # Resolve dates
    print(f"\n[DATE] Resolving dates...")
    receipts = resolve_dates(receipts)

    date_range = (receipts[-1]["parsed_date"] - receipts[0]["parsed_date"]).days + 1
    print(f"\n[CHART] Summary:")
    print(f"   Receipts:   {len(receipts)}")
    print(f"   Date range: {receipts[0]['parsed_date'].strftime('%Y-%m-%d')} to "
          f"{receipts[-1]['parsed_date'].strftime('%Y-%m-%d')} ({date_range} days)")
    print(f"   Total items: {sum(len(r['items']) for r in receipts)}")

    # Build daily signals
    rows, n_days, start = build_daily_signals(receipts)
    shopping_days = sum(1 for row in rows if row["calories"] > 0)
    print(f"   Signal length: {n_days} days (shopping: {shopping_days}, non-shopping: {n_days - shopping_days})")

    # Save daily_nutrition.csv
    csv_path = "daily_nutrition.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[SAVE] Saved: {csv_path} ({n_days} rows)")

    # Save purchase_history.json
    history = build_purchase_history(receipts)
    json_path = "purchase_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Saved: {json_path} ({len(history)} receipts)")

    # Next steps
    print(f"\n{'='*70}")
    print(f"[OK] Ready for SP pipeline! Run in order:")
    print(f"   1. python sp_moving_average.py")
    print(f"   2. python sp_threshold_detection.py")
    print(f"   3. python sp_fft_and_report.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
