"""
build_usda_db.py — Build USDA nutrition SQLite database from SR Legacy CSV files
================================================================================
Reads:
  data/FoodData_Central_sr_legacy_food_csv_2018-04/
    - food.csv              (fdc_id, description, food_category_id)
    - food_nutrient.csv     (fdc_id, nutrient_id, amount)
    - nutrient.csv          (id, name, unit_name)
    - food_category.csv     (id, description)
    - food_portion.csv      (fdc_id, gram_weight, portion_description)

Produces:
  data/db/usda_sr.db

Usage:
  cd OFFAPI_test
  python scripts/build_usda_db.py

The resulting database has ~7,800 foods with per-100g nutrition values
for: energy_kcal, fat, saturated_fat, carbohydrates, sugars, fiber,
proteins, salt, sodium — matching the same columns as off_products.db.
"""

import csv
import sqlite3
import os
import sys
from pathlib import Path
from collections import defaultdict

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "FoodData_Central_sr_legacy_food_csv_2018-04"
DB_PATH = PROJECT_ROOT / "data" / "db" / "usda_sr.db"

# ── Nutrient IDs we care about (from USDA nutrient.csv) ──
# These are the standard USDA nutrient IDs for our 9 target nutrients
TARGET_NUTRIENTS = {
    1008: "energy_kcal",       # Energy (kcal)
    1004: "fat",               # Total lipid (fat)
    1258: "saturated_fat",     # Fatty acids, total saturated
    1005: "carbohydrates",     # Carbohydrate, by difference
    2000: "sugars",            # Sugars, total including NLEA
    1079: "fiber",             # Fiber, total dietary
    1003: "proteins",          # Protein
    1093: "sodium",            # Sodium, Na (mg)
}
# salt = sodium * 2.5 / 1000 (we compute this)


def main():
    print("=" * 60)
    print("  USDA SR Legacy Database Builder")
    print("=" * 60)

    # Verify data directory
    if not DATA_DIR.exists():
        print(f"\nERROR: Data directory not found: {DATA_DIR}")
        print(f"Download SR Legacy CSV from:")
        print(f"  https://fdc.nal.usda.gov/download-datasets/")
        sys.exit(1)

    # Check required files
    required = ["food.csv", "food_nutrient.csv", "nutrient.csv"]
    for f in required:
        if not (DATA_DIR / f).exists():
            print(f"ERROR: Missing required file: {f}")
            sys.exit(1)
        print(f"  Found: {f}")

    # ── Step 1: Load nutrient definitions ──
    print(f"\n[1/6] Loading nutrient definitions...")
    nutrient_names = {}
    with open(DATA_DIR / "nutrient.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row["id"])
            if nid in TARGET_NUTRIENTS:
                nutrient_names[nid] = row["name"]
                print(f"  {nid}: {row['name']} ({row['unit_name']})")

    # Also check for alternative sugar nutrient ID
    # Some SR Legacy entries use 1063 (Sugars, total) instead of 2000
    SUGAR_ALT = 1063
    with open(DATA_DIR / "nutrient.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["id"]) == SUGAR_ALT:
                print(f"  {SUGAR_ALT}: {row['name']} (alternative sugar ID)")

    # ── Step 2: Load food list ──
    print(f"\n[2/6] Loading food list...")
    foods = {}  # fdc_id -> {description, category_id}
    with open(DATA_DIR / "food.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fdc_id = int(row["fdc_id"])
            foods[fdc_id] = {
                "description": row.get("description", ""),
                "category_id": row.get("food_category_id", ""),
            }
    print(f"  Loaded {len(foods)} foods")

    # ── Step 3: Load categories ──
    print(f"\n[3/6] Loading food categories...")
    categories = {}
    cat_file = DATA_DIR / "food_category.csv"
    if cat_file.exists():
        with open(cat_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                categories[row["id"]] = row.get("description", "")
        print(f"  Loaded {len(categories)} categories")
    else:
        print("  food_category.csv not found, skipping categories")

    # ── Step 4: Load food portions (for serving size info) ──
    print(f"\n[4/6] Loading food portions...")
    portions = {}  # fdc_id -> gram_weight (first/default portion)
    portion_file = DATA_DIR / "food_portion.csv"
    if portion_file.exists():
        with open(portion_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fdc_id = int(row["fdc_id"])
                gram_weight = row.get("gram_weight", "")
                if fdc_id not in portions and gram_weight:
                    try:
                        portions[fdc_id] = float(gram_weight)
                    except ValueError:
                        pass
        print(f"  Loaded portions for {len(portions)} foods")

    # ── Step 5: Load nutrient values ──
    print(f"\n[5/6] Loading nutrient values (this may take a moment)...")
    # Build: fdc_id -> {nutrient_key: amount}
    nutrition = defaultdict(dict)
    all_target_ids = set(TARGET_NUTRIENTS.keys()) | {SUGAR_ALT}
    row_count = 0

    with open(DATA_DIR / "food_nutrient.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            nid = int(row["nutrient_id"])
            if nid not in all_target_ids:
                continue
            fdc_id = int(row["fdc_id"])
            try:
                amount = float(row["amount"])
            except (ValueError, KeyError):
                continue

            if nid in TARGET_NUTRIENTS:
                key = TARGET_NUTRIENTS[nid]
                nutrition[fdc_id][key] = amount
            elif nid == SUGAR_ALT:
                # Use alternative sugar ID only if primary not present
                if "sugars" not in nutrition[fdc_id]:
                    nutrition[fdc_id]["sugars"] = amount

    print(f"  Scanned {row_count} nutrient rows")
    print(f"  Found nutrition data for {len(nutrition)} foods")

    # ── Step 6: Build SQLite database ──
    print(f"\n[6/6] Building SQLite database...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"  Removed existing {DB_PATH.name}")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE products (
            fdc_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT DEFAULT '',
            serving_g REAL,
            energy_kcal REAL,
            fat REAL,
            saturated_fat REAL,
            carbohydrates REAL,
            sugars REAL,
            fiber REAL,
            proteins REAL,
            salt REAL,
            sodium REAL
        )
    """)

    # Create indexes for fast lookup
    cur.execute("CREATE INDEX idx_usda_name ON products (product_name COLLATE NOCASE)")
    cur.execute("CREATE INDEX idx_usda_name_lower ON products (LOWER(product_name))")
    cur.execute("CREATE INDEX idx_usda_category ON products (category)")

    inserted = 0
    skipped = 0

    for fdc_id, food in foods.items():
        desc = food["description"]
        if not desc:
            skipped += 1
            continue

        cat_id = food["category_id"]
        category = categories.get(str(cat_id), "")

        n = nutrition.get(fdc_id, {})
        if not n:
            skipped += 1
            continue

        # Compute salt from sodium (sodium is in mg, salt = sodium * 2.5 / 1000 for g)
        sodium_mg = n.get("sodium")
        sodium_g = round(sodium_mg / 1000, 4) if sodium_mg is not None else None
        salt_g = round(sodium_mg * 2.5 / 1000, 4) if sodium_mg is not None else None

        serving = portions.get(fdc_id)

        cur.execute("""
            INSERT INTO products 
            (fdc_id, product_name, category, serving_g,
             energy_kcal, fat, saturated_fat, carbohydrates, sugars, fiber,
             proteins, salt, sodium)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fdc_id, desc, category, serving,
            n.get("energy_kcal"),
            n.get("fat"),
            n.get("saturated_fat"),
            n.get("carbohydrates"),
            n.get("sugars"),
            n.get("fiber"),
            n.get("proteins"),
            salt_g,
            sodium_g,
        ))
        inserted += 1

    conn.commit()

    # ── Summary ──
    cur.execute("SELECT COUNT(*) FROM products")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM products WHERE energy_kcal IS NOT NULL")
    with_kcal = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT category) FROM products WHERE category != ''")
    n_cats = cur.fetchone()[0]

    # Sample entries
    print(f"\n  Sample entries:")
    for row in cur.execute(
        "SELECT product_name, category, energy_kcal, proteins, fat, carbohydrates FROM products LIMIT 10"
    ):
        print(f"    {row[0][:40]:<42} cat={row[1][:15]:<17} kcal={row[2]}  prot={row[3]}  fat={row[4]}  carb={row[5]}")

    # Category breakdown
    print(f"\n  Categories:")
    for row in cur.execute(
        "SELECT category, COUNT(*) as cnt FROM products WHERE category != '' GROUP BY category ORDER BY cnt DESC LIMIT 15"
    ):
        print(f"    {row[0]:<35} {row[1]} foods")

    conn.close()

    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"{'=' * 60}")
    print(f"  Database: {DB_PATH}")
    print(f"  Size:     {db_size_mb:.1f} MB")
    print(f"  Foods:    {total} (inserted {inserted}, skipped {skipped})")
    print(f"  With kcal: {with_kcal}")
    print(f"  Categories: {n_cats}")
    print(f"\n  Next: integrate into run_receipt.py as USDA fallback")


if __name__ == "__main__":
    main()