"""
Generate category median weights from OFF database.
Reads all products with both categories and quantity, parses weights,
computes median per category, and saves as JSON for use in run_receipt.py.

Usage:
    python build_category_medians.py              # generate + save
    python build_category_medians.py --verbose    # show all categories
"""

import sqlite3
import re
import json
import sys
from collections import defaultdict
from pathlib import Path

DB_PATH = "off_products.db"
OUTPUT_PATH = "category_medians.json"

# ═══════════════════════ Weight Parsing ═══════════════════════

def parse_weight_grams(qty_str: str) -> float | None:
    """Parse a quantity string into grams. Returns None if unparseable."""
    if not qty_str:
        return None
    q = qty_str.strip()

    m = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(kg|g|lb|lbs|oz|fl oz|ml|l|cl)\b",
        q, re.IGNORECASE
    )
    if not m:
        return None

    value = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()

    conversions = {
        "g": 1, "kg": 1000,
        "oz": 28.35, "fl oz": 28.35,
        "lb": 453.6, "lbs": 453.6,
        "ml": 1, "l": 1000, "cl": 10,
    }

    grams = value * conversions.get(unit, 1)

    # Filter obviously wrong values
    if grams <= 0 or grams > 25000:
        return None

    return grams


def grams_to_best_unit(grams: float) -> tuple[float, str]:
    """Convert grams to the most natural unit for US groceries."""
    oz = grams / 28.35
    lb = grams / 453.6

    # Use lb if >= 2 lb
    if lb >= 2:
        return round(lb, 1), "lb"
    # Use oz for most things
    if oz >= 1:
        return round(oz, 1), "oz"
    # Small items in grams
    return round(grams, 1), "g"


# ═══════════════════════ Category Mapping ═══════════════════════

# Map OFF category tags to simple, readable names that match receipt text
# Only include categories useful for US grocery receipts
CATEGORY_MAP = {
    # Snacks
    "en:cookies": "cookies",
    "en:biscuits": "cookies",
    "en:chocolate-cookies": "cookies",
    "en:chips-and-fries": "chips",
    "en:crisps": "chips",
    "en:potato-chips": "chips",
    "en:tortilla-chips": "chips",
    "en:crackers": "crackers",
    "en:popcorn": "popcorn",
    "en:nuts": "nuts",
    "en:trail-mixes": "trail mix",
    "en:granola-bars": "granola bar",
    "en:energy-bars": "energy bar",
    "en:protein-bars": "protein bar",
    "en:pretzels": "pretzels",

    # Dairy
    "en:milks": "milk",
    "en:whole-milk": "milk",
    "en:semi-skimmed-milk": "milk",
    "en:skimmed-milk": "milk",
    "en:yogurts": "yogurt",
    "en:greek-yogurts": "yogurt",
    "en:cheeses": "cheese",
    "en:cheddar": "cheese",
    "en:mozzarella": "cheese",
    "en:shredded-cheese": "cheese",
    "en:cream-cheeses": "cream cheese",
    "en:butters": "butter",
    "en:creams": "cream",
    "en:ice-creams-and-sorbets": "ice cream",
    "en:ice-creams": "ice cream",
    "en:cottage-cheeses": "cottage cheese",
    "en:sour-cream": "sour cream",

    # Bread / Bakery
    "en:breads": "bread",
    "en:sandwich-bread": "bread",
    "en:white-breads": "bread",
    "en:whole-wheat-breads": "bread",
    "en:tortillas": "tortilla",
    "en:bagels": "bagels",
    "en:rolls": "rolls",
    "en:muffins": "muffins",
    "en:croissants": "croissants",
    "en:pita-breads": "pita",
    "en:english-muffins": "english muffins",
    "en:buns": "buns",

    # Cereals / Grains
    "en:breakfast-cereals": "cereal",
    "en:granolas": "granola",
    "en:oatmeals": "oatmeal",
    "en:pastas": "pasta",
    "en:rices": "rice",
    "en:noodles": "noodles",
    "en:flour": "flour",

    # Canned / Jarred
    "en:canned-foods": "canned",
    "en:canned-vegetables": "canned vegetables",
    "en:canned-beans": "canned beans",
    "en:canned-soups": "canned soup",
    "en:canned-fruits": "canned fruit",
    "en:canned-fish": "canned fish",

    # Frozen
    "en:frozen-foods": "frozen",
    "en:frozen-vegetables": "frozen vegetables",
    "en:frozen-fruits": "frozen fruit",
    "en:frozen-pizzas": "frozen pizza",
    "en:frozen-meals": "frozen meal",
    "en:frozen-desserts": "frozen dessert",
    "en:frozen-fries": "frozen fries",

    # Meat / Protein
    "en:chicken": "chicken",
    "en:chicken-breasts": "chicken breast",
    "en:ground-beef": "ground beef",
    "en:beef": "beef",
    "en:pork": "pork",
    "en:turkey": "turkey",
    "en:bacon": "bacon",
    "en:sausages": "sausage",
    "en:hot-dogs": "hot dog",
    "en:deli-meats": "deli meat",
    "en:ham": "ham",
    "en:salami": "salami",
    "en:salmon": "salmon",
    "en:tuna": "tuna",
    "en:shrimp": "shrimp",
    "en:eggs": "eggs",

    # Produce (per-unit or per-lb)
    "en:apples": "apples",
    "en:bananas": "banana",
    "en:oranges": "orange",
    "en:lemons": "lemon",
    "en:avocados": "avocado",
    "en:tomatoes": "tomato",
    "en:potatoes": "potato",
    "en:onions": "onion",
    "en:carrots": "carrot",
    "en:lettuce": "lettuce",
    "en:broccoli": "broccoli",
    "en:spinach": "spinach",
    "en:cucumbers": "cucumber",
    "en:peppers": "pepper",
    "en:mushrooms": "mushroom",
    "en:berries": "berries",
    "en:strawberries": "strawberries",
    "en:blueberries": "blueberries",
    "en:grapes": "grapes",
    "en:mangoes": "mango",

    # Beverages
    "en:juices": "juice",
    "en:orange-juices": "orange juice",
    "en:apple-juices": "apple juice",
    "en:sodas": "soda",
    "en:waters": "water",
    "en:sparkling-water": "sparkling water",
    "en:energy-drinks": "energy drink",
    "en:sports-drinks": "sports drink",
    "en:coffees": "coffee",
    "en:teas": "tea",
    "en:plant-milks": "plant milk",
    "en:almond-milks": "almond milk",
    "en:oat-milks": "oat milk",
    "en:soy-milks": "soy milk",
    "en:coconut-milks": "coconut milk",

    # Sauces / Condiments
    "en:ketchup": "ketchup",
    "en:mustards": "mustard",
    "en:mayonnaises": "mayonnaise",
    "en:barbecue-sauces": "bbq sauce",
    "en:hot-sauces": "hot sauce",
    "en:salsas": "salsa",
    "en:pasta-sauces": "pasta sauce",
    "en:salad-dressings": "salad dressing",
    "en:soy-sauces": "soy sauce",
    "en:vinegars": "vinegar",
    "en:oils": "oil",
    "en:olive-oils": "olive oil",
    "en:hummus": "hummus",
    "en:peanut-butters": "peanut butter",
    "en:jams": "jam",
    "en:honeys": "honey",
    "en:syrups": "syrup",
    "en:pickles": "pickles",

    # Baking
    "en:sugars": "sugar",
    "en:chocolate": "chocolate",
    "en:chocolate-bars": "chocolate bar",
    "en:baking-mixes": "baking mix",

    # Baby / Misc
    "en:baby-foods": "baby food",
    "en:pet-foods": "pet food",
    "en:soups": "soup",
    "en:instant-noodles": "instant noodles",
    "en:ramen": "ramen",
    "en:waffles": "waffle",
    "en:pancakes": "pancake",
    "en:tortellini": "tortellini",
    "en:ravioli": "ravioli",
    "en:pizzas": "pizza",
    "en:burritos": "burrito",

    # Spreads
    "en:hazelnut-spreads": "hazelnut spread",
    "en:chocolate-spreads": "chocolate spread",
    "en:cream-cheese-spreads": "cream cheese spread",
}


# ═══════════════════════ Main ═══════════════════════

def build_medians(db_path: str, verbose: bool = False) -> dict:
    """
    For each mapped category, collect all product weights (in grams),
    compute the median, and convert to the best display unit.
    """
    conn = sqlite3.connect(db_path)

    # Collect weights per category
    cat_weights: dict[str, list[float]] = defaultdict(list)

    rows = conn.execute(
        "SELECT categories, quantity FROM products WHERE categories != '' AND quantity != ''"
    ).fetchall()

    print(f"Processing {len(rows):,} products with both categories and quantity...")

    for categories_str, qty_str in rows:
        grams = parse_weight_grams(qty_str)
        if grams is None:
            continue

        # A product can have multiple categories
        for tag in categories_str.split(","):
            tag = tag.strip()
            simple_name = CATEGORY_MAP.get(tag)
            if simple_name:
                cat_weights[simple_name].append(grams)

    conn.close()

    # Compute medians
    results = {}
    print(f"\n{'Category':<25} {'Count':>8} {'Median':>10} {'P25':>10} {'P75':>10}")
    print("-" * 65)

    for cat_name in sorted(cat_weights.keys()):
        weights = sorted(cat_weights[cat_name])
        n = len(weights)
        if n < 5:  # skip categories with too few samples
            continue

        median_g = weights[n // 2]
        p25_g = weights[n // 4]
        p75_g = weights[3 * n // 4]

        med_val, med_unit = grams_to_best_unit(median_g)
        p25_val, p25_unit = grams_to_best_unit(p25_g)
        p75_val, p75_unit = grams_to_best_unit(p75_g)

        results[cat_name] = {
            "value": med_val,
            "unit": med_unit,
            "sample_count": n,
            "p25_grams": round(p25_g, 1),
            "median_grams": round(median_g, 1),
            "p75_grams": round(p75_g, 1),
        }

        if verbose or n >= 20:
            print(f"  {cat_name:<23} {n:>8,} {med_val:>8}{med_unit:<2} {p25_val:>8}{p25_unit:<2} {p75_val:>8}{p75_unit:<2}")

    # Also output a simplified version (just value + unit) for direct use
    simple = {}
    for cat_name, data in results.items():
        simple[cat_name] = [data["value"], data["unit"]]

    return {"detailed": results, "simple": simple}


def main():
    verbose = "--verbose" in sys.argv

    db = Path(DB_PATH)
    if not db.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    data = build_medians(DB_PATH, verbose=verbose)

    # Save detailed version
    output = Path(OUTPUT_PATH)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(data['simple'])} categories to {OUTPUT_PATH}")

    # Print the simple dict ready to paste into code
    print(f"\n# ─── Paste this into run_receipt.py as CATEGORY_DEFAULTS ───")
    print("CATEGORY_DEFAULTS = {")
    for cat, (val, unit) in sorted(data["simple"].items()):
        print(f'    "{cat}": ({val}, "{unit}"),')
    print("}")


if __name__ == "__main__":
    main()
