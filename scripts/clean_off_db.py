"""
OFF Database Cleaner — removes junk entries to improve lookup quality.
Usage:
    python clean_off_db.py                    # dry run (shows what would be deleted)
    python clean_off_db.py --apply            # actually delete
    python clean_off_db.py --apply --backup   # backup first, then delete
"""

import sqlite3
import re
import sys
import shutil
from pathlib import Path

DB_PATH = "off_products.db"
BACKUP_PATH = "off_products_backup.db"

# ═══════════════════════ Cleaning Rules ═══════════════════════

def is_junk_product_name(name: str) -> str | None:
    """
    Returns a reason string if the product name is junk, None if it's OK.
    """
    if not name or not name.strip():
        return "empty_name"

    n = name.strip()

    # Too short (1-2 chars like "A", "X", "12")
    if len(n) <= 2:
        return "too_short"

    # Starts with number/percentage: "0% Nature", "100 Calorie Packs", "3x Whiter"
    if re.match(r"^\d+%?\s", n):
        return "starts_with_number"

    # Pure numbers or codes: "12345", "ABC-123-456"
    if re.match(r"^[\d\-\.]+$", n):
        return "pure_number"

    # Contains non-Latin characters (Chinese, Arabic, Cyrillic, etc.)
    # Keep: Latin, digits, common punctuation
    # This filters non-English products that won't match US receipt text
    if re.search(r"[\u0400-\u04FF]", n):  # Cyrillic
        return "cyrillic"
    if re.search(r"[\u4E00-\u9FFF]", n):  # Chinese
        return "chinese"
    if re.search(r"[\u0600-\u06FF]", n):  # Arabic
        return "arabic"
    if re.search(r"[\u3040-\u30FF]", n):  # Japanese
        return "japanese"
    if re.search(r"[\uAC00-\uD7AF]", n):  # Korean
        return "korean"

    return None


def is_junk_quantity(qty: str, name: str) -> str | None:
    """
    Returns a reason string if the quantity is clearly wrong, None if OK.
    """
    if not qty or not qty.strip():
        return None  # empty is OK — just means no weight data

    q = qty.strip().lower()

    # Parse weight and check bounds
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g|lb|lbs|oz|ml|l|cl)\b", q, re.IGNORECASE)
    if not m:
        return None  # unparseable, leave it

    value = float(m.group(1))
    unit = m.group(2).lower()

    # Convert to grams for comparison
    conversions = {"g": 1, "kg": 1000, "oz": 28.35, "lb": 453.6, "lbs": 453.6,
                   "ml": 1, "l": 1000, "cl": 10}
    grams = value * conversions.get(unit, 1)

    # Too heavy: > 25kg (no single grocery item)
    if grams > 25000:
        return f"too_heavy_{value}{unit}={grams:.0f}g"

    # Suspiciously light for non-condiment: exactly 0
    if value == 0:
        return "zero_weight"

    return None


# ═══════════════════════ German/French Detection ═══════════════════════

# Common German/French words that indicate non-English product names
_NON_ENGLISH_MARKERS = {
    # German
    "und", "mit", "ohne", "für", "der", "die", "das", "ein", "eine",
    "nicht", "oder", "aber", "auch", "nach", "bei", "auf", "aus",
    "vom", "zum", "zur", "des", "dem", "den", "über", "unter",
    "milch", "brot", "käse", "wurst", "fleisch", "zucker", "mehl",
    "sahne", "butter", "joghurt", "schokolade", "kaffee", "tee",
    "wasser", "saft", "bier", "wein", "öl", "essig", "senf",
    "flächen", "reiniger", "bodenreinigung", "bürstenkopf",
    # French
    "avec", "sans", "pour", "dans", "sur", "sous", "entre",
    "lait", "pain", "fromage", "beurre", "crème", "farine",
    "sucre", "chocolat", "café", "thé", "eau", "jus", "vin",
    "huile", "vinaigre", "moutarde", "confiture",
    "pâte", "tartiner", "noisettes", "véritable",
    # Spanish
    "con", "sin", "para", "leche", "queso", "mantequilla",
    "azúcar", "harina", "aceite", "vinagre",
}


def has_non_english_markers(name: str) -> str | None:
    """Check if product name contains German/French/Spanish words."""
    words = set(re.sub(r"[^a-zA-ZäöüéèêàâîïôùûçñÄÖÜß]", " ", name.lower()).split())
    matches = words & _NON_ENGLISH_MARKERS
    if len(matches) >= 2:  # require 2+ markers to avoid false positives
        return f"non_english:{','.join(sorted(matches)[:3])}"
    return None


# ═══════════════════════ Main ═══════════════════════

def analyze_db(db_path: str) -> dict:
    """Analyze the database and return deletion candidates."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT rowid, code, product_name, quantity, categories FROM products")

    stats = {
        "total": 0,
        "junk_name": 0,
        "junk_quantity": 0,
        "non_english": 0,
        "delete_rowids": [],
        "reasons": {},
    }

    for rowid, code, name, qty, categories in cursor:
        stats["total"] += 1
        name = name or ""
        qty = qty or ""

        # Check product name
        reason = is_junk_product_name(name)
        if reason:
            stats["junk_name"] += 1
            stats["delete_rowids"].append(rowid)
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            continue

        # Check quantity
        reason = is_junk_quantity(qty, name)
        if reason:
            stats["junk_quantity"] += 1
            stats["delete_rowids"].append(rowid)
            category = reason.split("_")[0] + "_" + reason.split("_")[1] if "_" in reason else reason
            stats["reasons"][category] = stats["reasons"].get(category, 0) + 1
            continue

        # Check for non-English
        reason = has_non_english_markers(name)
        if reason:
            stats["non_english"] += 1
            stats["delete_rowids"].append(rowid)
            stats["reasons"]["non_english"] = stats["reasons"].get("non_english", 0) + 1
            continue

    conn.close()
    return stats


def show_samples(db_path: str, limit: int = 20):
    """Show sample entries that would be deleted."""
    conn = sqlite3.connect(db_path)

    print("\n--- Sample junk product names ---")
    for row in conn.execute(
        "SELECT product_name, quantity FROM products WHERE product_name LIKE '0%' OR product_name LIKE '1%' LIMIT ?",
        (limit,),
    ):
        print(f"  {row[0]:<50} qty={row[1]}")

    print("\n--- Sample non-English entries ---")
    for row in conn.execute(
        "SELECT product_name, quantity FROM products WHERE product_name LIKE '%für%' OR product_name LIKE '%avec%' OR product_name LIKE '%pâte%' LIMIT ?",
        (limit,),
    ):
        print(f"  {row[0]:<50} qty={row[1]}")

    print("\n--- Sample extreme weights ---")
    for row in conn.execute(
        "SELECT product_name, quantity FROM products WHERE quantity LIKE '%00 l%' OR quantity LIKE '%00 kg%' LIMIT ?",
        (limit,),
    ):
        print(f"  {row[0]:<50} qty={row[1]}")

    conn.close()


def apply_deletions(db_path: str, rowids: list[int]):
    """Delete junk rows from the database."""
    conn = sqlite3.connect(db_path)

    # Delete in batches of 500
    batch_size = 500
    deleted = 0
    for i in range(0, len(rowids), batch_size):
        batch = rowids[i:i + batch_size]
        placeholders = ",".join("?" * len(batch))
        conn.execute(f"DELETE FROM products WHERE rowid IN ({placeholders})", batch)
        deleted += len(batch)
        if deleted % 10000 == 0:
            print(f"  Deleted {deleted}/{len(rowids)}...")

    conn.commit()

    # Reclaim disk space
    print("  Running VACUUM to reclaim disk space...")
    conn.execute("VACUUM")
    conn.close()

    return deleted


def main():
    apply = "--apply" in sys.argv
    backup = "--backup" in sys.argv

    db = Path(DB_PATH)
    if not db.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    # Show file size
    size_mb = db.stat().st_size / (1024 * 1024)
    print(f"Database: {DB_PATH} ({size_mb:.1f} MB)")

    # Show samples first
    show_samples(DB_PATH)

    # Analyze
    print("\nAnalyzing database...")
    stats = analyze_db(DB_PATH)

    total_delete = len(stats["delete_rowids"])
    pct = total_delete / stats["total"] * 100 if stats["total"] > 0 else 0

    print(f"\n{'='*50}")
    print(f"Total records:      {stats['total']:>10,}")
    print(f"Junk names:         {stats['junk_name']:>10,}")
    print(f"Junk quantities:    {stats['junk_quantity']:>10,}")
    print(f"Non-English:        {stats['non_english']:>10,}")
    print(f"{'─'*50}")
    print(f"Total to delete:    {total_delete:>10,}  ({pct:.1f}%)")
    print(f"Records remaining:  {stats['total'] - total_delete:>10,}")

    print(f"\nBreakdown by reason:")
    for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason:<30} {count:>8,}")

    if not apply:
        print(f"\n*** DRY RUN — no changes made ***")
        print(f"Run with --apply to delete, or --apply --backup to backup first")
        return

    # Backup
    if backup:
        print(f"\nBacking up to {BACKUP_PATH}...")
        shutil.copy2(DB_PATH, BACKUP_PATH)
        print(f"  Backup saved ({size_mb:.1f} MB)")

    # Apply
    print(f"\nDeleting {total_delete:,} junk records...")
    deleted = apply_deletions(DB_PATH, stats["delete_rowids"])
    print(f"  Done! Deleted {deleted:,} records")

    # Show new size
    new_size_mb = db.stat().st_size / (1024 * 1024)
    print(f"\nDatabase size: {size_mb:.1f} MB → {new_size_mb:.1f} MB (saved {size_mb - new_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()