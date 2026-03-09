"""
Receipt Line Annotation Tool + ML Line Classifier

Two modes:
  1. ANNOTATE: Run OCR + rules on images, output pre-labeled CSV for human review
     python annotate_lines.py annotate data_Picture/*.png data_Picture/*.jpg

  2. TRAIN: Train line classifier from corrected CSV
     python annotate_lines.py train annotations.csv

  3. PREDICT: Use trained model to classify lines (replaces rule-based parsing)
     python annotate_lines.py predict <image>
"""

from __future__ import annotations

import sys
import re
import csv
import json
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

# ═══════════════════════ Line Types ═══════════════════════

LINE_TYPES = [
    "item",       # Product line (name + price)
    "discount",   # Cartwheel, coupon, savings, negative price
    "header",     # Store name, address, date, section titles
    "metadata",   # Subtotal, total, tax, payment info
    "qty",        # Quantity line (2EA @ 1.49/EA)
    "weight",     # Weight line (1.08 lb @ 1.99/lb)
    "noise",      # Barcode numbers, blank, unrecognizable
]

# ═══════════════════════ Feature Extraction ═══════════════════════

def extract_features(line: str, line_pos: float) -> dict:
    """
    Extract format-agnostic features from a receipt line.
    line_pos: position in receipt (0.0 = top, 1.0 = bottom)
    """
    text = line.strip()
    upper = text.upper()
    words = text.split()
    
    features = {}
    
    # ── Length features ──
    features["char_count"] = len(text)
    features["word_count"] = len(words)
    
    # ── Position in receipt ──
    features["line_pos"] = round(line_pos, 3)
    features["is_top_10pct"] = int(line_pos < 0.1)
    features["is_bottom_20pct"] = int(line_pos > 0.8)
    
    # ── Price patterns ──
    features["has_dollar_sign"] = int("$" in text)
    features["ends_with_price"] = int(bool(re.search(r"\$?\d+\.\d{2}\s*-?\s*$", text)))
    features["has_price"] = int(bool(re.search(r"\$?\d+\.\d{2}", text)))
    features["price_count"] = len(re.findall(r"\$?\d+\.\d{2}", text))
    features["has_negative_price"] = int(bool(re.search(r"\d+\.\d{2}\s*-", text)))
    
    # ── Number patterns ──
    features["starts_with_digits"] = int(bool(re.match(r"^\d", text)))
    features["starts_with_long_number"] = int(bool(re.match(r"^\d{6,}", text)))  # UPC
    features["digit_ratio"] = round(sum(c.isdigit() for c in text) / max(len(text), 1), 3)
    features["is_all_digits_symbols"] = int(bool(re.match(r"^[\d\s#*=.$]+$", text)))
    
    # ── Case patterns ──
    features["upper_ratio"] = round(sum(c.isupper() for c in text) / max(sum(c.isalpha() for c in text), 1), 3)
    features["has_mixed_case"] = int(any(c.islower() for c in text) and any(c.isupper() for c in text))
    
    # ── Keyword features ──
    # Item indicators
    features["has_weight_unit"] = int(bool(re.search(r"\b(oz|lb|lbs|kg|g)\b", text, re.IGNORECASE)))
    features["has_each"] = int("EACH" in upper or " EA " in upper or upper.endswith(" EA"))
    features["has_organic"] = int("ORG" in upper)
    
    # Discount indicators
    features["has_off"] = int(bool(re.search(r"\boff\b", text, re.IGNORECASE)))
    features["has_saved"] = int("SAVED" in upper)
    features["has_coupon"] = int("COUPON" in upper or "CPN" in upper or "CARTWHEEL" in upper)
    features["has_discount"] = int("DISCOUNT" in upper or "SAVINGS" in upper)
    
    # Metadata indicators
    features["has_subtotal"] = int("SUBTOTAL" in upper)
    features["has_total"] = int("TOTAL" in upper)
    features["has_tax"] = int("TAX" in upper)
    features["has_payment"] = int(any(k in upper for k in ["VISA", "CASH", "DEBIT", "CREDIT", "CARD"]))
    
    # Header indicators
    features["has_store_name"] = int(any(k in upper for k in [
        "WHOLE FOODS", "TRADER JOE", "WALMART", "COSTCO", "TARGET",
        "KROGER", "SAFEWAY",
    ]))
    features["has_address_pattern"] = int(bool(re.search(r"\d+\s+\w+\s+(St|Rd|Ave|Blvd|Dr)\b", text, re.IGNORECASE)))
    features["has_phone"] = int(bool(re.search(r"\(\d{3}\)\s*\d{3}", text)))
    features["has_date"] = int(bool(re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)))
    features["has_time"] = int(bool(re.search(r"\d{1,2}:\d{2}\s*(AM|PM)", text, re.IGNORECASE)))
    
    # Qty indicators
    features["has_at_sign"] = int("@" in text)
    features["has_qty_pattern"] = int(bool(re.search(r"\d+\s*(EA|DZ)\b", text, re.IGNORECASE)))
    
    # Weight indicators  
    features["has_weight_price"] = int(bool(re.search(r"\d+\.\d+\s*(lb|oz|kg)\s*@", text, re.IGNORECASE)))
    features["has_per_unit"] = int(bool(re.search(r"/\s*(lb|oz|kg|ea)\b", text, re.IGNORECASE)))
    features["has_tare"] = int("TARE" in upper)
    
    # Tax code indicators (Target: FC, T at end)
    features["has_tax_code"] = int(bool(re.search(r"\b(FC|T)\s+\$?\d+\.\d{2}", text)))
    
    return features


def features_to_array(features: dict, feature_names: list[str]) -> list[float]:
    """Convert feature dict to ordered array for model input."""
    return [features.get(name, 0) for name in feature_names]


# ═══════════════════════ Rule-Based Pre-Labeling ═══════════════════════

def rule_label(line: str, line_pos: float) -> str:
    """Assign a label using current rules (for pre-annotation)."""
    text = line.strip()
    upper = text.upper()
    
    if not text or len(text) <= 1:
        return "noise"
    
    # Header patterns
    if line_pos < 0.15:
        if any(k in upper for k in ["WHOLE FOODS", "TRADER JOE", "TARGET", "WALMART", "COSTCO"]):
            return "header"
        if re.search(r"\d+\s+\w+\s+(St|Rd|Ave|Blvd)\b", text, re.IGNORECASE):
            return "header"
        if re.search(r"\(\d{3}\)", text):
            return "header"
        if re.search(r"STORE\s*#", upper):
            return "header"
        if re.search(r"OPEN\s+\d", upper):
            return "header"
        if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text):
            return "header"
    
    # Section headers (Target)
    if upper in {"CLEANING SUPPLIES", "GROCERY", "HEALTH-BEAUTY-COSMETICS", "HOME"}:
        return "header"
    if "EXPECT MORE" in upper or "PAY LESS" in upper or "EXPIRES" in upper:
        return "header"
    
    # Metadata
    if "SUBTOTAL" in upper:
        return "metadata"
    if "TAX" in upper:
        return "metadata"
    if "TOTAL" in upper:
        return "metadata"
    if "BAL" in upper and re.search(r"\d+\.\d{2}", text):
        return "metadata"
    if any(k in upper for k in ["VISA", "CASH", "DEBIT", "CREDIT", "APPROVED",
                                  "AUTH CODE", "CARD #", "TRANS ID", "REDCARD"]):
        return "metadata"
    
    # Discount
    if re.search(r"\boff\b", text, re.IGNORECASE):
        return "discount"
    if re.search(r"\d+\.\d{2}\s*-\s*$", text):
        return "discount"
    if re.search(r"\$\d+\.\d{2}\s*-", text):
        return "discount"
    if re.match(r"^Cartwheel\b", text, re.IGNORECASE):
        return "discount"
    if re.match(r"^Saved\b", text, re.IGNORECASE):
        return "discount"
    if upper.startswith("*") and not upper.startswith("*WT"):
        return "discount"
    
    # Weight line
    if re.search(r"\d+\.\d+\s*(lb|oz|kg)\s*(@|.*?/\s*(lb|oz|kg))", text, re.IGNORECASE):
        return "weight"
    if "TARE" in upper:
        return "weight"
    
    # Quantity line
    if "@" in text and re.search(r"\d+\s*(EA|DZ)\b", text, re.IGNORECASE):
        return "qty"
    if re.match(r"^\d+\s*@\s*\$?\d+\.\d{2}", text):
        return "qty"
    
    # Noise
    if re.match(r"^[\d\s#*=.$]+$", text):
        return "noise"
    if re.match(r"^(ITEM|TARE)\b", upper):
        return "noise"
    if len(text) <= 2:
        return "noise"
    
    # Item (has price at end)
    if re.search(r"\$?\d+\.\d{2}\s*-?\s*[B]?\s*$", text):
        return "item"
    
    # Item without price (name only, price on next line)
    if len(text) > 3 and any(c.isalpha() for c in text):
        return "item"
    
    return "noise"


# ═══════════════════════ Annotate Mode ═══════════════════════

def annotate_images(image_paths: list[str], output_csv: str = "annotations/annotations.csv"):
    """Run OCR on all images and output pre-labeled CSV."""
    from run_receipt import OCREngine, rebuild_text
    
    ocr = OCREngine()
    rows = []
    
    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists():
            path = Path("data_Picture") / img_path
        if not path.exists():
            print(f"  SKIP: {img_path} not found")
            continue
        
        print(f"Processing: {path}")
        fragments = ocr.run(str(path))
        raw_text = rebuild_text(fragments)
        lines = raw_text.split("\n")
        total_lines = len(lines)
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            line_pos = i / max(total_lines - 1, 1)
            label = rule_label(line, line_pos)
            rows.append({
                "image": path.name,
                "line_num": i,
                "line_pos": round(line_pos, 3),
                "text": line,
                "auto_label": label,
                "corrected_label": "",  # Human fills this if auto_label is wrong
            })
    
    # Save CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "line_num", "line_pos", "text", "auto_label", "corrected_label"
        ])
        writer.writeheader()
        writer.writerows(rows)
    
    # Stats
    label_counts = Counter(r["auto_label"] for r in rows)
    print(f"\nSaved {len(rows)} lines to {output_csv}")
    print(f"Label distribution: {dict(label_counts)}")
    print(f"\nNext steps:")
    print(f"  1. Open {output_csv} in Excel")
    print(f"  2. Review 'auto_label' column")
    print(f"  3. If wrong, put correct label in 'corrected_label'")
    print(f"  4. Then run: python annotate_lines.py train {output_csv}")


# ═══════════════════════ Text Preprocessing ═══════════════════════

def _clean_text_for_tfidf(text: str) -> str:
    """Normalize text for TF-IDF: lowercase, strip numbers/symbols."""
    t = text.lower().strip()
    # Replace prices/numbers with token
    t = re.sub(r"\$?\d+\.\d{2}", " _PRICE_ ", t)
    t = re.sub(r"\d{6,}", " _UPC_ ", t)
    t = re.sub(r"\d+", " _NUM_ ", t)
    # Remove excess punctuation but keep @#%
    t = re.sub(r"[^\w\s@#%_]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ═══════════════════════ Train Mode ═══════════════════════

def train_classifier(csv_path: str, model_path: str = "models/line_classifier.pkl"):
    """Train line classifier from annotated CSV using handcrafted + TF-IDF features."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack, csr_matrix
    
    # Label mapping: merge to 5 classes
    LABEL_MAP = {
        "metadata": "noise",
        "discount": "noise",
    }
    
    # Load data
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("corrected_label", "").strip()
            if not label:
                label = row["auto_label"]
            # Apply mapping
            if label in LABEL_MAP:
                label = LABEL_MAP[label]
                if label is None:
                    continue  # discard
            rows.append({
                "text": row["text"],
                "line_pos": float(row["line_pos"]),
                "label": label,
            })
    
    print(f"Loaded {len(rows)} annotated lines")
    
    # ── Handcrafted features ──
    feature_dicts = [extract_features(r["text"], r["line_pos"]) for r in rows]
    feature_names = sorted(feature_dicts[0].keys())
    X_hand = np.array([features_to_array(fd, feature_names) for fd in feature_dicts])
    
    # ── TF-IDF text features ──
    texts = [_clean_text_for_tfidf(r["text"]) for r in rows]
    tfidf = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=2,                  # appear in at least 2 lines
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(texts)
    
    # ── Combine features ──
    X = hstack([csr_matrix(X_hand), X_tfidf]).toarray()
    
    tfidf_names = [f"tfidf_{n}" for n in tfidf.get_feature_names_out()]
    all_feature_names = feature_names + tfidf_names
    
    le = LabelEncoder()
    y = le.fit_transform([r["label"] for r in rows])
    
    n_hand = len(feature_names)
    n_tfidf = len(tfidf_names)
    print(f"Features: {n_hand} handcrafted + {n_tfidf} TF-IDF = {n_hand + n_tfidf} total")
    print(f"Classes: {list(le.classes_)}")
    print(f"Distribution: {dict(Counter(r['label'] for r in rows))}")
    
    # Train
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=3,
        random_state=42,
    )
    
    # Cross-validation
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"\nCross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    
    # Train on full data
    clf.fit(X, y)
    
    # Feature importance (top 15, mix of hand + tfidf)
    importances = sorted(zip(all_feature_names, clf.feature_importances_), key=lambda x: -x[1])
    print(f"\nTop 15 features:")
    for name, imp in importances[:15]:
        print(f"  {name:<30} {imp:.4f}")
    
    # Save model (including tfidf vectorizer)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "classifier": clf,
        "label_encoder": le,
        "feature_names": feature_names,
        "tfidf_vectorizer": tfidf,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"To use: python annotate_lines.py predict <image>")


# ═══════════════════════ Predict Mode ═══════════════════════

def predict_lines(image_path: str, model_path: str = "models/line_classifier.pkl"):
    """Use trained model to classify lines in a receipt image."""
    from run_receipt import OCREngine, rebuild_text
    
    # Load model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    clf = model_data["classifier"]
    le = model_data["label_encoder"]
    feature_names = model_data["feature_names"]
    
    # OCR
    ocr = OCREngine()
    path = Path(image_path)
    if not path.exists():
        path = Path("data_Picture") / image_path
    
    fragments = ocr.run(str(path))
    raw_text = rebuild_text(fragments)
    lines = raw_text.split("\n")
    total_lines = len(lines)
    
    print(f"\n===== ML Line Classification =====")
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        line_pos = i / max(total_lines - 1, 1)
        features = extract_features(line, line_pos)
        X_hand = np.array([features_to_array(features, feature_names)])
        
        # TF-IDF features if available
        tfidf = model_data.get("tfidf_vectorizer")
        if tfidf is not None:
            X_tfidf = tfidf.transform([_clean_text_for_tfidf(line)]).toarray()
            X = np.hstack([X_hand, X_tfidf])
        else:
            X = X_hand
        
        pred = le.inverse_transform(clf.predict(X))[0]
        proba = clf.predict_proba(X)[0]
        conf = max(proba)
        print(f"  [{pred:<10}] ({conf:.2f}) {line[:60]}")


# ═══════════════════════ Merge Mode ═══════════════════════

def merge_annotations(csv_paths: list[str], output_csv: str = "annotations/annotations_merged.csv"):
    """
    Merge multiple annotation CSVs into one.
    - Deduplicates by (image, line_num)
    - Later files override earlier ones (so corrected versions win)
    - Preserves corrected_label if present
    """
    merged: dict[tuple, dict] = {}  # (image, line_num) -> row
    order: list[tuple] = []

    for path in csv_paths:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (r["image"], r["line_num"])
                if key not in merged:
                    order.append(key)
                merged[key] = r
                count += 1
        print(f"  Loaded {count} lines from {path}")

    # Write merged
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "line_num", "line_pos", "text", "auto_label", "corrected_label"
        ])
        writer.writeheader()
        for key in order:
            writer.writerow(merged[key])

    # Stats
    rows = [merged[k] for k in order]
    final = Counter((r.get("corrected_label") or "").strip() or r["auto_label"] for r in rows)
    images = Counter(r["image"] for r in rows)
    print(f"\nMerged: {len(rows)} lines from {len(images)} images")
    print(f"Saved to: {output_csv}")
    print(f"\nLabel distribution:")
    for k, v in final.most_common():
        print(f"  {k:<12} {v}")
    print(f"\nNext: python annotate_lines.py train {output_csv}")


# ═══════════════════════ Entry Point ═══════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python annotate_lines.py annotate <image1> [image2 ...]")
        print("  python annotate_lines.py train <annotations.csv>")
        print("  python annotate_lines.py predict <image>")
        print("  python annotate_lines.py merge <csv1> <csv2> [csv3 ...]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "annotate":
        if len(sys.argv) < 3:
            print("Usage: python annotate_lines.py annotate <image1> [image2 ...]")
            sys.exit(1)
        annotate_images(sys.argv[2:])
    
    elif mode == "train":
        if len(sys.argv) < 3:
            print("Usage: python annotate_lines.py train annotations.csv")
            sys.exit(1)
        train_classifier(sys.argv[2])
    
    elif mode == "predict":
        if len(sys.argv) < 3:
            print("Usage: python annotate_lines.py predict <image>")
            sys.exit(1)
        predict_lines(sys.argv[2])
    
    elif mode == "merge":
        if len(sys.argv) < 4:
            print("Usage: python annotate_lines.py merge <csv1> <csv2> [csv3 ...]")
            sys.exit(1)
        merge_annotations(sys.argv[2:])
    
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
