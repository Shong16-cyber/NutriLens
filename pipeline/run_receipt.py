"""
Receipt OCR Parser — PaddleOCR + abbreviation expansion + OFF weight & nutrition lookup
Usage:
    python run_receipt.py <image>           # single image
    python run_receipt.py <img1> <img2> ... # batch (model loaded once)
"""

from __future__ import annotations

import sys
import re
import json
import csv
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field, asdict

from paddleocr import PaddleOCR

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

# Resolve project root from this script's location (pipeline/)
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent


# ═══════════════════════ ML Classifier ═══════════════════════

def _load_ml_classifier(model_path: str = str(PROJECT_ROOT / "models" / "line_classifier.pkl")):
    """Load the trained ML line classifier. Returns None if model not found."""
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        import pickle
        import numpy as np
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ML] Warning: could not load model from {model_path}: {e}")
        return None


def _classify_lines_ml(lines: list[str], model_data: dict) -> list[str]:
    """
    Classify a list of receipt lines using the trained ML model.
    Returns a list of labels: 'item' | 'noise' | 'header' | 'date' | 'qty' | 'weight'
    """
    import numpy as np

    clf = model_data["classifier"]
    le = model_data["label_encoder"]
    tfidf = model_data.get("tfidf") or model_data.get("tfidf_vectorizer")

    # Old-style model (has feature_names key)
    if "feature_names" in model_data:
        from annotate_lines import extract_features, features_to_array, _clean_text_for_tfidf
        feature_names = model_data["feature_names"]
        total = len(lines)
        labels = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                labels.append("noise")
                continue
            line_pos = i / max(total - 1, 1)
            features = extract_features(line, line_pos)
            X_hand = np.array([features_to_array(features, feature_names)])
            if tfidf is not None:
                X_tfidf = tfidf.transform([_clean_text_for_tfidf(line)]).toarray()
                X = np.hstack([X_hand, X_tfidf])
            else:
                X = X_hand
            pred = le.inverse_transform(clf.predict(X))[0]
            labels.append(pred)
        return labels

    # New-style model: inline feature extraction
    labels = []
    for line in lines:
        line = line.strip()
        if not line:
            labels.append("noise")
            continue
        X_hand = np.array([_extract_ml_features(line)])
        if tfidf is not None:
            X_tfidf = tfidf.transform([line]).toarray()
            X = np.hstack([X_hand, X_tfidf])
        else:
            X = X_hand
        pred = le.inverse_transform(clf.predict(X))[0]
        labels.append(pred)
    return labels


def _extract_ml_features(text):
    """Extract handcrafted features matching retrain_classifier.py."""
    t = text.strip()
    upper = t.upper()
    words = t.split()
    n_words = len(words)
    n_chars = len(t)
    n_alpha = sum(1 for c in t if c.isalpha())
    n_digit = sum(1 for c in t if c.isdigit())
    n_upper = sum(1 for c in t if c.isupper())
    n_special = sum(1 for c in t if not c.isalnum() and not c.isspace())
    alpha_ratio = n_alpha / max(n_chars, 1)
    digit_ratio = n_digit / max(n_chars, 1)
    upper_ratio = n_upper / max(n_alpha, 1)
    special_ratio = n_special / max(n_chars, 1)
    has_price = 1 if re.search(r"\d+\.\d{2}", t) else 0
    n_prices = len(re.findall(r"\d+\.\d{2}", t))
    has_dollar = 1 if "$" in t else 0
    ends_with_price = 1 if re.search(r"\d+\.\d{2}\s*$", t) else 0
    has_tax_code = 1 if re.search(r"\s[FTNXOAB]\s*$", t) else 0
    starts_with_upc = 1 if re.match(r"^\d{8,15}", t) else 0
    has_date = 1 if re.search(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", t) else 0
    starts_with_date = 1 if re.match(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", t) else 0
    has_time = 1 if re.search(r"\d{1,2}:\d{2}", t) else 0
    has_month_name = 1 if re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", t, re.I) else 0
    has_phone = 1 if re.search(r"\d{3}[\-.)+]\d{3}[\-.)+]\d{4}", t) else 0
    has_address = 1 if re.search(r"\b(?:Ave|St|Blvd|Rd|Dr|Ln|Ct|Way|Hwy)\b", t, re.I) else 0
    has_zip = 1 if re.search(r"\b\d{5}(?:\-\d{4})?\b", t) else 0
    has_state = 1 if re.search(r"\b(?:WA|CA|OR|TX|NY|FL|IL|PA|OH|GA)\b", t) else 0
    has_total = 1 if "TOTAL" in upper else 0
    has_subtotal = 1 if "SUBTOTAL" in upper else 0
    has_tax = 1 if re.search(r"\bTAX\b", upper) else 0
    has_change = 1 if "CHANGE" in upper else 0
    has_card = 1 if any(w in upper for w in ["VISA", "MASTERCARD", "DEBIT", "CREDIT", "CARD"]) else 0
    has_aid = 1 if re.search(r"AID\s*:", upper) else 0
    has_auth = 1 if "AUTH" in upper else 0
    has_return = 1 if "RETURN" in upper else 0
    has_save = 1 if "SAVE" in upper or "SAVING" in upper else 0
    has_ea = 1 if re.search(r"\b\d+\s*(?:EA|EACH)\b", upper) else 0
    has_qty_prefix = 1 if re.match(r"^\s*\d+\s*(?:EA|EACH|PC|PCS|DZ|@)", upper) else 0
    has_weight = 1 if re.search(r"\d+\.?\d*\s*(?:lb|lbs|oz|kg|g)\b", t, re.I) else 0
    has_at_per = 1 if re.search(r"@|per\s*(?:lb|oz|kg)", t, re.I) else 0
    return [
        n_words, n_chars, alpha_ratio, digit_ratio, upper_ratio, special_ratio,
        has_price, n_prices, has_dollar, ends_with_price, has_tax_code, starts_with_upc,
        has_date, starts_with_date, has_time, has_month_name,
        has_phone, has_address, has_zip, has_state,
        has_total, has_subtotal, has_tax, has_change,
        has_card, has_aid, has_auth, has_return, has_save,
        has_ea, has_qty_prefix, has_weight, has_at_per,
    ]


# ═══════════════════════ Configuration ═══════════════════════

CONFIG = {
    "off_db_path": str(PROJECT_ROOT / "data" / "db" / "off_products.db"),
    "image_dir": str(PROJECT_ROOT / "data_Picture"),
    "usda_db_path": str(PROJECT_ROOT / "data" / "db" / "usda_sr.db"),
}

NOISE_KEYWORDS = [
    "COUPON", "DISCOUNT", "SAVINGS", "PROMO", "REWARD", "MEMBER",
    "SUBTOTAL", "TOTAL", "TAX", "CHANGE", "CASH", "CREDIT", "DEBIT",
    "BALANCE", "AMOUNT", "DUE", "ROUNDING", "TIP", "GRATUITY",
    "TARE", "ITEM", "BAL", "BAG FEE", "FEE",
    "INVOICE", "RECEIPT", "SIGNATURE", "AUTHORIZED",
]

ABBREVIATIONS = {
    # Store brands
    "WFM": "Whole Foods Market", "TJ": "Trader Joe's",
    "GV": "Great Value", "KS": "Kirkland Signature", "365": "365",
    # Organic / quality
    "ORG": "Organic", "OG": "Organic", "ORGN": "Organic",
    "CF": "Cage Free", "CGFR": "Cage Free",
    "NHP": "NHP",
    # Preparation
    "FZN": "Frozen", "FRZ": "Frozen", "FROZ": "Frozen",
    "BRDD": "Breaded", "SLCD": "Sliced",
    "SHRDD": "Shredded", "SHRD": "Shredded",
    "RSTD": "Roasted", "RSTED": "Roasted", "RO": "Roasted",
    "GRLD": "Grilled", "SMK": "Smoked", "SMKD": "Smoked",
    "CHPD": "Chopped", "UNSWT": "Unsweetened",
    "RDC": "Reduced", "RDCD": "Reduced",
    # Protein
    "CHKN": "Chicken", "CHCKN": "Chicken",
    "BRST": "Breast", "BNLS": "Boneless", "SKNLS": "Skinless",
    "TNDRL": "Tenderloin", "FLLT": "Fillet", "FLLTS": "Fillets",
    "BF": "Beef", "PRK": "Pork", "TRKY": "Turkey",
    "SLMN": "Salmon", "SHRMP": "Shrimp",
    # Vegetables
    "GRN": "Green", "BNS": "Beans", "BLK": "Black",
    "VEG": "Vegetable", "VEGG": "Veggie",
    "SWT": "Sweet", "PTT": "Potato", "POTA": "Potato", "PTO": "Potato",
    "TMT": "Tomato", "TOMA": "Tomato",
    "MUSH": "Mushroom", "MSHRM": "Mushroom",
    "BRCL": "Broccoli", "BROC": "Broccoli",
    "SPNCH": "Spinach", "CUKE": "Cucumber", "CUCS": "Cucumbers",
    "LTT": "Lettuce", "CRRT": "Carrot",
    "ONI": "Onion", "PEP": "Pepper", "PPRS": "Peppers",
    "GRLC": "Garlic",
    # Fruit
    "STRW": "Strawberry", "STRWB": "Strawberries",
    "BLUBR": "Blueberry", "BLUEB": "Blueberries",
    "BANA": "Banana", "APPL": "Apple",
    "MNG": "Mango", "MNGO": "Mango",
    "AVCD": "Avocado", "LMN": "Lemon", "ORNG": "Orange",
    # Dairy
    "MLK": "Milk", "CTTG": "Cottage", "CTGE": "Cottage",
    "CHDR": "Cheddar", "MOZZ": "Mozzarella", "PARM": "Parmesan",
    "YGT": "Yogurt", "YGRT": "Yogurt",
    "CRM": "Cream", "BTR": "Butter",
    "CHEE": "Cheese", "CHS": "Cheese", "CHSE": "Cheese",
    "WHT": "White", "WHIT": "White",
    "WHL": "Whole", "HLF": "Half", "GAL": "Gallon", "QRT": "Quart",
    "HG": "Half Gallon",
    "LF": "Low Fat", "LWFT": "Low Fat", "FF": "Fat Free",
    "OGLF": "Organic Low Fat", "OGLWF": "Organic Low Fat",
    # Grains / bakery
    "TORTL": "Tortilla", "TORT": "Tortellini",
    "PL": "Plant", "PLT": "Plant", "PLNT": "Plant",
    "RAV": "Ravioli", "ARUG": "Arugula",
    "ITAL": "Italian", "ROOLS": "Rolls", "PAR": "Par",
    # Sauces / condiments
    "SW": "Southwest", "SLD": "Salad",
    "FLVR": "Flavor", "VAN": "Vanilla", "VNLA": "Vanilla",
    "CHOC": "Chocolate", "DK": "Dark",
    "PB": "Peanut Butter", "HNY": "Honey", "CINN": "Cinnamon",
    "SLT": "Salt", "LS": "Low Sodium",
    "EVOO": "Extra Virgin Olive Oil", "OO": "Olive Oil",
    # Packaging / sizing
    "PCK": "Pack", "PK": "Pack", "CTN": "Carton",
    "BTL": "Bottle", "BG": "Bag", "BX": "Box",
    "EA": "Each", "DZ": "Dozen",
    "LG": "Large", "SM": "Small", "MED": "Medium", "REG": "Regular",
    "CV": "Conventional",
    "FRT": "Fruit", "FRSH": "Fresh",
    "BBY": "Baby", "BLLA": "Bella",
    "OV": "Oven", "STRAW": "Straw", "BERIES": "Berries",
    "REC": "Recommended",
    # Target brand abbreviations
    "MP": "Market Pantry", "GG": "Good & Gather",
    "FRTSN": "Fruit Snacks", "ENER": "Energy",
    "MASTERPC": "Masterpiece", "SDNLY": "Suddenly",
    "CUISIN": "Cuisine",
    "AQUAF": "Aquafina",
    "SUBB": "Sub Sandwich",
    "REDDI": "Reddi", "WIP": "Wip",
    "PANT": "Pantry",
    "NATVAL": "Nature Valley",
    # Costco / Kirkland
    "ATAULFO": "Ataulfo Mango",
    # Sam's Club (MM = Member's Mark)
    "MM": "Member's Mark",
    "MEDL": "Medley",
    "SWAI": "Swai",
    # Kroger
    "KRO": "Kroger",
    "HLAM": "Hoffman's American", "MLLR": "Miller",
    "ALPRS": "Alps", "BRDV": "Birds Eye Voila",
    "TRSME": "TRESemmé", "DSCH": "Deschutes",
    "KOSH": "Kosher", "TMTE": "Tomato",
    "SNDWCH": "Sandwich", "PLNTR": "Planter",
    "MSTRPC": "Masterpiece", "SCE": "Sauce",
    "MDJL": "Medjool", "PTD": "Pitted",
    "BNTY": "Bounty",
    # Walmart
    "GRANU": "Granulated",
    "BANAN": "Banana",
    # H-E-B
    "HEB": "H-E-B",
    # Safeway
    "CHP": "Chip",
    "THNS": "Thins",
}


# ═══════════════════════ Unit Conversion ═══════════════════════

# Conversion factors to grams
_TO_GRAMS = {
    "g": 1.0,
    "kg": 1000.0,
    "oz": 28.3495,
    "lb": 453.592,
    "ml": 1.0,      # approximate: 1ml ≈ 1g for water-based products
    "l": 1000.0,
}


def _convert_to_grams(value: float, unit: str) -> float:
    """Convert any weight/volume to grams."""
    return value * _TO_GRAMS.get(unit, 1.0)


# ═══════════════════════ Data Structures ═══════════════════════

@dataclass
class Weight:
    value: float
    unit: str

    @property
    def grams(self) -> float:
        """Return weight converted to grams."""
        return _convert_to_grams(self.value, self.unit)


@dataclass
class Nutriments:
    """Nutritional info per 100g from OFF database."""
    energy_kcal: float | None = None
    fat: float | None = None
    saturated_fat: float | None = None
    carbohydrates: float | None = None
    sugars: float | None = None
    fiber: float | None = None
    proteins: float | None = None
    salt: float | None = None
    sodium: float | None = None

    def scale_to_weight(self, weight_grams: float) -> "Nutriments":
        """Return a new Nutriments scaled from per-100g to actual weight."""
        factor = weight_grams / 100.0
        def _scale(v):
            return round(v * factor, 2) if v is not None else None
        return Nutriments(
            energy_kcal=_scale(self.energy_kcal),
            fat=_scale(self.fat),
            saturated_fat=_scale(self.saturated_fat),
            carbohydrates=_scale(self.carbohydrates),
            sugars=_scale(self.sugars),
            fiber=_scale(self.fiber),
            proteins=_scale(self.proteins),
            salt=_scale(self.salt),
            sodium=_scale(self.sodium),
        )

    def has_data(self) -> bool:
        """Return True if at least one nutrient value is not None."""
        return any(v is not None for v in [
            self.energy_kcal, self.fat, self.saturated_fat,
            self.carbohydrates, self.sugars, self.fiber,
            self.proteins, self.salt, self.sodium,
        ])

    def is_all_zero(self) -> bool:
        """Return True if all major nutrients are 0 (likely bad OFF data)."""
        return (self.energy_kcal == 0 and self.proteins == 0
                and self.fat == 0 and self.carbohydrates == 0)


@dataclass
class Item:
    name: str
    resolved_name: str = ""
    qty: int | None = None
    weight: Weight | None = None
    weight_grams: float | None = None       # total weight in grams (weight × qty)
    weight_source: str = ""
    line_price: float | None = None
    nutriments_per100g: Nutriments | None = None   # per 100g from OFF
    nutriments_actual: Nutriments | None = None     # scaled to actual total weight

@dataclass
class Receipt:
    store_name: str = ""
    store_addr: str = ""
    telephone: str = ""
    date: str = ""
    time: str = ""
    subtotal: str = ""
    tax: str = ""
    total: str = ""
    items: list[Item] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def save_json(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def save_csv(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "name", "resolved_name", "qty",
                "weight_value", "weight_unit", "weight_grams", "weight_source",
                "line_price",
                # per 100g
                "energy_kcal_100g", "fat_100g", "saturated_fat_100g",
                "carbohydrates_100g", "sugars_100g", "fiber_100g",
                "proteins_100g", "salt_100g", "sodium_100g",
                # actual (scaled to total weight)
                "energy_kcal_actual", "fat_actual", "saturated_fat_actual",
                "carbohydrates_actual", "sugars_actual", "fiber_actual",
                "proteins_actual", "salt_actual", "sodium_actual",
            ])
            for item in self.items:
                n100 = item.nutriments_per100g or Nutriments()
                nact = item.nutriments_actual or Nutriments()
                writer.writerow([
                    item.name,
                    item.resolved_name,
                    item.qty or "",
                    item.weight.value if item.weight else "",
                    item.weight.unit if item.weight else "",
                    round(item.weight_grams, 1) if item.weight_grams else "",
                    item.weight_source,
                    item.line_price or "",
                    # per 100g
                    n100.energy_kcal if n100.energy_kcal is not None else "",
                    n100.fat if n100.fat is not None else "",
                    n100.saturated_fat if n100.saturated_fat is not None else "",
                    n100.carbohydrates if n100.carbohydrates is not None else "",
                    n100.sugars if n100.sugars is not None else "",
                    n100.fiber if n100.fiber is not None else "",
                    n100.proteins if n100.proteins is not None else "",
                    n100.salt if n100.salt is not None else "",
                    n100.sodium if n100.sodium is not None else "",
                    # actual
                    nact.energy_kcal if nact.energy_kcal is not None else "",
                    nact.fat if nact.fat is not None else "",
                    nact.saturated_fat if nact.saturated_fat is not None else "",
                    nact.carbohydrates if nact.carbohydrates is not None else "",
                    nact.sugars if nact.sugars is not None else "",
                    nact.fiber if nact.fiber is not None else "",
                    nact.proteins if nact.proteins is not None else "",
                    nact.salt if nact.salt is not None else "",
                    nact.sodium if nact.sodium is not None else "",
                ])


# ═══════════════════════ Utility ═══════════════════════

def extract_weight(name: str) -> Weight | None:
    """Extract weight/volume from product name (e.g. '16oz', '2 LB')."""
    if not name:
        return None
    s = re.sub(r"(\d+(?:\.\d+)?)\s*o\b", r"\1 oz", name.strip(), flags=re.IGNORECASE)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g|lb|lbs|oz|ml|l)\b", s, flags=re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    if value == 0:
        return None
    unit = m.group(2).lower()
    if unit == "lbs":
        unit = "lb"
    return Weight(value=value, unit=unit)


def is_noise(name: str) -> bool:
    """Check if a name is receipt metadata noise."""
    if not name or len(name.strip()) <= 1:
        return True
    n = name.strip().upper()
    if n.startswith("*"):
        return True
    for k in NOISE_KEYWORDS:
        if len(k) <= 4:
            if re.search(r"\b" + re.escape(k) + r"\b", n) and len(n.split()) <= 3:
                return True
        else:
            if k in n:
                return True
    return False


def to_title_case(text: str) -> str:
    """
    Normalize to title case, preserving:
    - ALL-CAPS brand names (NHP, FUJI)
    - Measurements (16oz, 13.6 OZ, 2 LB, 4PK)
    - Numbers
    """
    words = text.split()
    result = []
    keep_upper = {"NHP", "OZ", "LB", "KG", "ML", "EA", "DZ", "FUJI"}
    for w in words:
        wu = w.upper()
        if re.match(r"^\d", w):
            result.append(w)
        elif wu in keep_upper:
            result.append(wu)
        elif w == "&":
            result.append("&")
        else:
            result.append(w.capitalize())
    return " ".join(result)


# ═══════════════════════ Target Format Cleaning ═══════════════════════

_UNIVERSAL_TAX_CODES = re.compile(
    r"\s+(?:FC|FT|NF|TF\s*P|TP|FA|FB|FW|PC|NP|KF|[FTNXOAB])\s*$",
    re.IGNORECASE,
)

_SECTION_HEADERS = {
    "GROCERY", "CLEANING SUPPLIES", "HEALTH-BEAUTY-COSMETICS",
    "HEALTH AND BEAUTY", "HOME", "ELECTRONICS", "APPAREL",
    "BABY", "PETS", "TOYS", "SEASONAL", "MARKET",
    "SPECIAL PROMOTION",
    "REFRIG/FROZEN", "PRODUCE", "MISCELLANEOUS",
    "DAIRY", "BAKERY", "DELI", "MEAT", "SEAFOOD", "FROZEN",
    "BEVERAGES", "SNACKS", "PHARMACY",
}

_NONFOOD_SECTIONS = {
    "CLEANING SUPPLIES", "HEALTH-BEAUTY-COSMETICS", "HEALTH AND BEAUTY",
    "HOME", "ELECTRONICS", "APPAREL", "BABY", "PETS", "TOYS", "SEASONAL",
    "MISCELLANEOUS", "PHARMACY",
}


def _strip_upc_and_tax(name: str) -> str:
    text = name.strip()
    text = re.sub(r"^\d{4,15}(?:KF)?\s+", "", text)
    text = re.sub(r"^\d{4,15}(?:-\d+)?\s+", "", text)
    text = re.sub(r"^E\s+", "", text)
    text = re.sub(r"^\d{4,15}\s+", "", text)
    text = re.sub(r"^\^\s*", "", text)
    text = re.sub(r"^\d\s+(?=[A-Z])", "", text)
    text = _UNIVERSAL_TAX_CODES.sub("", text)
    text = re.sub(r"\s*[↓]\s*$", "", text)
    text = re.sub(r"\s+t\s+F\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\d{8,15}(?:KF)?\s*$", "", text)
    # Also strip trailing "NF $", "NF$", "F $" patterns (Google Vision)
    text = re.sub(r"\s+NF\s*\$?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+[FTNXOAB]\s*\$\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _is_section_header(line: str) -> bool:
    stripped = line.strip().upper()
    return stripped in _SECTION_HEADERS or any(
        stripped.startswith(h) for h in _SECTION_HEADERS
    )


_NOISE_PATTERNS = [
    re.compile(r"^Cartwheel\b", re.IGNORECASE),
    re.compile(r"^Target Circle", re.IGNORECASE),
    re.compile(r"^Buy\d+Get", re.IGNORECASE),
    re.compile(r"^Your REDcard\b", re.IGNORECASE),
    re.compile(r"^PACKAGE\s+(SUBTOTAL|TOTAL)\b", re.IGNORECASE),
    re.compile(r"^Regular Price\b", re.IGNORECASE),
    re.compile(r"^RegPrice\b", re.IGNORECASE),
    re.compile(r"CardSav\b", re.IGNORECASE),
    re.compile(r"^Refund Value\b", re.IGNORECASE),
    re.compile(r"^Saved\b.*\boff\b", re.IGNORECASE),
    re.compile(r"^You Saved\b", re.IGNORECASE),
    re.compile(r"^Promotion\b", re.IGNORECASE),
    re.compile(r"^Transaction Disc", re.IGNORECASE),
    re.compile(r"^Savings Summary\b", re.IGNORECASE),
    re.compile(r"^Order Total\b", re.IGNORECASE),
    re.compile(r"^Grand Total\b", re.IGNORECASE),
    re.compile(r"^MFR COUPON\b", re.IGNORECASE),
    re.compile(r"^Mfr\.?\s*eCoupon\b", re.IGNORECASE),
    re.compile(r"^Paper Mfr Coupon\b", re.IGNORECASE),
    re.compile(r"^MC\s+SCANNED COUPON\b", re.IGNORECASE),
    re.compile(r"^DIGITAL COUPON\b", re.IGNORECASE),
    re.compile(r"^CATEGORY COUPON\b", re.IGNORECASE),
    re.compile(r"^Bottle Deposit\b", re.IGNORECASE),
    re.compile(r"^SC\s+KROGER\s+SAVINGS\b", re.IGNORECASE),
    re.compile(r"^KROGER PLUS\b", re.IGNORECASE),
    re.compile(r"^SC\s+\w+\s+SAVINGS\b", re.IGNORECASE),
    re.compile(r"^DC\s+\w", re.IGNORECASE),
    re.compile(r"^OUR BRAND SAVINGS\b", re.IGNORECASE),
    re.compile(r"^DIGITAL COUPONS\b", re.IGNORECASE),
    re.compile(r"^YOU SAVED\b", re.IGNORECASE),
    re.compile(r"^\*+\s*Sale Subtotal", re.IGNORECASE),
    re.compile(r"^\*+\s*Total Sale", re.IGNORECASE),
    re.compile(r"^ITEMS PURCHASED\b", re.IGNORECASE),
    re.compile(r"^EFT\s+DEBIT\b", re.IGNORECASE),
    re.compile(r"^PAY FROM\b", re.IGNORECASE),
    re.compile(r"^PRIMARY\b", re.IGNORECASE),
    re.compile(r"^CHANGE DUE\b", re.IGNORECASE),
    re.compile(r"^DEBIT TEND\b", re.IGNORECASE),
    re.compile(r"^CASH TEND\b", re.IGNORECASE),
    re.compile(r"^# ITEMS\b", re.IGNORECASE),
    re.compile(r"^# ITEMS SOLD\b", re.IGNORECASE),
    re.compile(r"^Low Prices You Can", re.IGNORECASE),
    re.compile(r"^OK Member\b", re.IGNORECASE),
    re.compile(r"^APPROVED\b", re.IGNORECASE),
    re.compile(r"^CLUB MANAGER\b", re.IGNORECASE),
    re.compile(r"^MASTERCARD\b", re.IGNORECASE),
    re.compile(r"^TOTAL NUMBER OF ITEMS\b", re.IGNORECASE),
    re.compile(r"^AMOUNT DUE\b", re.IGNORECASE),
    re.compile(r"^\+\+APPROVED", re.IGNORECASE),
    re.compile(r"^Credit Card\s+\$", re.IGNORECASE),
    re.compile(r"^PROMO\b", re.IGNORECASE),
    re.compile(r"^New Bal", re.IGNORECASE),
    re.compile(r"^\d{3}-\d{3}-\d{3}-\d{3}"),
    re.compile(r"^\*\d{4}\b"),
    re.compile(r"^x{3,}", re.IGNORECASE),
    re.compile(r"^AID\s+A\d+", re.IGNORECASE),
    re.compile(r"^AAC\s+\w+", re.IGNORECASE),
    re.compile(r"^TERMINAL\s+#", re.IGNORECASE),
    re.compile(r"^REF\s+#", re.IGNORECASE),
    re.compile(r"^NETWORK\s+ID", re.IGNORECASE),
    re.compile(r"^Auth\s+#", re.IGNORECASE),
    re.compile(r"^TVR\s+\d", re.IGNORECASE),
    re.compile(r"^IAD\s+\d", re.IGNORECASE),
    re.compile(r"^TSI\s+\w", re.IGNORECASE),
    re.compile(r"^TC#\s+\d", re.IGNORECASE),
    re.compile(r"^Seq#", re.IGNORECASE),
    re.compile(r"^OP#", re.IGNORECASE),
    re.compile(r"^US DEBIT\b", re.IGNORECASE),
    re.compile(r"^VISA\b", re.IGNORECASE),
    re.compile(r"^SOLD\s+\d+", re.IGNORECASE),
    re.compile(r"^Give us feedback", re.IGNORECASE),
    re.compile(r"^survey\.", re.IGNORECASE),
    re.compile(r"^Thank you", re.IGNORECASE),
    re.compile(r"^Please Come\b", re.IGNORECASE),
    re.compile(r"^Your cashier\b", re.IGNORECASE),
    re.compile(r"^Show your\b", re.IGNORECASE),
    re.compile(r"^Date of Birth\b", re.IGNORECASE),
    re.compile(r"^Scan with\b", re.IGNORECASE),
    re.compile(r"^Together,?\s+we", re.IGNORECASE),
    re.compile(r"^Savings at Pub", re.IGNORECASE),
    re.compile(r"INDICATES SAVINGS", re.IGNORECASE),
    re.compile(r"ALL ITEMS MUST BE RETURNED", re.IGNORECASE),
    re.compile(r"TOTAL SAVINGS THIS TRIP", re.IGNORECASE),
    # Payment / card info
    re.compile(r"AID\s*:?\s*A\d{10,}", re.IGNORECASE),
    re.compile(r"RRN\s*:", re.IGNORECASE),
    re.compile(r"Entry Method", re.IGNORECASE),
    re.compile(r"CHIP READ", re.IGNORECASE),
    re.compile(r"CONTACTLESS", re.IGNORECASE),
    re.compile(r"Cntctless", re.IGNORECASE),
    re.compile(r"NO SIGNATURE REQUIRED", re.IGNORECASE),
    re.compile(r"AUTH CODE\s*:", re.IGNORECASE),
    re.compile(r"^AUTH CODE$", re.IGNORECASE),
    re.compile(r"ACCOMPANIED WITH RECEIPT", re.IGNORECASE),
    re.compile(r"^USD\s*\$", re.IGNORECASE),
    re.compile(r"^Issuer$", re.IGNORECASE),
    re.compile(r"ENTRY LEGEND", re.IGNORECASE),
    re.compile(r"^\d{10,}.*RRN", re.IGNORECASE),
    # Receipt footer text
    re.compile(r"APPLIED TO THE ORIGINAL ORDER", re.IGNORECASE),
    re.compile(r"YOU.RE SHOPPING SMARTER", re.IGNORECASE),
    re.compile(r"WHEN YOU RETURN ANY ITEM", re.IGNORECASE),
    re.compile(r"RETURN CREDIT WILL NOT", re.IGNORECASE),
    re.compile(r"PROMOTIONAL DISCOUNT OR COUP", re.IGNORECASE),
    re.compile(r"CUENTENOS EN ESPA", re.IGNORECASE),
    re.compile(r"FOR SHOPPING AT", re.IGNORECASE),
    re.compile(r"^TID\s*:", re.IGNORECASE),
    re.compile(r"INVOICE\s*:", re.IGNORECASE),
    # Store names repeated as items
    re.compile(r"^UWAJIMAYA", re.IGNORECASE),
    re.compile(r"^TRADER JOE.S\s+TRADER", re.IGNORECASE),
    re.compile(r"FAMILY MARKET\s+\d{3}", re.IGNORECASE),
    # Price-only fragments
    re.compile(r"^[FP\s\$\d.]+$"),
    re.compile(r"^\$\s*\d+\.\d{2}(\s+[FT])?$", re.IGNORECASE),
    re.compile(r"^P\s*\$\s*\d", re.IGNORECASE),
    re.compile(r"^FP\s*\$", re.IGNORECASE),
    # Descriptions / quantity headers
    re.compile(r"^Qty\s+Description", re.IGNORECASE),
    # Hex/transaction codes
    re.compile(r"^[0-9A-F]{20,}", re.IGNORECASE),
    re.compile(r"^\d{13,}\s+ACCOMPANIED", re.IGNORECASE),
    # Korean/Chinese-only lines (non-food descriptions on Asian receipts)
    re.compile(r"^[　-鿿가-힯\s/\[\]()]+$"),
    # Uwajimaya footer
    re.compile(r"ALL ALCOHOL SALES", re.IGNORECASE),
    re.compile(r"SEE STORE FOR MORE DETAILS", re.IGNORECASE),
    re.compile(r"NO RETURNS FOR SPECIFIED", re.IGNORECASE),
    re.compile(r"RESTROOM CODE", re.IGNORECASE),
    re.compile(r"THANK YOU FOR SHOPPING", re.IGNORECASE),
    re.compile(r"NOW HIRING", re.IGNORECASE),
    # Trader Joe's noise
    re.compile(r"^STORE TILL$", re.IGNORECASE),
    re.compile(r"^STORE$", re.IGNORECASE),
    re.compile(r"^TILL$", re.IGNORECASE),
    # Store names as items
    re.compile(r"^WHOLE FOODS MARKET", re.IGNORECASE),
    re.compile(r"^WHOLE FOODS$", re.IGNORECASE),
    # Random codes / barcode strings
    re.compile(r"^[A-Z0-9]{10,}$"),
    re.compile(r"^\d{20,}"),
    # Whole Foods footer
    re.compile(r"SHOPPING EXPERIENCE", re.IGNORECASE),
    re.compile(r"CHANCE TO WIN", re.IGNORECASE),
    re.compile(r"GIFT CARD", re.IGNORECASE),
    re.compile(r"wfm\.com", re.IGNORECASE),
    re.compile(r"amazon\.com", re.IGNORECASE),
    re.compile(r"Earn.*back at", re.IGNORECASE),
    re.compile(r"Prime Visa", re.IGNORECASE),

]


def _is_store_noise(line: str) -> bool:
    stripped = line.strip()
    return any(pat.search(stripped) for pat in _NOISE_PATTERNS)


# ═══════════════════════ Name Resolution ═══════════════════════

_OCR_TYPOS = {
    "STRAWBERIES": "STRAWBERRIES",
    "STRWBERRIES": "STRAWBERRIES",
    "BLUBERRIES": "BLUEBERRIES",
    "RASBERRIES": "RASPBERRIES",
    "BROCOLI": "BROCCOLI",
    "TORTILA": "TORTILLA",
    "CHOCLATE": "CHOCOLATE",
    "YOGRT": "YOGURT",
    "MOZZARELA": "MOZZARELLA",
    "PARMESN": "PARMESAN",
}


def resolve_name(raw_name: str) -> str:
    """Expand abbreviations, fix typos, and normalize product name."""
    if not raw_name:
        return raw_name

    text = raw_name.strip()
    text = re.sub(r"(\d+(?:\.\d+)?)\s*o\b", r"\1oz", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\d+\.\s*$", "", text)

    trailing_s = False
    if re.search(r"['\u2019][sS]$", text):
        text = re.sub(r"['\u2019][sS]$", "", text)
        trailing_s = True

    text = " ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", text).split())

    words = text.split()
    words = [_OCR_TYPOS.get(w.upper(), w) for w in words]
    text = " ".join(words)

    words = text.split()
    text = " ".join(
        ABBREVIATIONS.get(w.upper().strip("'\".,;:"), w) for w in words
    )

    if trailing_s:
        last_word = text.split()[-1] if text.split() else ""
        if not last_word.endswith("s"):
            text += "s"

    text = to_title_case(text)
    return text


# ═══════════════════════ OCR Engine ═══════════════════════

class OCREngine:
    """
    OCR engine with Google Vision API (primary) and PaddleOCR (fallback).
    Google Vision gives much better results on receipt images.
    """
    _instance: OCREngine | None = None
    _paddle_ocr: PaddleOCR | None = None
    _use_google: bool = False
    _google_client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self):
        # Try Google Vision first
        if self._google_client is None and not self._use_google:
            try:
                import os
                cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

                if not cred_path:
                    # Auto-search for Google Cloud credentials JSON
                    search_dirs = [
                        Path.home() / "Downloads",
                        PROJECT_ROOT,
                        PROJECT_ROOT / "config",
                        Path.home() / "Desktop",
                    ]
                    # Search patterns for common credential file names
                    patterns = ["*receiptproject*.json", "*service*account*.json", "*vision*.json"]
                    for d in search_dirs:
                        if not d.exists():
                            continue
                        for pattern in patterns:
                            matches = sorted(d.glob(pattern))
                            for f in matches:
                                # Verify it looks like a service account key
                                try:
                                    import json as _json
                                    data = _json.loads(f.read_text(encoding="utf-8"))
                                    if data.get("type") == "service_account":
                                        cred_path = str(f)
                                        break
                                except Exception:
                                    continue
                            if cred_path:
                                break
                        if cred_path:
                            break

                if cred_path and Path(cred_path).exists():
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
                    from google.cloud import vision
                    self._google_client = vision.ImageAnnotatorClient()
                    self._use_google = True
                    print(f"  [OCR] Using Google Vision API (key: {Path(cred_path).name})")
                else:
                    print("  [OCR] No Google credentials found, using PaddleOCR")
            except Exception as e:
                print(f"  [OCR] Google Vision not available ({e}), falling back to PaddleOCR")

        # Fallback to PaddleOCR
        if not self._use_google and self._paddle_ocr is None:
            self._paddle_ocr = PaddleOCR(lang="en")
            print("  [OCR] Using PaddleOCR")

    def run(self, image_path: str) -> list[dict]:
        self._ensure_loaded()
        if self._use_google:
            return self._run_google(image_path)
        else:
            return self._run_paddle(image_path)

    def _run_google(self, image_path: str) -> list[dict]:
        """
        Run Google Vision API text_detection.
        
        Strategy: use text_detection (not document_text_detection).
        The first annotation contains the FULL TEXT with natural line breaks
        that work well for receipts. We store this as _last_full_text.
        
        We also return word-level fragments for compatibility with
        rebuild_text(), but process_image() will prefer the full text.
        """
        from google.cloud import vision

        with open(image_path, "rb") as f:
            content = f.read()

        image = vision.Image(content=content)
        response = self._google_client.text_detection(image=image)

        if response.error.message:
            print(f"  [OCR] Google Vision error: {response.error.message}")
            self._last_full_text = None
            return []

        annotations = response.text_annotations
        if not annotations:
            self._last_full_text = None
            return []

        # First annotation = full text with line breaks (best for receipts)
        self._last_full_text = annotations[0].description

        # Also return word-level fragments
        fragments = []
        for ann in annotations[1:]:
            text = ann.description.strip()
            if not text:
                continue
            vertices = ann.bounding_poly.vertices
            if vertices:
                y_center = sum(v.y for v in vertices) / len(vertices)
                x_left = min(v.x for v in vertices)
            else:
                y_center = 0
                x_left = 0
            fragments.append({
                "x": x_left,
                "y": y_center,
                "text": text,
                "conf": 0.95,
            })

        return fragments

    def get_full_text(self) -> str | None:
        """Return full text from last Google Vision call, or None for PaddleOCR."""
        return getattr(self, '_last_full_text', None)

    def _run_google_simple(self, image_path: str) -> list[dict]:
        """Fallback: simple text_detection returning word-level fragments."""
        from google.cloud import vision

        with open(image_path, "rb") as f:
            content = f.read()

        image = vision.Image(content=content)
        response = self._google_client.text_detection(image=image)

        fragments = []
        annotations = response.text_annotations
        if not annotations:
            return []

        for ann in annotations[1:]:
            text = ann.description.strip()
            if not text:
                continue
            vertices = ann.bounding_poly.vertices
            if vertices:
                y_center = sum(v.y for v in vertices) / len(vertices)
                x_left = min(v.x for v in vertices)
            else:
                y_center = 0
                x_left = 0
            fragments.append({
                "x": x_left,
                "y": y_center,
                "text": text,
                "conf": 0.95,
            })

        return fragments

    def _run_paddle(self, image_path: str) -> list[dict]:
        """Run PaddleOCR (fallback)."""
        if PILImage is not None:
            img = PILImage.open(image_path)
            w, h = img.size
            if h > 10000:
                return self._run_paddle_split(image_path)

        results = list(self._paddle_ocr.predict(image_path))
        return self._extract_paddle_fragments(results)

    def _run_paddle_split(self, image_path: str) -> list[dict]:
        """Run PaddleOCR with splitting for very tall images."""
        img = PILImage.open(image_path)
        w, h = img.size
        max_h = 10000
        overlap = 150

        print(f"  [OCR] Image is {w}x{h} — splitting into segments")
        import tempfile, os
        all_fragments = []
        y_offset = 0
        seg_idx = 0

        while y_offset < h:
            y_end = min(y_offset + max_h, h)
            segment = img.crop((0, y_offset, w, y_end))

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            segment.save(tmp.name)
            tmp.close()

            try:
                results = list(self._paddle_ocr.predict(tmp.name))
                frags = self._extract_paddle_fragments(results)
                for f in frags:
                    f["y"] += y_offset
                all_fragments.extend(frags)
                seg_idx += 1
                print(f"  [OCR] Segment {seg_idx}: y={y_offset}-{y_end} -> {len(frags)} regions")
            finally:
                os.unlink(tmp.name)

            y_offset += max_h - overlap
            if y_end >= h:
                break

        all_fragments = self._deduplicate_fragments(all_fragments)
        return all_fragments

    @staticmethod
    def _deduplicate_fragments(fragments: list[dict], y_tol: float = 15, x_tol: float = 20) -> list[dict]:
        """Remove duplicate fragments from overlapping segments."""
        if not fragments:
            return fragments
        fragments.sort(key=lambda f: (f["y"], f["x"]))
        unique = [fragments[0]]
        for f in fragments[1:]:
            prev = unique[-1]
            if (abs(f["y"] - prev["y"]) < y_tol
                    and abs(f["x"] - prev["x"]) < x_tol
                    and f["text"] == prev["text"]):
                if f["conf"] > prev["conf"]:
                    unique[-1] = f
            else:
                unique.append(f)
        return unique

    @staticmethod
    def _extract_paddle_fragments(ocr_results: list, min_conf: float = 0.4) -> list[dict]:
        fragments = []
        for res in ocr_results:
            if not isinstance(res, dict) or "rec_texts" not in res:
                continue
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            boxes = res.get("dt_polys", [])
            for i, (text, score) in enumerate(zip(texts, scores)):
                text = text.strip()
                if not text or score < min_conf:
                    continue
                if i < len(boxes):
                    box = boxes[i]
                    y_center = sum(p[1] for p in box) / len(box)
                    x_left = min(p[0] for p in box)
                else:
                    y_center = i * 30
                    x_left = 0
                fragments.append({"x": x_left, "y": y_center, "text": text, "conf": score})
        return fragments


# ═══════════════════════ Text Reconstruction ═══════════════════════

def rebuild_text(fragments: list[dict]) -> str:
    frags = [f for f in fragments if len(f["text"].strip()) > 1]
    if not frags:
        return ""

    frags.sort(key=lambda f: f["y"])
    threshold = _calc_line_threshold(frags)

    rows: list[list[dict]] = [[frags[0]]]
    for i in range(1, len(frags)):
        row_mean_y = sum(f["y"] for f in rows[-1]) / len(rows[-1])
        if abs(frags[i]["y"] - row_mean_y) > threshold:
            rows.append([frags[i]])
        else:
            rows[-1].append(frags[i])

    lines = []
    for row in rows:
        row.sort(key=lambda f: f["x"])
        split = _try_split_merged_row(row)
        lines.extend(split)

    return "\n".join(lines)


def _try_split_merged_row(row: list[dict]) -> list[str]:
    if len(row) <= 2:
        return [" ".join(f["text"] for f in row)]

    full_text = " ".join(f["text"] for f in row)
    has_trailing_price = bool(re.search(r"\d+\.\d{2}\s*$", full_text))
    if has_trailing_price and len(full_text) < 50:
        return [full_text]

    by_y = sorted(row, key=lambda f: f["y"])
    y_values = [f["y"] for f in by_y]

    max_gap = 0
    split_idx = -1
    for i in range(1, len(y_values)):
        gap = abs(y_values[i] - y_values[i - 1])
        if gap > max_gap:
            max_gap = gap
            split_idx = i

    if max_gap > 2 and split_idx > 0:
        group_a_y = set(id(f) for f in by_y[:split_idx])
        line_a = [f for f in row if id(f) in group_a_y]
        line_b = [f for f in row if id(f) not in group_a_y]

        if line_a and line_b:
            line_a.sort(key=lambda f: f["x"])
            line_b.sort(key=lambda f: f["x"])

            text_a = " ".join(f["text"] for f in line_a)
            text_b = " ".join(f["text"] for f in line_b)

            if len(text_a) > 3 and len(text_b) > 3:
                mean_y_a = sum(f["y"] for f in line_a) / len(line_a)
                mean_y_b = sum(f["y"] for f in line_b) / len(line_b)
                if mean_y_a <= mean_y_b:
                    return [text_a, text_b]
                else:
                    return [text_b, text_a]

    return [full_text]


def _calc_line_threshold(frags: list[dict]) -> float:
    price_frags = [f for f in frags if re.match(r"^\$?\d+\.\d{2}$", f["text"].strip())]
    non_price = [f for f in frags if not re.match(r"^\$?\d+\.\d{2}$", f["text"].strip())]

    same_line_gaps = []
    if price_frags and non_price:
        for pf in price_frags:
            best_dy = min(abs(pf["y"] - nf["y"]) for nf in non_price)
            if best_dy < 50:
                same_line_gaps.append(best_dy)

    if same_line_gaps:
        p90 = sorted(same_line_gaps)[int(len(same_line_gaps) * 0.9)]
        return max(p90 + 3, 5)

    y_gaps = [abs(frags[i]["y"] - frags[i - 1]["y"])
              for i in range(1, len(frags))
              if abs(frags[i]["y"] - frags[i - 1]["y"]) > 1]
    if y_gaps:
        return max(sorted(y_gaps)[len(y_gaps) // 2] * 0.7, 5)
    return 8


# ═══════════════════════ Receipt Parsing ═══════════════════════

_RE_ITEM = re.compile(r"^(.+?)\s+(\d+\.\d{2})\s*-?\s*[B]?\s*$")
_RE_QTY_EA = re.compile(r"(\d+)\s*EA", re.IGNORECASE)
_RE_QTY_DZ = re.compile(r"(\d+)\s*DZ", re.IGNORECASE)
_RE_WEIGHT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(lb|lbs|oz|kg|g)\s*(?:@|.*?/\s*(?:lb|oz|kg|g))",
    re.IGNORECASE,
)

_HEADER_KEYWORDS = [
    "WHOLE FOODS", "TRADER JOE", "WALMART", "COSTCO", "TARGET",
    "KROGER", "SAFEWAY", "MARKET", "STORE #", "OPEN ",
    "DAILY", "SHARON", "Charlotte",
    "H-E-B", "HEB", "ALDI", "PUBLIX", "SAM'S CLUB", "SAMS CLUB",
    "WHOLESALE",
    "MEIJER", "WINCO", "SPROUTS",
]

_STORE_NAME_MAP = {
    "WHOLE FOODS":  "Whole Foods",
    "TRADER JOE":   "Trader Joe's",
    "WALMART":      "Walmart",
    "COSTCO":       "Costco",
    "WHOLESALE":    "Costco",
    "TARGET":       "Target",
    "KROGER":       "Kroger",
    "SAFEWAY":      "Safeway",
    "ALDI":         "Aldi",
    "H-E-B":        "H-E-B",
    "HEB":          "H-E-B",
    "PUBLIX":       "Publix",
    "SAM'S CLUB":   "Sam's Club",
    "SAMS CLUB":    "Sam's Club",
    "SAM'S":        "Sam's Club",
    "MEIJER":       "Meijer",
    "WINCO":        "WinCo",
    "SPROUTS":      "Sprouts",
    "H MART":       "H Mart",
    "HMART":        "H Mart",
    "UWAJIMAYA":    "Uwajimaya",
    "T&T SUPERMARKET": "T&T Supermarket",
    "T & T SUPERMARKET": "T&T Supermarket",
    "T & T":        "T&T Supermarket",
    "ASIAN FAMILY": "Asian Family Market",
    "ASIANFAMILYMKT": "Asian Family Market",
    "99 RANCH":     "99 Ranch Market",
    "RANCH 99":     "99 Ranch Market",
    "MITSUWA":      "Mitsuwa",
    "MARUKAI":      "Marukai",
    "LOTTE":        "Lotte Plaza",
    "FRED MEYER":   "Fred Meyer",
    "QFC":          "QFC",
    "ALBERTSONS":   "Albertsons",
    "VONS":         "Vons",
    "RALPHS":       "Ralphs",
    "WEGMANS":      "Wegmans",
    "GROCERY OUTLET": "Grocery Outlet",
    "TRADER JOE'S": "Trader Joe's",
}
_STORE_NAMES = list(_STORE_NAME_MAP.keys())


def _fix_ocr(text: str) -> str:
    text = re.sub(r"\b1b\b", "lb", text)
    text = re.sub(r"/1b\b", "/lb", text)
    return text


def _is_header(text: str) -> bool:
    upper = text.upper().strip()
    return any(k in upper for k in _HEADER_KEYWORDS)


def _clean_item_name(name: str) -> str:
    name = re.sub(r"^\*?WT\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^[*#]+\s*", "", name).strip()
    # Strip trailing $ or $XX.XX residue BEFORE tax code removal
    name = re.sub(r"\s*\$\s*\d*\.?\d*\s*$", "", name).strip()
    name = re.sub(r"\s*\$$", "", name).strip()
    name = _strip_upc_and_tax(name)
    # Strip UPC codes in the MIDDLE of text (e.g. "GG CREAM 261050447 GHIRARDELLI")
    name = re.sub(r"\s+\d{9,15}\s+", " ", name).strip()
    # Strip trailing UPC that wasn't caught
    name = re.sub(r"\s+\d{9,15}\s*$", "", name).strip()
    # Strip "APPROVED . THANK YOU" and similar suffixes
    name = re.sub(r"\s+APPROVED.*$", "", name, flags=re.IGNORECASE).strip()
    # Strip leading/trailing periods, colons, parens
    name = re.sub(r"^[.:()]+\s*|\s*[.:()]+$", "", name).strip()
    # Strip leading "1 " or "2 " (qty prefix that leaked into name)
    name = re.sub(r"^\d\s+(?=[A-Z]{2})", "", name).strip()
    return name


def parse_receipt_text(raw_text: str, model_data: dict | None = None) -> Receipt:
    if model_data is not None:
        return _parse_receipt_ml(raw_text, model_data)
    return _parse_receipt_rules(raw_text)


# =============================== Date Extraction ===============================

_DATE_PATTERNS = [
    re.compile(r"\.?DATE:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})", re.IGNORECASE),
    re.compile(r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})", re.IGNORECASE),
    re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})"),
    re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2})(?!\d)"),
]


def _extract_date_from_line(line: str) -> str | None:
    """Try to extract a date string from a line classified as 'date'."""
    for pat in _DATE_PATTERNS:
        m = pat.search(line)
        if m:
            return m.group(1).strip().rstrip(",")
    return None


# ─── ML-based parser ────────────────────────────────────────────────────────

def _split_google_merged_lines(lines: list[str]) -> list[str]:
    """
    Split lines that Google Vision merged from different parts of the receipt.
    Common patterns:
      - Date + footer text: "02/10/2026 12:55 PM APPLIED TO THE ORIGINAL ORDER"
      - Multiple items: "210070458 GG SHRIMP 267008011 GG FRUIT"
      - Price + footer: "$ 8.49 $ 1.99 $ 0.99 ..."
    """
    result = []
    for line in lines:
        if not line.strip():
            result.append(line)
            continue

        # Split if line contains a date followed by unrelated text
        m = re.match(r'^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[AP]M)?)\s+(.{10,})$', line, re.IGNORECASE)
        if m:
            result.append(m.group(1).strip())
            result.append(m.group(2).strip())
            continue

        # Split if line has multiple UPC codes (9+ digits)
        upcs = list(re.finditer(r'(?:^|\s)(\d{9,13})\s', line + ' '))
        if len(upcs) >= 2:
            parts = []
            for i, match in enumerate(upcs):
                start = match.start()
                if i == 0 and start > 0:
                    pre = line[:start].strip()
                    if pre:
                        parts.append(pre)
                end = upcs[i + 1].start() if i + 1 < len(upcs) else len(line)
                part = line[start:end].strip()
                if part:
                    parts.append(part)
            result.extend(parts)
            continue

        result.append(line)
    return result


def _parse_receipt_ml(raw_text: str, model_data: dict) -> Receipt:
    receipt = Receipt()
    items: list[Item] = []

    raw_lines = [_fix_ocr(l).strip() for l in raw_text.split("\n")]
    # Split lines that Google Vision merged from different receipt areas
    raw_lines = _split_google_merged_lines(raw_lines)
    nonempty = [(i, l) for i, l in enumerate(raw_lines) if l]

    if not nonempty:
        return receipt

    all_text = [l for _, l in nonempty]
    total = len(all_text)
    labels = _classify_lines_ml(all_text, model_data)

    current_section = ""
    for idx, (line, label) in enumerate(zip(all_text, labels)):
        if _is_section_header(line):
            labels[idx] = "header"
            current_section = line.strip().upper()
        elif _is_store_noise(line):
            labels[idx] = "noise"

    section_map: list[str] = []
    active_section = ""
    for idx, (line, label) in enumerate(zip(all_text, labels)):
        if _is_section_header(line):
            active_section = line.strip().upper()
        section_map.append(active_section)

    print(f"\n===== ML Classification ({total} lines) =====")
    for (orig_i, line), label in zip(nonempty, labels):
        print(f"  [{label:<8}] {line[:70]}")

    classified = list(zip(all_text, labels))

    pending_items: list[dict] = []
    pending_qty_line: str | None = None
    pending_weight_line: str | None = None

    for idx, (line, label) in enumerate(classified):
        if label == "item":
            pending_items.append({
                "line": line,
                "qty_line": None,
                "weight_line": None,
                "section": section_map[idx] if idx < len(section_map) else "",
            })
        elif label == "qty":
            if pending_items and pending_items[-1]["qty_line"] is None:
                pending_items[-1]["qty_line"] = line
            else:
                pending_qty_line = line
        elif label == "weight":
            if pending_items and pending_items[-1]["weight_line"] is None:
                pending_items[-1]["weight_line"] = line
            else:
                pending_weight_line = line
        elif label == "header":
            pass  # Store detection moved to post-scan below
        elif label == "date":
            if not receipt.date:
                extracted = _extract_date_from_line(line)
                if extracted:
                    receipt.date = extracted
                    print(f"  [DATE] Extracted: {extracted} from: {line[:50]}")
        elif label == "noise":
            upper = line.upper()
            if "SUBTOTAL" in upper and "PACKAGE" not in upper:
                m = re.search(r"\$?(\d+[.,]\d{2})", line)
                if m:
                    receipt.subtotal = m.group(1).replace(",", ".")
            elif re.search(r"\bTAX\b", upper):
                prices = re.findall(r"\$?(\d+[.,]\d{2})", line)
                if prices:
                    receipt.tax = prices[-1].replace(",", ".")
            elif "TOTAL" in upper and "SUB" not in upper and "PACKAGE" not in upper and "SAVINGS" not in upper:
                m = re.search(r"\$?(\d+[.,]\d{2})", line)
                if m:
                    receipt.total = m.group(1).replace(",", ".")

    # Post-scan: detect store name from full text
    if not receipt.store_name:
        full_upper = " ".join(all_text).upper()
        for keyword, display in _STORE_NAME_MAP.items():
            if keyword in full_upper:
                receipt.store_name = display
                print(f"  [STORE] Detected: {display}")
                break

    # Post-scan: extract date from ANY line if not found yet
    if not receipt.date:
        for line, label in classified:
            extracted = _extract_date_from_line(line)
            if extracted:
                receipt.date = extracted
                print(f'  [DATE] Post-scan extracted: {extracted} from: {line[:50]}')
                break

    for entry in pending_items:
        item = _extract_item_from_ml_entry(entry)
        if item:
            items.append(item)

    receipt.items = _merge_duplicates(items)
    return receipt


def _extract_item_from_ml_entry(entry: dict) -> Item | None:
    line = entry["line"]
    qty_line = entry.get("qty_line")
    weight_line = entry.get("weight_line")
    section = entry.get("section", "")

    is_nonfood = section in _NONFOOD_SECTIONS

    name = line
    price: float | None = None
    m = _RE_ITEM.match(line)
    if m:
        name = m.group(1).strip()
        price = float(m.group(2))
    else:
        m2 = re.search(r"\$?(\d+\.\d{2})\s*-?\s*$", line)
        if m2:
            price = float(m2.group(1))
            name = line[: m2.start()].strip()

    name = _clean_item_name(name)
    if not name or len(name) <= 1 or is_noise(name):
        return None
    if re.match(r"^\d+$", name.strip()):
        return None
    if _is_store_noise(name):
        return None

    # Additional filters for Google Vision false positives
    # Skip lines that are mostly numbers/symbols (price fragments)
    alpha_chars = sum(1 for c in name if c.isalpha())
    if alpha_chars < 3:
        return None
    # Skip lines with payment/card keywords
    upper_name = name.upper()
    if any(kw in upper_name for kw in [
        "AID :", "AUTH CODE", "MASTERCARD", "ENTRY METHOD",
        "CONTACTLESS", "CHIP READ", "INVOICE :", "RRN :",
        "USD $", "ACCOMPANIED", "ORIGINAL ORDER",
        "SHOPPING SMARTER", "CUENTENOS", "REUSABLE BAG",
        "SHOPPING BAG", "FIGMINT", "COLGATE", "RAW SUGAR",
    ]):
        return None
    # Skip non-food brands/items
    _NONFOOD_BRANDS = {
        "FIGMINT", "COLGATE", "RAW SUGAR", "UP & UP",
        "EQUATE", "CLOROX", "LYSOL", "WINDEX", "TIDE",
    }
    if name.upper().strip() in _NONFOOD_BRANDS:
        return None
    # Skip cashier/store lines
    if re.match(r"^[A-Z],\s+\w+\s+STORE", name, re.IGNORECASE):
        return None
    if re.match(r"^TILL$", name.strip(), re.IGNORECASE):
        return None
    # Skip lines that are just "USD" or currency
    if name.strip().upper() in ("USD", "CAD", "TILL", "VISA", "LEOSO"):
        return None
    # Skip Korean/Chinese only lines (descriptions, not item names with prices)
    if re.match(r"^[\u3000-\u9fff\uac00-\ud7af\s/\[\]()·]+$", name):
        return None

    qty = 1
    if qty_line:
        m = re.match(
            r"^\s*(\d+)\s*(EA|EACH|PC|PCS|DZ|DOZ|DOZEN)?\b",
            qty_line,
            re.IGNORECASE,
        )
        if m:
            count = int(m.group(1))
            unit = (m.group(2) or "").upper()
            qty = count * 12 if unit in ("DZ", "DOZ", "DOZEN") else count

    weight: Weight | None = extract_weight(name)
    weight_source = ""
    if weight:
        weight_source = "receipt"
    elif weight_line:
        m_w = _RE_WEIGHT.search(weight_line)
        if m_w:
            val = float(m_w.group(1).replace(",", "."))
            unit = m_w.group(2).lower()
            if unit == "lbs":
                unit = "lb"
            weight = Weight(value=val, unit=unit)
            weight_source = "receipt"

    return Item(
        name=name,
        resolved_name=resolve_name(name),
        qty=qty,
        weight=weight,
        weight_source="nonfood" if is_nonfood and not weight else weight_source,
        line_price=price,
    )


# ─── Rule-based parser ─────────────────────────────────────────────

def _parse_receipt_rules(raw_text: str) -> Receipt:
    receipt = Receipt()
    items: list[Item] = []
    pending_weight: Weight | None = None
    pending_qty: int | None = None
    pending_name: str | None = None
    pending_price: float | None = None
    header_parts: list[str] = []

    def _add_item(name: str, price: float):
        nonlocal pending_weight, pending_qty
        name = _clean_item_name(name)
        if not name or len(name) <= 1 or is_noise(name):
            return
        item = Item(
            name=name,
            resolved_name=resolve_name(name),
            qty=1,
            weight=extract_weight(name),
            line_price=price,
        )
        if item.weight:
            item.weight_source = "receipt"
        if pending_weight and not item.weight:
            item.weight = pending_weight
            item.weight_source = "receipt"
            pending_weight = None
        if pending_qty:
            item.qty = pending_qty
            pending_qty = None
        items.append(item)

    for line in raw_text.split("\n"):
        line = _fix_ocr(line).strip()
        if not line:
            continue
        upper = line.upper()

        # Try date extraction in rule-based mode too
        if not receipt.date and re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", line):
            extracted = _extract_date_from_line(line)
            if extracted:
                receipt.date = extracted

        if not receipt.store_name and _is_header(line):
            header_parts.append(upper)
            combined = " ".join(header_parts)
            for sn in _STORE_NAMES:
                if sn in combined:
                    receipt.store_name = sn
                    break
            continue

        if _is_section_header(line):
            if not receipt.store_name:
                receipt.store_name = "Target"
            continue

        if _is_store_noise(line):
            continue

        if "SUBTOTAL" in upper:
            m = re.search(r"\$?(\d+[.,]\d{2})", line)
            if m:
                receipt.subtotal = m.group(1).replace(",", ".")
            continue
        if "TAX" in upper:
            m = re.search(r"\$?(\d*\.?\d{2})", line)
            if m:
                receipt.tax = m.group(1)
            continue
        if "BAL" in upper or ("TOTAL" in upper and "SUB" not in upper):
            m = re.search(r"\$?(\d+[.,]\d{2})", line)
            if m:
                receipt.total = m.group(1).replace(",", ".")
            continue

        if re.search(r"\boff\b", line, re.IGNORECASE):
            continue
        if re.search(r"\d+\.\d{2}\s*-\s*$", line):
            continue
        if re.match(r"^.*(TARE|ITEM)\s*=", upper):
            continue
        if re.match(r"^[\d\s#*=.$]+$", line):
            continue

        m_w = _RE_WEIGHT.search(line)
        if m_w:
            val = float(m_w.group(1).replace(",", "."))
            unit = m_w.group(2).lower()
            if unit == "lbs":
                unit = "lb"
            pending_weight = Weight(value=val, unit=unit)
            continue

        if "@" in line:
            m_ea = _RE_QTY_EA.search(line)
            m_dz = _RE_QTY_DZ.search(line)
            if m_ea:
                pending_qty = int(m_ea.group(1))
                continue
            if m_dz:
                pending_qty = int(m_dz.group(1)) * 12
                continue

        if re.match(r"^\d+\s*(OZ|EA|DZ)\b", line, re.IGNORECASE):
            continue

        if re.match(r"^\d+\.\d{2}$", line.strip()):
            price_val = float(line.strip())
            if pending_name:
                _add_item(pending_name, price_val)
                pending_name = None
                pending_price = None
            else:
                pending_price = price_val
            continue

        m = _RE_ITEM.match(line)
        if m:
            pending_name = None
            pending_price = None
            _add_item(m.group(1).strip(), float(m.group(2)))
        else:
            clean = _clean_item_name(line)
            if clean and len(clean) > 1 and not is_noise(clean):
                if pending_price is not None:
                    _add_item(clean, pending_price)
                    pending_price = None
                    pending_name = None
                else:
                    pending_name = clean
            else:
                pending_name = None
                pending_price = None

    receipt.items = _merge_duplicates(items)
    return receipt


def _merge_duplicates(items: list[Item]) -> list[Item]:
    merged: dict[str, Item] = {}
    order: list[str] = []
    for item in items:
        key = item.name.upper()
        if key in merged:
            existing = merged[key]
            existing.qty = (existing.qty or 1) + (item.qty or 1)
            if existing.line_price and item.line_price:
                existing.line_price = round(existing.line_price + item.line_price, 2)
        else:
            merged[key] = item
            order.append(key)
    return [merged[k] for k in order]


# ═══════════════════════ Weight Resolution ═══════════════════════

CATEGORY_DEFAULTS = {
    "cookie": (6.3, "oz"), "cookies": (6.3, "oz"),
    "chip": (4.8, "oz"), "chips": (4.8, "oz"),
    "crackers": (5.0, "oz"), "cracker": (5.0, "oz"),
    "popcorn": (3.5, "oz"), "nuts": (7.1, "oz"), "pretzels": (5.0, "oz"),
    "granola bar": (1.8, "oz"), "energy bar": (1.8, "oz"),
    "protein bar": (1.8, "oz"), "trail mix": (7.1, "oz"),
    "milk": (2.2, "lb"), "yogurt": (7.1, "oz"), "cheese": (7.1, "oz"),
    "cottage cheese": (7.1, "oz"), "cream cheese": (7.8, "oz"),
    "butter": (8.8, "oz"), "cream": (8.8, "oz"), "sour cream": (8.8, "oz"),
    "ice cream": (14.4, "oz"),
    "bread": (11.6, "oz"), "tortilla": (12.6, "oz"), "bagels": (12.0, "oz"),
    "bagel": (12.0, "oz"), "rolls": (11.6, "oz"), "muffins": (7.9, "oz"),
    "croissants": (8.1, "oz"), "pita": (8.0, "oz"), "buns": (11.6, "oz"),
    "english muffins": (12.0, "oz"), "baguette": (10.0, "oz"),
    "cereal": (14.1, "oz"), "granola": (14.1, "oz"), "oatmeal": (14.1, "oz"),
    "pasta": (16.0, "oz"), "rice": (28.2, "oz"), "noodles": (8.0, "oz"),
    "flour": (16.0, "oz"),
    "canned": (11.6, "oz"), "beans": (15.0, "oz"),
    "canned vegetables": (14.1, "oz"), "canned fruit": (15.0, "oz"),
    "canned soup": (15.3, "oz"), "canned fish": (5.0, "oz"),
    "soup": (14.8, "oz"),
    "frozen": (14.1, "oz"), "frozen vegetables": (21.2, "oz"),
    "frozen fruit": (15.9, "oz"), "frozen pizza": (13.8, "oz"),
    "frozen meal": (14.1, "oz"), "frozen dessert": (14.6, "oz"),
    "cuisine": (14.1, "oz"),
    "frozen fries": (26.5, "oz"),
    "chicken": (12.5, "oz"), "chicken breast": (12.5, "oz"),
    "ground beef": (16.0, "oz"), "beef": (10.6, "oz"),
    "pork": (7.1, "oz"), "turkey": (10.6, "oz"), "turkey breast": (8.0, "oz"),
    "bacon": (6.3, "oz"), "sausage": (10.6, "oz"), "hot dog": (16.0, "oz"),
    "deli meat": (8.0, "oz"), "ham": (8.0, "oz"), "salami": (3.5, "oz"),
    "salmon": (12.0, "oz"), "tuna": (5.0, "oz"), "shrimp": (16.0, "oz"),
    "eggs": (18.0, "oz"),
    "apples": (2.2, "lb"), "banana": (27.1, "oz"), "orange": (3.3, "lb"),
    "lemon": (17.6, "oz"), "avocado": (12.0, "oz"), "mango": (10.6, "oz"),
    "strawberries": (15.9, "oz"), "blueberries": (10.6, "oz"),
    "berries": (10.6, "oz"), "grapes": (17.6, "oz"),
    "tomato": (14.1, "oz"), "potato": (2.2, "lb"), "onion": (17.6, "oz"),
    "carrot": (17.6, "oz"), "lettuce": (15.9, "oz"),
    "broccoli": (17.6, "oz"), "spinach": (10.6, "oz"),
    "cucumber": (15.9, "oz"), "pepper": (1.8, "oz"),
    "mushroom": (8.8, "oz"), "plums": (1.0, "lb"),
    "juice": (2.2, "lb"), "orange juice": (2.2, "lb"),
    "apple juice": (2.2, "lb"), "soda": (17.6, "oz"),
    "water": (26.5, "oz"), "sparkling water": (26.5, "oz"),
    "energy drink": (12.5, "oz"), "sports drink": (12.5, "oz"),
    "coffee": (8.8, "oz"), "tea": (2.0, "oz"),
    "plant milk": (2.2, "lb"), "almond milk": (2.2, "lb"),
    "oat milk": (2.2, "lb"), "soy milk": (2.2, "lb"),
    "coconut milk": (2.2, "lb"),
    "ketchup": (17.6, "oz"), "mustard": (8.0, "oz"),
    "mayonnaise": (13.2, "oz"), "bbq sauce": (13.4, "oz"),
    "hot sauce": (8.6, "oz"), "salsa": (8.8, "oz"),
    "pasta sauce": (12.3, "oz"), "sauce": (12.3, "oz"),
    "salad dressing": (12.3, "oz"), "soy sauce": (8.8, "oz"),
    "vinegar": (17.6, "oz"), "oil": (26.5, "oz"), "olive oil": (26.5, "oz"),
    "hummus": (7.1, "oz"), "peanut butter": (14.5, "oz"),
    "jam": (12.0, "oz"), "honey": (16.0, "oz"), "syrup": (17.6, "oz"),
    "pickles": (11.3, "oz"),
    "sugar": (17.6, "oz"), "chocolate": (6.3, "oz"),
    "chocolate bar": (6.3, "oz"), "baking mix": (14.8, "oz"),
    "pizza": (14.1, "oz"), "burrito": (9.7, "oz"),
    "ramen": (3.0, "oz"), "instant noodles": (3.0, "oz"),
    "tortellini": (8.8, "oz"), "ravioli": (10.6, "oz"),
    "waffle": (7.1, "oz"), "pancake": (8.8, "oz"),
    "baby food": (8.4, "oz"),
    "hazelnut spread": (12.7, "oz"), "chocolate spread": (12.3, "oz"),
    "cashew": (7.1, "oz"),
}


def _parse_quantity_str(qty_str: str) -> Weight | None:
    if not qty_str:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g|lb|lbs|oz|fl oz|ml|l|cl)\b", qty_str, re.IGNORECASE)
    if not m:
        return None
    unit = m.group(2).lower()
    if unit == "lbs":
        unit = "lb"
    if unit == "fl oz":
        unit = "oz"
    if unit == "cl":
        return Weight(value=float(m.group(1)) * 10, unit="ml")
    return Weight(value=float(m.group(1)), unit=unit)


_CONDIMENT_KEYWORDS = {
    "salt", "pepper", "spice", "seasoning", "sauce packet", "powder",
    "extract", "flavor", "syrup", "vinegar", "dressing",
}

def _weight_to_grams(w: Weight) -> float:
    return w.grams


def _weight_is_plausible(w: Weight, product_name: str) -> bool:
    grams = w.grams
    if grams > 25000:
        return False
    if grams >= 20:
        return True
    name_lower = product_name.lower()
    if any(k in name_lower for k in _CONDIMENT_KEYWORDS):
        return True
    return False


class OffDatabase:
    """Cached OFF database lookups — connection reused, results cached."""

    def __init__(self, db_path: str):
        self._path = db_path
        self._conn: sqlite3.Connection | None = None
        self._cache: dict[str, tuple[Weight | None, Nutriments | None]] = {}

    def _ensure_conn(self) -> sqlite3.Connection | None:
        if self._conn is not None:
            return self._conn
        if not Path(self._path).exists():
            return None
        self._conn = sqlite3.connect(self._path)
        return self._conn

    def _has_nutrition_columns(self) -> bool:
        """Check if the database has nutrition columns."""
        conn = self._ensure_conn()
        if conn is None:
            return False
        try:
            conn.execute("SELECT energy_kcal FROM products LIMIT 1")
            return True
        except sqlite3.OperationalError:
            return False

    def lookup(self, product_name: str, store_name: str = "") -> tuple[Weight | None, Nutriments | None]:
        """Lookup product weight and nutriments. Returns (Weight, Nutriments) tuple."""
        cache_key = f"{product_name.lower()}|{store_name.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = self._do_lookup(product_name, store_name)
        self._cache[cache_key] = result
        return result

    def _row_to_nutriments(self, row, offset: int) -> Nutriments | None:
        """Extract Nutriments from a DB row starting at column offset."""
        if row is None:
            return None
        vals = row[offset:offset+9]
        if all(v is None for v in vals):
            return None
        return Nutriments(
            energy_kcal=vals[0],
            fat=vals[1],
            saturated_fat=vals[2],
            carbohydrates=vals[3],
            sugars=vals[4],
            fiber=vals[5],
            proteins=vals[6],
            salt=vals[7],
            sodium=vals[8],
        )

    def _select_cols(self) -> str:
        """Return the SELECT columns string."""
        if self._has_nutrition_columns():
            return ("quantity, energy_kcal, fat, saturated_fat, "
                    "carbohydrates, sugars, fiber, proteins, salt, sodium")
        return "quantity"

    def _parse_row(self, row) -> tuple[Weight | None, Nutriments | None]:
        """Parse a DB row into (Weight, Nutriments)."""
        if row is None:
            return None, None
        w = _parse_quantity_str(row[0])
        if self._has_nutrition_columns():
            n = self._row_to_nutriments(row, 1)
        else:
            n = None
        return w, n

    def _do_lookup(self, product_name: str, store_name: str) -> tuple[Weight | None, Nutriments | None]:
        conn = self._ensure_conn()
        if conn is None:
            return None, None
        name_lower = product_name.lower()
        cols = self._select_cols()

        # 1. Exact match
        row = conn.execute(
            f"SELECT {cols} FROM products WHERE LOWER(product_name) = ? AND quantity != '' LIMIT 1",
            (name_lower,),
        ).fetchone()
        if row:
            return self._parse_row(row)

        # 2. Contains match
        row = conn.execute(
            f"SELECT {cols} FROM products WHERE LOWER(product_name) LIKE ? AND quantity != '' LIMIT 1",
            (f"%{name_lower}%",),
        ).fetchone()
        if row:
            return self._parse_row(row)

        # 3. Store-specific match
        if store_name:
            row = conn.execute(
                f"""SELECT {cols} FROM products 
                   WHERE LOWER(product_name) LIKE ? 
                   AND LOWER(brands) LIKE ? 
                   AND quantity != '' LIMIT 1""",
                (f"%{name_lower}%", f"%{store_name.lower()}%"),
            ).fetchone()
            if row:
                return self._parse_row(row)

        # 4. Last-two-words match
        words = product_name.split()
        if len(words) >= 2:
            tail = " ".join(words[-2:]).lower()
            row = conn.execute(
                f"SELECT {cols} FROM products WHERE LOWER(product_name) LIKE ? AND quantity != '' LIMIT 1",
                (f"%{tail}%",),
            ).fetchone()
            if row:
                return self._parse_row(row)

        # 5. Strip common modifiers and retry
        skip_words = {"organic", "fresh", "frozen", "natural", "whole", "raw",
                      "unsweetened", "roasted", "sliced", "shredded", "chopped",
                      "breaded", "grilled", "smoked", "reduced", "low", "fat",
                      "free", "each", "baby", "par", "baked", "just"}
        core_words = [w for w in words if w.lower() not in skip_words]
        if core_words and len(core_words) < len(words):
            core_name = " ".join(core_words).lower()
            row = conn.execute(
                f"SELECT {cols} FROM products WHERE LOWER(product_name) LIKE ? AND quantity != '' LIMIT 1",
                (f"%{core_name}%",),
            ).fetchone()
            if row:
                return self._parse_row(row)

        return None, None

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Fuzzy name resolution ──────────────────────────────────────────────

    _GENERIC_WORDS = {
        "organic", "fresh", "frozen", "natural", "whole", "raw", "just",
        "original", "classic", "premium", "select", "best", "great", "value",
        "bag", "box", "pack", "each", "size", "large", "small", "mini",
        "new", "old", "no", "with", "and", "or", "the", "in", "of",
        "pasta", "sauce", "salad", "bread", "milk", "juice", "water",
        "bar", "mix", "style", "free", "less", "more", "extra", "super",
    }

    _SENSITIVE_WORDS = {
        "dog", "cat", "pet", "baby", "infant", "puppy", "kitten",
        "canine", "feline", "veterinary", "vet",
    }

    def fuzzy_resolve_name(self, raw_name: str, threshold: int = 75) -> str | None:
        conn = self._ensure_conn()
        if conn is None:
            return None

        from rapidfuzz import fuzz

        cleaned_name = _strip_upc_and_tax(raw_name)
        if not cleaned_name or len(cleaned_name) <= 1:
            return None

        words = re.sub(r"[^a-zA-Z\s]", " ", cleaned_name).split()
        filter_words = [
            w for w in words
            if len(w) >= 3 and w.lower() not in self._GENERIC_WORDS
        ]
        if not filter_words:
            return None

        filter_words = filter_words[:3]
        conditions = " AND ".join(
            "LOWER(product_name) LIKE ?" for _ in filter_words
        )
        params = [f"%{w.lower()}%" for w in filter_words]

        rows = conn.execute(
            f"SELECT product_name FROM products WHERE {conditions} LIMIT 200",
            params,
        ).fetchall()

        if not rows:
            rows = conn.execute(
                "SELECT product_name FROM products WHERE LOWER(product_name) LIKE ? LIMIT 200",
                (f"%{filter_words[0].lower()}%",),
            ).fetchall()

        if not rows:
            return None

        candidates = [r[0] for r in rows if r[0]]
        query = cleaned_name.upper()

        best_name = None
        best_score = 0
        for candidate in candidates:
            score = fuzz.token_set_ratio(query, candidate.upper())
            if score > best_score:
                best_score = score
                best_name = candidate

        if best_score >= threshold and best_name:
            query_words = set(cleaned_name.lower().split())
            result_words = set(best_name.lower().split())
            intruding = (result_words & self._SENSITIVE_WORDS) - query_words
            if intruding:
                return None

            if len(best_name.split()) > len(cleaned_name.split()) + 3:
                return None

            result_alpha_words = [w for w in best_name.split() if re.search(r"[a-zA-Z]", w)]
            query_alpha_words = [w for w in cleaned_name.split() if re.search(r"[a-zA-Z]", w)]
            if len(result_alpha_words) < len(query_alpha_words) - 1:
                return None

            if re.match(r"^\d+%?\s", best_name):
                return None

            cleaned = best_name.strip()
            if len(cleaned) > 50:
                cleaned = " ".join(cleaned.split()[:5])
            return cleaned.title()

        return None



class UsdaDatabase:
    """USDA SR Legacy database lookup — fallback when OFF has no nutrition data."""

    def __init__(self, db_path: str):
        self._path = db_path
        self._conn: sqlite3.Connection | None = None
        self._cache: dict[str, Nutriments | None] = {}

    def _ensure_conn(self) -> sqlite3.Connection | None:
        if self._conn is not None:
            return self._conn
        if not Path(self._path).exists():
            return None
        self._conn = sqlite3.connect(self._path)
        return self._conn

    def lookup(self, product_name: str) -> Nutriments | None:
        cache_key = product_name.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = self._do_lookup(product_name)
        self._cache[cache_key] = result
        return result

    def _row_to_nutriments(self, row) -> Nutriments | None:
        if row is None:
            return None
        return Nutriments(
            energy_kcal=row[0], fat=row[1], saturated_fat=row[2],
            carbohydrates=row[3], sugars=row[4], fiber=row[5],
            proteins=row[6], salt=row[7], sodium=row[8],
        )

    def _do_lookup(self, product_name: str) -> Nutriments | None:
        conn = self._ensure_conn()
        if conn is None:
            return None
        name_lower = product_name.lower()
        cols = "energy_kcal, fat, saturated_fat, carbohydrates, sugars, fiber, proteins, salt, sodium"
        row = conn.execute(
            f"SELECT {cols} FROM products WHERE LOWER(product_name) = ? LIMIT 1",
            (name_lower,),).fetchone()
        if row and row[0] is not None:
            return self._row_to_nutriments(row)
        row = conn.execute(
            f"SELECT {cols} FROM products WHERE LOWER(product_name) LIKE ? LIMIT 1",
            (f"%{name_lower}%",),).fetchone()
        if row and row[0] is not None:
            return self._row_to_nutriments(row)
        words = product_name.split()
        if len(words) >= 2:
            tail = " ".join(words[-2:]).lower()
            row = conn.execute(
                f"SELECT {cols} FROM products WHERE LOWER(product_name) LIKE ? LIMIT 1",
                (f"%{tail}%",),).fetchone()
            if row and row[0] is not None:
                return self._row_to_nutriments(row)
        keywords = [w for w in words if len(w) >= 4]
        keywords.sort(key=len, reverse=True)
        for kw in keywords[:2]:
            row = conn.execute(
                f"SELECT {cols} FROM products WHERE LOWER(product_name) LIKE ? LIMIT 1",
                (f"%{kw.lower()}%",),).fetchone()
            if row and row[0] is not None:
                return self._row_to_nutriments(row)
        return None

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None



def _guess_category_default(product_name: str) -> Weight | None:
    name_lower = product_name.lower()
    for keyword, (value, unit) in CATEGORY_DEFAULTS.items():
        if keyword in name_lower:
            return Weight(value=value, unit=unit)
    for brand, category in _BRAND_CATEGORY_MAP.items():
        if brand in name_lower:
            if category in CATEGORY_DEFAULTS:
                value, unit = CATEGORY_DEFAULTS[category]
                return Weight(value=value, unit=unit)
    return None


_BRAND_CATEGORY_MAP = {
    "breyers": "ice cream", "haagen": "ice cream", "ben & jerry": "ice cream",
    "talenti": "ice cream", "blue bunny": "ice cream",
    "reddi wip": "cream", "cool whip": "cream",
    "chobani": "yogurt", "fage": "yogurt", "dannon": "yogurt",
    "yoplait": "yogurt", "siggi": "yogurt", "oikos": "yogurt",
    "cheerios": "cereal", "frosted flakes": "cereal", "quaker": "cereal",
    "coca cola": "soda", "pepsi": "soda", "sprite": "soda",
    "dr pepper": "soda", "mountain dew": "soda",
    "tropicana": "juice", "simply": "juice", "minute maid": "juice",
    "doritos": "chips", "lays": "chips", "pringles": "chips",
    "tostitos": "chips", "ruffles": "chips", "fritos": "chips",
    "oreo": "cookie", "chips ahoy": "cookie", "nutter butter": "cookie",
    "ritz": "crackers", "triscuit": "crackers", "wheat thins": "crackers",
    "goldfish": "crackers", "cheez-it": "crackers",
    "lean cuisine": "frozen meal", "stouffer": "frozen meal",
    "marie callender": "frozen meal", "healthy choice": "frozen meal",
    "evol": "frozen meal", "amy's": "frozen meal",
    "digiorno": "frozen pizza", "totino": "frozen pizza",
    "red baron": "frozen pizza",
    "heinz": "ketchup", "french's": "mustard",
    "hellmann": "mayonnaise", "hidden valley": "salad dressing",
    "smucker": "jam", "welch": "jam",
    "jif": "peanut butter", "skippy": "peanut butter", "peter pan": "peanut butter",
    "oscar mayer": "deli meat", "jennie-o": "turkey", "jennie-0": "turkey",
    "tyson": "chicken", "perdue": "chicken", "butterball": "turkey",
    "hillshire": "sausage", "jimmy dean": "sausage",
    "bird": "frozen meal",
    "naked": "juice", "v8": "juice",
    "aquafina": "water", "dasani": "water",
    "gatorade": "sports drink", "powerade": "sports drink",
    "starbucks": "coffee", "dunkin": "coffee",
    "nature valley": "granola bar", "kind": "granola bar",
    "clif": "energy bar", "rxbar": "protein bar",
    "pepperidge": "bread", "arnold": "bread",
    "sara lee": "bread", "dave's killer": "bread",
}


_CONFIDENCE_RATIO_THRESHOLD = 5.0


def _compare_weights(off_w: Weight, default_w: Weight) -> float:
    off_g = off_w.grams
    def_g = default_w.grams
    if off_g == 0 or def_g == 0:
        return 999.0
    ratio = off_g / def_g
    return max(ratio, 1.0 / ratio)


def resolve_weight(item: Item, store_name: str, off_db: OffDatabase, usda_db: "UsdaDatabase | None" = None):
    """
    Resolve weight, nutriments, and compute actual nutrition values.

    Priority:
      1. Weight already on the item (from receipt) → keep it
      2. Direct OFF lookup → sanity-checked AND cross-checked vs category default
      3. Fuzzy OFF lookup → updates resolved_name AND weight (same cross-check)
      4. Category default → weight only (no nutriments)

    After weight is resolved:
      - Convert to grams (weight_grams = weight × qty)
      - If nutriments_per100g available, scale to actual weight
    """
    if item.weight and item.weight_source:
        # Already have weight from receipt, but still try to get nutriments
        if item.weight_source != "nonfood":
            _lookup_nutriments_only(item, store_name, off_db, usda_db=usda_db)
        _finalize_weight_and_nutrition(item, usda_db=usda_db)
        return

    # Skip OFF lookup for non-food items
    if item.weight_source == "nonfood":
        return

    search_name = item.resolved_name or item.name
    category_w = _guess_category_default(search_name)

    # 2. Direct lookup
    w, n = off_db.lookup(search_name, store_name)
    if w and _weight_is_plausible(w, search_name):
        if category_w:
            ratio = _compare_weights(w, category_w)
            if ratio > _CONFIDENCE_RATIO_THRESHOLD:
                print(f"  [WARNING] {search_name}: OFF={w.value}{w.unit} vs "
                      f"default={category_w.value}{category_w.unit} "
                      f"(ratio={ratio:.1f}x) → using default")
                item.weight = category_w
                item.weight_source = "category_default_override"
                item.nutriments_per100g = n  # still use OFF nutriments
                _finalize_weight_and_nutrition(item, usda_db=usda_db)
                return
        item.weight = w
        item.weight_source = "off_db"
        item.nutriments_per100g = n
        _finalize_weight_and_nutrition(item, usda_db=usda_db)
        return

    # 3. Fuzzy lookup
    fuzzy_query = item.resolved_name or item.name
    better_name = off_db.fuzzy_resolve_name(fuzzy_query)
    if better_name:
        if better_name.upper() != item.resolved_name.upper():
            item.resolved_name = better_name
        w, n = off_db.lookup(better_name, store_name)
        if w and _weight_is_plausible(w, better_name):
            cat_w = category_w or _guess_category_default(better_name)
            if cat_w:
                ratio = _compare_weights(w, cat_w)
                if ratio > _CONFIDENCE_RATIO_THRESHOLD:
                    print(f"  [WARNING] {better_name}: OFF_fuzzy={w.value}{w.unit} vs "
                          f"default={cat_w.value}{cat_w.unit} "
                          f"(ratio={ratio:.1f}x) → using default")
                    item.weight = cat_w
                    item.weight_source = "category_default_override"
                    item.nutriments_per100g = n
                    _finalize_weight_and_nutrition(item, usda_db=usda_db)
                    return
            item.weight = w
            item.weight_source = "off_db_fuzzy"
            item.nutriments_per100g = n
            _finalize_weight_and_nutrition(item, usda_db=usda_db)
            return

    # 4. Category default (no nutriments)
    if category_w:
        item.weight = category_w
        item.weight_source = "category_default"
        _finalize_weight_and_nutrition(item, usda_db=usda_db)


def _lookup_nutriments_only(item: Item, store_name: str, off_db: OffDatabase, usda_db: "UsdaDatabase | None" = None):
    """When weight is already known (from receipt), still look up nutriments from OFF.
    Falls back to USDA produce table if OFF has no data."""
    search_name = item.resolved_name or item.name
    _, n = off_db.lookup(search_name, store_name)
    if n and n.has_data() and not n.is_all_zero():
        item.nutriments_per100g = n
        return

    # Try fuzzy
    better_name = off_db.fuzzy_resolve_name(search_name)
    if better_name:
        _, n = off_db.lookup(better_name, store_name)
        if n and n.has_data() and not n.is_all_zero():
            item.nutriments_per100g = n
            if better_name.upper() != item.resolved_name.upper():
                item.resolved_name = better_name
            return

    # USDA fallback for fresh produce / staples
    usda = _lookup_usda_fallback(search_name, usda_db=usda_db)
    if usda:
        item.nutriments_per100g = usda


# ═══════════════════════ USDA Produce Nutrition Fallback ═══════════════════════
# Per 100g values from USDA FoodData Central for common fresh produce & staples.
# Used when OFF database has no nutrition data (or all-zero bad data).

_USDA_NUTRITION: dict[str, Nutriments] = {
    # ── Fruits ──
    "apple":        Nutriments(energy_kcal=52,  fat=0.2, saturated_fat=0.0, carbohydrates=13.8, sugars=10.4, fiber=2.4, proteins=0.3, salt=0.0, sodium=0.001),
    "banana":       Nutriments(energy_kcal=89,  fat=0.3, saturated_fat=0.1, carbohydrates=22.8, sugars=12.2, fiber=2.6, proteins=1.1, salt=0.0, sodium=0.001),
    "orange":       Nutriments(energy_kcal=47,  fat=0.1, saturated_fat=0.0, carbohydrates=11.8, sugars=9.4,  fiber=2.4, proteins=0.9, salt=0.0, sodium=0.0),
    "lemon":        Nutriments(energy_kcal=29,  fat=0.3, saturated_fat=0.0, carbohydrates=9.3,  sugars=2.5,  fiber=2.8, proteins=1.1, salt=0.0, sodium=0.002),
    "mango":        Nutriments(energy_kcal=60,  fat=0.4, saturated_fat=0.1, carbohydrates=15.0, sugars=13.7, fiber=1.6, proteins=0.8, salt=0.0, sodium=0.001),
    "avocado":      Nutriments(energy_kcal=160, fat=14.7,saturated_fat=2.1, carbohydrates=8.5,  sugars=0.7,  fiber=6.7, proteins=2.0, salt=0.0, sodium=0.007),
    "strawberr":    Nutriments(energy_kcal=32,  fat=0.3, saturated_fat=0.0, carbohydrates=7.7,  sugars=4.9,  fiber=2.0, proteins=0.7, salt=0.0, sodium=0.001),
    "blueberr":     Nutriments(energy_kcal=57,  fat=0.3, saturated_fat=0.0, carbohydrates=14.5, sugars=10.0, fiber=2.4, proteins=0.7, salt=0.0, sodium=0.001),
    "blackberr":    Nutriments(energy_kcal=43,  fat=0.5, saturated_fat=0.0, carbohydrates=9.6,  sugars=4.9,  fiber=5.3, proteins=1.4, salt=0.0, sodium=0.001),
    "raspberr":     Nutriments(energy_kcal=52,  fat=0.7, saturated_fat=0.0, carbohydrates=11.9, sugars=4.4,  fiber=6.5, proteins=1.2, salt=0.0, sodium=0.001),
    "grape":        Nutriments(energy_kcal=69,  fat=0.2, saturated_fat=0.1, carbohydrates=18.1, sugars=15.5, fiber=0.9, proteins=0.7, salt=0.0, sodium=0.002),
    "pear":         Nutriments(energy_kcal=57,  fat=0.1, saturated_fat=0.0, carbohydrates=15.2, sugars=9.8,  fiber=3.1, proteins=0.4, salt=0.0, sodium=0.001),
    "peach":        Nutriments(energy_kcal=39,  fat=0.3, saturated_fat=0.0, carbohydrates=9.5,  sugars=8.4,  fiber=1.5, proteins=0.9, salt=0.0, sodium=0.0),
    "plum":         Nutriments(energy_kcal=46,  fat=0.3, saturated_fat=0.0, carbohydrates=11.4, sugars=9.9,  fiber=1.4, proteins=0.7, salt=0.0, sodium=0.0),
    "cherry":       Nutriments(energy_kcal=50,  fat=0.3, saturated_fat=0.1, carbohydrates=12.2, sugars=8.5,  fiber=1.6, proteins=1.0, salt=0.0, sodium=0.003),
    "watermelon":   Nutriments(energy_kcal=30,  fat=0.2, saturated_fat=0.0, carbohydrates=7.6,  sugars=6.2,  fiber=0.4, proteins=0.6, salt=0.0, sodium=0.001),
    "cantaloupe":   Nutriments(energy_kcal=34,  fat=0.2, saturated_fat=0.1, carbohydrates=8.2,  sugars=7.9,  fiber=0.9, proteins=0.8, salt=0.0, sodium=0.016),
    "pineapple":    Nutriments(energy_kcal=50,  fat=0.1, saturated_fat=0.0, carbohydrates=13.1, sugars=9.9,  fiber=1.4, proteins=0.5, salt=0.0, sodium=0.001),
    "kiwi":         Nutriments(energy_kcal=61,  fat=0.5, saturated_fat=0.0, carbohydrates=14.7, sugars=9.0,  fiber=3.0, proteins=1.1, salt=0.0, sodium=0.003),
    # ── Vegetables ──
    "broccoli":     Nutriments(energy_kcal=34,  fat=0.4, saturated_fat=0.0, carbohydrates=6.6,  sugars=1.7,  fiber=2.6, proteins=2.8, salt=0.0, sodium=0.033),
    "carrot":       Nutriments(energy_kcal=41,  fat=0.2, saturated_fat=0.0, carbohydrates=9.6,  sugars=4.7,  fiber=2.8, proteins=0.9, salt=0.0, sodium=0.069),
    "spinach":      Nutriments(energy_kcal=23,  fat=0.4, saturated_fat=0.1, carbohydrates=3.6,  sugars=0.4,  fiber=2.2, proteins=2.9, salt=0.0, sodium=0.079),
    "lettuce":      Nutriments(energy_kcal=15,  fat=0.2, saturated_fat=0.0, carbohydrates=2.9,  sugars=1.3,  fiber=1.3, proteins=1.4, salt=0.0, sodium=0.028),
    "tomato":       Nutriments(energy_kcal=18,  fat=0.2, saturated_fat=0.0, carbohydrates=3.9,  sugars=2.6,  fiber=1.2, proteins=0.9, salt=0.0, sodium=0.005),
    "cucumber":     Nutriments(energy_kcal=15,  fat=0.1, saturated_fat=0.0, carbohydrates=3.6,  sugars=1.7,  fiber=0.5, proteins=0.7, salt=0.0, sodium=0.002),
    "potato":       Nutriments(energy_kcal=77,  fat=0.1, saturated_fat=0.0, carbohydrates=17.5, sugars=0.8,  fiber=2.1, proteins=2.0, salt=0.0, sodium=0.006),
    "sweet potato": Nutriments(energy_kcal=86,  fat=0.1, saturated_fat=0.0, carbohydrates=20.1, sugars=4.2,  fiber=3.0, proteins=1.6, salt=0.0, sodium=0.055),
    "onion":        Nutriments(energy_kcal=40,  fat=0.1, saturated_fat=0.0, carbohydrates=9.3,  sugars=4.2,  fiber=1.7, proteins=1.1, salt=0.0, sodium=0.004),
    "garlic":       Nutriments(energy_kcal=149, fat=0.5, saturated_fat=0.1, carbohydrates=33.1, sugars=1.0,  fiber=2.1, proteins=6.4, salt=0.0, sodium=0.017),
    "pepper":       Nutriments(energy_kcal=20,  fat=0.2, saturated_fat=0.0, carbohydrates=4.6,  sugars=2.4,  fiber=1.7, proteins=0.9, salt=0.0, sodium=0.003),
    "bell pepper":  Nutriments(energy_kcal=20,  fat=0.2, saturated_fat=0.0, carbohydrates=4.6,  sugars=2.4,  fiber=1.7, proteins=0.9, salt=0.0, sodium=0.003),
    "mushroom":     Nutriments(energy_kcal=22,  fat=0.3, saturated_fat=0.0, carbohydrates=3.3,  sugars=2.0,  fiber=1.0, proteins=3.1, salt=0.0, sodium=0.005),
    "celery":       Nutriments(energy_kcal=16,  fat=0.2, saturated_fat=0.0, carbohydrates=3.0,  sugars=1.3,  fiber=1.6, proteins=0.7, salt=0.0, sodium=0.080),
    "corn":         Nutriments(energy_kcal=86,  fat=1.2, saturated_fat=0.2, carbohydrates=19.0, sugars=3.2,  fiber=2.7, proteins=3.3, salt=0.0, sodium=0.015),
    "zucchini":     Nutriments(energy_kcal=17,  fat=0.3, saturated_fat=0.1, carbohydrates=3.1,  sugars=2.5,  fiber=1.0, proteins=1.2, salt=0.0, sodium=0.008),
    "cauliflower":  Nutriments(energy_kcal=25,  fat=0.3, saturated_fat=0.1, carbohydrates=5.0,  sugars=1.9,  fiber=2.0, proteins=1.9, salt=0.0, sodium=0.030),
    "asparagus":    Nutriments(energy_kcal=20,  fat=0.1, saturated_fat=0.0, carbohydrates=3.9,  sugars=1.9,  fiber=2.1, proteins=2.2, salt=0.0, sodium=0.002),
    "green bean":   Nutriments(energy_kcal=31,  fat=0.1, saturated_fat=0.0, carbohydrates=7.0,  sugars=3.3,  fiber=3.4, proteins=1.8, salt=0.0, sodium=0.006),
    "cabbage":      Nutriments(energy_kcal=25,  fat=0.1, saturated_fat=0.0, carbohydrates=5.8,  sugars=3.2,  fiber=2.5, proteins=1.3, salt=0.0, sodium=0.018),
    "kale":         Nutriments(energy_kcal=49,  fat=0.9, saturated_fat=0.1, carbohydrates=8.8,  sugars=2.3,  fiber=3.6, proteins=4.3, salt=0.0, sodium=0.038),
    # ── Asian Groceries ──
    "dumpling":     Nutriments(energy_kcal=206, fat=6.0, saturated_fat=2.0, carbohydrates=28.0, sugars=2.0,  fiber=1.5, proteins=8.0, salt=0.0, sodium=0.450),
    "vermicelli":   Nutriments(energy_kcal=335, fat=0.1, saturated_fat=0.0, carbohydrates=82.3, sugars=0.0,  fiber=0.9, proteins=0.2, salt=0.0, sodium=0.010),
    "bean curd":    Nutriments(energy_kcal=76,  fat=4.8, saturated_fat=0.7, carbohydrates=1.9,  sugars=0.0,  fiber=0.3, proteins=8.1, salt=0.0, sodium=0.007),
    "beancurd":     Nutriments(energy_kcal=76,  fat=4.8, saturated_fat=0.7, carbohydrates=1.9,  sugars=0.0,  fiber=0.3, proteins=8.1, salt=0.0, sodium=0.007),
    "tofu":         Nutriments(energy_kcal=76,  fat=4.8, saturated_fat=0.7, carbohydrates=1.9,  sugars=0.0,  fiber=0.3, proteins=8.1, salt=0.0, sodium=0.007),
    "sesame oil":   Nutriments(energy_kcal=884, fat=100.0,saturated_fat=14.2,carbohydrates=0.0, sugars=0.0,  fiber=0.0, proteins=0.0, salt=0.0, sodium=0.0),
    "ginger":       Nutriments(energy_kcal=80,  fat=0.8, saturated_fat=0.2, carbohydrates=17.8, sugars=1.7,  fiber=2.0, proteins=1.8, salt=0.0, sodium=0.013),
    "green onion":  Nutriments(energy_kcal=32,  fat=0.2, saturated_fat=0.0, carbohydrates=7.3,  sugars=2.3,  fiber=2.6, proteins=1.8, salt=0.0, sodium=0.016),
    "noodle":       Nutriments(energy_kcal=138, fat=2.1, saturated_fat=0.3, carbohydrates=25.2, sugars=0.6,  fiber=1.2, proteins=4.5, salt=0.0, sodium=0.234),
    "starch noodle": Nutriments(energy_kcal=335,fat=0.1, saturated_fat=0.0, carbohydrates=82.3, sugars=0.0,  fiber=0.0, proteins=0.1, salt=0.0, sodium=0.010),
    "rice cake":    Nutriments(energy_kcal=387, fat=0.7, saturated_fat=0.2, carbohydrates=85.5, sugars=0.0,  fiber=1.6, proteins=7.0, salt=0.0, sodium=0.018),
    "kimchi":       Nutriments(energy_kcal=15,  fat=0.5, saturated_fat=0.1, carbohydrates=2.4,  sugars=1.1,  fiber=1.6, proteins=1.1, salt=0.0, sodium=0.498),
    "mayonnaise":   Nutriments(energy_kcal=680, fat=74.9,saturated_fat=12.0,carbohydrates=0.6,  sugars=0.6,  fiber=0.0, proteins=1.0, salt=0.0, sodium=0.635),
    "chili oil":    Nutriments(energy_kcal=884, fat=100.0,saturated_fat=13.5,carbohydrates=0.0, sugars=0.0,  fiber=0.0, proteins=0.0, salt=0.0, sodium=0.0),
    "ice tea":      Nutriments(energy_kcal=33,  fat=0.0, saturated_fat=0.0, carbohydrates=8.0,  sugars=8.0,  fiber=0.0, proteins=0.0, salt=0.0, sodium=0.004),
    "topo chico":   Nutriments(energy_kcal=0,   fat=0.0, saturated_fat=0.0, carbohydrates=0.0,  sugars=0.0,  fiber=0.0, proteins=0.0, salt=0.0, sodium=0.018),
    "lactaid":      Nutriments(energy_kcal=52,  fat=2.0, saturated_fat=1.2, carbohydrates=6.0,  sugars=6.0,  fiber=0.0, proteins=3.3, salt=0.0, sodium=0.050),
    "tropicana":    Nutriments(energy_kcal=45,  fat=0.2, saturated_fat=0.0, carbohydrates=10.4, sugars=8.4,  fiber=0.2, proteins=0.7, salt=0.0, sodium=0.001),
    "ghirardelli":  Nutriments(energy_kcal=500, fat=28.0,saturated_fat=17.0,carbohydrates=60.0, sugars=50.0, fiber=4.0, proteins=5.0, salt=0.0, sodium=0.020),
    "hot pot":      Nutriments(energy_kcal=30,  fat=1.0, saturated_fat=0.2, carbohydrates=4.0,  sugars=2.0,  fiber=0.5, proteins=1.0, salt=0.0, sodium=0.800),
    "soup base":    Nutriments(energy_kcal=30,  fat=1.0, saturated_fat=0.2, carbohydrates=4.0,  sugars=2.0,  fiber=0.5, proteins=1.0, salt=0.0, sodium=0.800),
    "napa":         Nutriments(energy_kcal=16,  fat=0.2, saturated_fat=0.0, carbohydrates=3.2,  sugars=1.4,  fiber=1.2, proteins=1.2, salt=0.0, sodium=0.009),
    "sujebi":       Nutriments(energy_kcal=110, fat=1.0, saturated_fat=0.2, carbohydrates=22.0, sugars=1.0,  fiber=1.0, proteins=3.5, salt=0.0, sodium=0.350),
    "quaker":       Nutriments(energy_kcal=389, fat=6.9, saturated_fat=1.2, carbohydrates=66.3, sugars=0.0,  fiber=10.6,proteins=16.9,salt=0.0, sodium=0.002),
    "foster farm":  Nutriments(energy_kcal=120, fat=3.0, saturated_fat=0.8, carbohydrates=0.0,  sugars=0.0,  fiber=0.0, proteins=21.0,salt=0.0, sodium=0.340),
    # ── Staples / Protein ──
    "egg":          Nutriments(energy_kcal=155, fat=11.0,saturated_fat=3.3, carbohydrates=1.1,  sugars=1.1,  fiber=0.0, proteins=13.0,salt=0.0, sodium=0.124),
    "chicken breast":Nutriments(energy_kcal=165,fat=3.6, saturated_fat=1.0, carbohydrates=0.0,  sugars=0.0,  fiber=0.0, proteins=31.0,salt=0.0, sodium=0.074),
    "chicken":      Nutriments(energy_kcal=239, fat=13.6,saturated_fat=3.8, carbohydrates=0.0,  sugars=0.0,  fiber=0.0, proteins=27.3,salt=0.0, sodium=0.082),
    "ground beef":  Nutriments(energy_kcal=254, fat=20.0,saturated_fat=7.7, carbohydrates=0.0,  sugars=0.0,  fiber=0.0, proteins=17.2,salt=0.0, sodium=0.075),
    "salmon":       Nutriments(energy_kcal=208, fat=13.4,saturated_fat=3.1, carbohydrates=0.0,  sugars=0.0,  fiber=0.0, proteins=20.4,salt=0.0, sodium=0.059),
    "shrimp":       Nutriments(energy_kcal=99,  fat=0.3, saturated_fat=0.1, carbohydrates=0.2,  sugars=0.0,  fiber=0.0, proteins=24.0,salt=0.0, sodium=0.111),
    "pork":         Nutriments(energy_kcal=242, fat=14.0,saturated_fat=5.2, carbohydrates=0.0,  sugars=0.0,  fiber=0.0, proteins=27.3,salt=0.0, sodium=0.062),
    "turkey":       Nutriments(energy_kcal=189, fat=7.4, saturated_fat=2.0, carbohydrates=0.0,  sugars=0.0,  fiber=0.0, proteins=29.3,salt=0.0, sodium=0.068),
    "bacon":        Nutriments(energy_kcal=541, fat=42.0,saturated_fat=14.0,carbohydrates=1.4,  sugars=0.0,  fiber=0.0, proteins=37.0,salt=0.0, sodium=1.717),
    "ham":          Nutriments(energy_kcal=145, fat=5.5, saturated_fat=1.8, carbohydrates=1.5,  sugars=0.0,  fiber=0.0, proteins=21.8,salt=0.0, sodium=1.203),
    "hot dog":      Nutriments(energy_kcal=290, fat=26.0,saturated_fat=9.6, carbohydrates=2.0,  sugars=0.0,  fiber=0.0, proteins=10.3,salt=0.0, sodium=1.090),
    "sausage":      Nutriments(energy_kcal=301, fat=24.0,saturated_fat=8.5, carbohydrates=2.4,  sugars=0.0,  fiber=0.0, proteins=19.0,salt=0.0, sodium=0.870),
    # ── Dairy ──
    "milk":         Nutriments(energy_kcal=61,  fat=3.3, saturated_fat=1.9, carbohydrates=4.8,  sugars=5.1,  fiber=0.0, proteins=3.2, salt=0.0, sodium=0.043),
    "butter":       Nutriments(energy_kcal=717, fat=81.1,saturated_fat=51.4,carbohydrates=0.1,  sugars=0.1,  fiber=0.0, proteins=0.9, salt=0.0, sodium=0.643),
    "yogurt":       Nutriments(energy_kcal=59,  fat=0.4, saturated_fat=0.1, carbohydrates=3.6,  sugars=3.2,  fiber=0.0, proteins=10.0,salt=0.0, sodium=0.036),
    "cheddar":      Nutriments(energy_kcal=403, fat=33.1,saturated_fat=21.1,carbohydrates=1.3,  sugars=0.5,  fiber=0.0, proteins=24.9,salt=0.0, sodium=0.621),
    "mozzarella":   Nutriments(energy_kcal=280, fat=17.1,saturated_fat=10.9,carbohydrates=3.1,  sugars=1.0,  fiber=0.0, proteins=27.5,salt=0.0, sodium=0.627),
    "cream cheese": Nutriments(energy_kcal=342, fat=34.2,saturated_fat=19.2,carbohydrates=4.1,  sugars=3.2,  fiber=0.0, proteins=5.9, salt=0.0, sodium=0.321),
    # ── Grains ──
    "rice":         Nutriments(energy_kcal=130, fat=0.3, saturated_fat=0.1, carbohydrates=28.2, sugars=0.0,  fiber=0.4, proteins=2.7, salt=0.0, sodium=0.001),
    "bread":        Nutriments(energy_kcal=265, fat=3.2, saturated_fat=0.7, carbohydrates=49.0, sugars=5.0,  fiber=2.7, proteins=9.4, salt=0.0, sodium=0.491),
    "pasta":        Nutriments(energy_kcal=131, fat=1.1, saturated_fat=0.2, carbohydrates=25.0, sugars=0.6,  fiber=1.8, proteins=5.0, salt=0.0, sodium=0.001),
}


def _lookup_usda_fallback(product_name: str, usda_db: "UsdaDatabase | None" = None) -> Nutriments | None:
    """Try to match product name against USDA database, fall back to hardcoded table."""
    if usda_db is not None:
        result = usda_db.lookup(product_name)
        if result and result.has_data() and not result.is_all_zero():
            return result
    name_lower = product_name.lower()
    for keyword in sorted(_USDA_NUTRITION.keys(), key=len, reverse=True):
        if keyword in name_lower:
            return _USDA_NUTRITION[keyword]
    return None


def _finalize_weight_and_nutrition(item: Item, usda_db: "UsdaDatabase | None" = None):
    """
    After weight and nutriments_per100g are set:
    1. Filter all-zero bad OFF data
    2. Fall back to USDA produce table if no nutrition data
    3. Convert weight to grams (× qty for total)
    4. Scale nutriments from per-100g to actual total weight
    """
    if item.weight:
        single_grams = item.weight.grams
        total_grams = single_grams * (item.qty or 1)
        item.weight_grams = round(total_grams, 2)

        # Filter all-zero bad data from OFF
        if item.nutriments_per100g and item.nutriments_per100g.is_all_zero():
            item.nutriments_per100g = None

        # USDA fallback if no nutrition data
        if not item.nutriments_per100g or not item.nutriments_per100g.has_data():
            usda = _lookup_usda_fallback(item.resolved_name or item.name, usda_db=usda_db)
            if usda:
                item.nutriments_per100g = usda

        if item.nutriments_per100g and item.nutriments_per100g.has_data():
            item.nutriments_actual = item.nutriments_per100g.scale_to_weight(total_grams)


# ═══════════════════════ Pipeline ═══════════════════════

def process_image(image_path: str, ocr: OCREngine, off_db: OffDatabase,
                  model_data: dict | None = None, usda_db: "UsdaDatabase | None" = None):
    """Full pipeline: image -> OCR -> text -> ML/rule parse -> weights + nutrition -> save."""
    path = Path(image_path)
    if not path.exists():
        path = Path(CONFIG["image_dir"]) / image_path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Image  : {path}")
    if model_data is not None:
        print("Parser : ML classifier")
    else:
        print("Parser : rule-based (no model found)")

    # 1. OCR
    print("\n===== Running PaddleOCR =====")
    fragments = ocr.run(str(path))
    print(f"Detected {len(fragments)} text regions")

    # 2. Text reconstruction
    # Prefer Google Vision's full text (better line breaks for receipts)
    google_text = ocr.get_full_text() if hasattr(ocr, 'get_full_text') else None
    if google_text:
        raw_text = google_text
        print(f"\n===== Google Vision Full Text =====")
    else:
        raw_text = rebuild_text(fragments)
        print(f"\n===== Reconstructed Text =====")
    print(raw_text)

    # 3. Parse
    receipt = parse_receipt_text(raw_text, model_data=model_data)

    # 4. Name resolution report
    print(f"\n===== Name Resolution ({len(receipt.items)} items) =====")
    for item in receipt.items:
        if item.name != item.resolved_name:
            print(f"  {item.name:<40} -> {item.resolved_name}")
        else:
            print(f"  {item.name} (unchanged)")

    # 5. Weight + Nutrition resolution
    print(f"\n===== Weight & Nutrition Resolution =====")
    for item in receipt.items:
        resolve_weight(item, receipt.store_name, off_db, usda_db=usda_db)

    # Print as aligned table
    header = (f"  {'Name':<35} {'Qty':>3} {'Weight(g)':>10} {'Source':<25} "
              f"{'kcal':>8} {'Prot(g)':>8} {'Fat(g)':>8} {'Carb(g)':>8}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for item in receipt.items:
        name = (item.resolved_name or item.name)[:35]
        qty = item.qty or 1
        wg = f"{item.weight_grams:.1f}" if item.weight_grams else "—"
        src = f"[{item.weight_source}]" if item.weight_source else "[—]"
        na = item.nutriments_actual
        kcal = f"{na.energy_kcal:.1f}" if na and na.energy_kcal is not None else "—"
        prot = f"{na.proteins:.1f}" if na and na.proteins is not None else "—"
        fat  = f"{na.fat:.1f}" if na and na.fat is not None else "—"
        carb = f"{na.carbohydrates:.1f}" if na and na.carbohydrates is not None else "—"
        print(f"  {name:<35} {qty:>3} {wg:>10} {src:<25} {kcal:>8} {prot:>8} {fat:>8} {carb:>8}")

    # 6. Return receipt (saving handled by run_full_pipeline.py)
    return receipt


# ═══════════════════════ Entry Point ═══════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_receipt.py <image> [image2 ...]")
        sys.exit(1)

    ocr = OCREngine()
    off_db = OffDatabase(CONFIG["off_db_path"])
    usda_db = UsdaDatabase(CONFIG["usda_db_path"])
    if Path(CONFIG["usda_db_path"]).exists():
        print(f"[USDA] Loaded: {CONFIG['usda_db_path']}")
    else:
        print("[USDA] Database not found — using hardcoded fallback only")
        usda_db = None

    model_data = _load_ml_classifier()
    if model_data:
        print(f"[ML] Loaded classifier: {list(model_data['label_encoder'].classes_)}")
    else:
        print("[ML] Model not found — using rule-based parser")

    try:
        for img_path in sys.argv[1:]:
            process_image(img_path, ocr, off_db, model_data=model_data, usda_db=usda_db)
            if len(sys.argv) > 2:
                print("\n" + "=" * 60 + "\n")
    finally:
        off_db.close()
        if usda_db:
            usda_db.close()


if __name__ == "__main__":
    main()