"""
run_full_pipeline.py — Complete pipeline: receipt images -> nutrition report
=============================================================================
User-based storage: each person's data lives under users/<name>/

Structure:
  users/<name>/
    profile.json          User info (name, created date, receipt count)
    receipt_info/          Per-receipt JSON/CSV, named by date (YYYY-MM-DD)
    adapted/              Converted JSONs for SP pipeline
    reports/
      figures/
        1_smoothing/      Moving average plots
        2_detection/      DRI threshold plots
        3_fft_report/     FFT + nutrition report
      csv_output/         daily_nutrition.csv, smoothed, etc.

Usage:
  python pipeline/run_full_pipeline.py --user suhyun --all
  python pipeline/run_full_pipeline.py --user suhyun img1.jpg img2.jpg
  python pipeline/run_full_pipeline.py --user suhyun --append img1.jpg img2.jpg
  python pipeline/run_full_pipeline.py --user suhyun --sp-only
"""

from __future__ import annotations

import sys
import os
import json
import glob
import shutil
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Resolve paths
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
SP_SCRIPTS_DIR = PROJECT_ROOT / "SP_Pipeline"
USERS_DIR = PROJECT_ROOT / "users"


# =============================== User Profile ===============================

@dataclass
class UserProfile:
    name: str
    created: str = ""
    receipt_count: int = 0
    last_run: str = ""

    # Derived paths
    @property
    def root(self) -> Path:
        return USERS_DIR / self.name

    @property
    def receipt_info_dir(self) -> Path:
        return self.root / "receipt_info"

    @property
    def adapted_dir(self) -> Path:
        return self.root / "adapted"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def csv_output_dir(self) -> Path:
        return self.reports_dir / "csv_output"

    @property
    def profile_path(self) -> Path:
        return self.root / "profile.json"

    def ensure_dirs(self):
        for d in [self.receipt_info_dir, self.adapted_dir,
                  self.figures_dir / "1_smoothing",
                  self.figures_dir / "2_detection",
                  self.figures_dir / "3_fft_report",
                  self.csv_output_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save(self):
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )

    @classmethod
    def load_or_create(cls, name: str) -> "UserProfile":
        profile_path = USERS_DIR / name / "profile.json"
        if profile_path.exists():
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            profile = cls(**data)
            print(f"  [USER] Loaded profile: {name} ({profile.receipt_count} receipts)")
        else:
            profile = cls(
                name=name,
                created=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
            print(f"  [USER] Created new profile: {name}")
        profile.ensure_dirs()
        return profile


# =============================== Date Utilities ===============================

def parse_date_to_ymd(date_str: str) -> str | None:
    """Parse various date formats to YYYY-MM-DD string."""
    formats = [
        "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y",
        "%Y-%m-%d", "%b %d, %Y", "%b %d %Y", "%B %d, %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip().rstrip(","), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def make_receipt_filename(receipt_data: dict, user: UserProfile) -> str:
    """
    Generate filename based on receipt date.
    Uses disk state (not process state) to avoid counter issues.
    """
    date_str = receipt_data.get("date", "")
    if date_str:
        parsed = parse_date_to_ymd(date_str)
        if parsed:
            base = parsed
            store = receipt_data.get("store_name", "").strip().lower()
            store = store.replace(" ", "_").replace("'", "").replace("&", "and")

            # Check for duplicate — append store name
            if (user.receipt_info_dir / f"{base}.json").exists() and store:
                base = f"{parsed}_{store}"

            # If still duplicate — add numeric suffix
            final = base
            counter = 1
            while (user.receipt_info_dir / f"{final}.json").exists():
                counter += 1
                final = f"{base}_{counter}"
            return final

    # No date: scan disk for highest missing_date number
    existing = sorted(user.receipt_info_dir.glob("missing_date_*.json"))
    if existing:
        last_num = int(existing[-1].stem.split("_")[-1])
        return f"missing_date_{last_num + 1:03d}"
    return "missing_date_001"


def get_output_prefix(adapted_paths: list[Path]) -> str:
    """
    Generate output file prefix: YYYYMMDD (today's date).
    This ensures each run produces uniquely named outputs.
    """
    return datetime.now().strftime("%Y%m%d")


def get_receipt_date_range(adapted_paths: list[Path]) -> str:
    """
    Get receipt date range from adapted paths for display purposes.
    Returns '2026-01-23 to 2026-03-07' or 'undated'.
    """
    dates = []
    for jp in adapted_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            date_str = data.get("date", "")
            if date_str:
                parsed = parse_date_to_ymd(date_str)
                if parsed:
                    dates.append(parsed)
        except Exception:
            continue

    if not dates:
        return "undated"

    dates.sort()
    return f"{dates[0]} to {dates[-1]}"


# =============================== Pipeline Results ===============================

@dataclass
class PipelineResult:
    """Track success/failure for each image."""
    image: str
    success: bool
    base_name: str = ""
    date: str = ""
    store: str = ""
    n_items: int = 0
    error: str = ""
    stage: str = ""


# =============================== Stage 1: OCR + Nutrition ===============================

def run_receipt_pipeline(image_paths: list[str], user: UserProfile) -> list[PipelineResult]:
    """Run run_receipt.py on each image, save to user's receipt_info/."""
    sys.path.insert(0, str(_SCRIPT_DIR))
    from run_receipt import OCREngine, OffDatabase, UsdaDatabase, process_image, CONFIG

    ocr = OCREngine()
    off_db = OffDatabase(CONFIG["off_db_path"])

    usda_db = None
    usda_path = CONFIG.get("usda_db_path", "")
    if usda_path and Path(usda_path).exists():
        usda_db = UsdaDatabase(usda_path)
        print(f"  [USDA] Loaded: {usda_path}")

    from run_receipt import _load_ml_classifier
    model_data = _load_ml_classifier()
    if model_data:
        print(f"  [ML] Loaded classifier: {list(model_data['label_encoder'].classes_)}")

    results = []
    try:
        for img_path in image_paths:
            try:
                receipt = process_image(img_path, ocr, off_db,
                                        model_data=model_data, usda_db=usda_db)
                receipt_dict = receipt.to_dict()

                # Generate date-based filename
                base_name = make_receipt_filename(receipt_dict, user)
                json_path = user.receipt_info_dir / f"{base_name}.json"
                csv_path = user.receipt_info_dir / f"{base_name}.csv"

                # Save
                json_path.write_text(
                    json.dumps(receipt_dict, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                receipt.save_csv(str(csv_path))

                n_items = len(receipt_dict.get("items", []))
                date_info = receipt_dict.get("date", "")

                results.append(PipelineResult(
                    image=str(img_path), success=True,
                    base_name=base_name, date=date_info,
                    store=receipt_dict.get("store_name", ""),
                    n_items=n_items, stage="ocr",
                ))
                print(f"  [OK] {Path(img_path).name} -> {base_name}.json "
                      f"({n_items} items, date={date_info or 'none'})")

            except Exception as e:
                results.append(PipelineResult(
                    image=str(img_path), success=False,
                    error=str(e), stage="ocr",
                ))
                print(f"  [ERROR] {Path(img_path).name}: {e}")
                traceback.print_exc()
    finally:
        off_db.close()
        if usda_db:
            usda_db.close()

    return results


# =============================== Stage 2: Adapt JSON ===============================

def adapt_receipt_json(receipt_data: dict) -> dict | None:
    """Convert run_receipt.py output to format expected by 0_aggregate_receipts.py."""
    adapted_items = []
    for item in receipt_data.get("items", []):
        na = item.get("nutriments_actual") or {}

        # Stricter matching: need at least energy + one macro
        has_energy = na.get("energy_kcal") is not None and na.get("energy_kcal") != ""
        has_macro = any(
            na.get(k) is not None and na.get(k) != ""
            for k in ["proteins", "fat", "carbohydrates"]
        )
        has_nutrition = has_energy and has_macro

        nutrition = {
            "calories": na.get("energy_kcal", 0) or 0,
            "protein_g": na.get("proteins", 0) or 0,
            "carbs_g": na.get("carbohydrates", 0) or 0,
            "fat_g": na.get("fat", 0) or 0,
            "fiber_g": na.get("fiber", 0) or 0,
        }

        adapted_items.append({
            "original_name": item.get("name", ""),
            "matched_name": item.get("resolved_name", item.get("name", "")),
            "portion_g": item.get("weight_grams") or 0,
            "weight_source": item.get("weight_source", ""),
            "nutrition": nutrition,
            "status": "matched" if has_nutrition else "unmatched",
            "price": item.get("line_price"),
        })

    matched_items = [it for it in adapted_items if it["status"] == "matched"]
    totals = {
        "calories": sum(it["nutrition"]["calories"] for it in matched_items),
        "protein_g": sum(it["nutrition"]["protein_g"] for it in matched_items),
        "carbs_g": sum(it["nutrition"]["carbs_g"] for it in matched_items),
        "fat_g": sum(it["nutrition"]["fat_g"] for it in matched_items),
        "fiber_g": sum(it["nutrition"]["fiber_g"] for it in matched_items),
    }

    return {
        "store_name": receipt_data.get("store_name", ""),
        "date": receipt_data.get("date", ""),
        "total": receipt_data.get("total", ""),
        "items": adapted_items,
        "totals": totals,
    }


def adapt_and_save(pipeline_results: list[PipelineResult], user: UserProfile) -> list[Path]:
    """Adapt successful receipt results and save to user's adapted/."""
    adapted_paths = []
    for r in pipeline_results:
        if not r.success:
            continue

        json_path = user.receipt_info_dir / f"{r.base_name}.json"
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        adapted = adapt_receipt_json(data)
        if adapted and adapted["items"]:
            out_path = user.adapted_dir / f"{r.base_name}.adapted.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(adapted, f, indent=2, ensure_ascii=False)
            n_matched = sum(1 for it in adapted["items"] if it["status"] == "matched")
            print(f"  [ADAPT] {r.base_name}: {n_matched}/{len(adapted['items'])} matched")
            adapted_paths.append(out_path)
        else:
            print(f"  [SKIP] {r.base_name}: no items")

    return adapted_paths


def adapt_all_existing(user: UserProfile) -> list[Path]:
    """Adapt all receipt_info/ JSONs that aren't yet in adapted/."""
    adapted_paths = []
    for jp in sorted(user.receipt_info_dir.glob("*.json")):
        adapted_path = user.adapted_dir / f"{jp.stem}.adapted.json"
        if adapted_path.exists():
            adapted_paths.append(adapted_path)
            continue

        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            adapted = adapt_receipt_json(data)
            if adapted and adapted["items"]:
                with open(adapted_path, "w", encoding="utf-8") as f:
                    json.dump(adapted, f, indent=2, ensure_ascii=False)
                adapted_paths.append(adapted_path)
        except Exception as e:
            print(f"  [WARN] Could not adapt {jp.name}: {e}")

    return adapted_paths


# =============================== Stage 3-5: SP Pipeline ===============================

def run_sp_pipeline(adapted_paths: list[Path], user: UserProfile):
    """Run signal processing pipeline scripts in sequence."""
    sp_scripts = [
        ("0_aggregate_receipts.py", "Aggregating receipts into daily signals"),
        ("1_sp_moving_average.py", "Moving average filtering"),
        ("2_sp_threshold_detection.py", "DRI threshold detection"),
        ("3_sp_fft_and_report.py", "FFT analysis & report generation"),
    ]

    # Use a temp working directory for SP scripts
    import tempfile
    work_dir = Path(tempfile.mkdtemp(prefix="sp_pipeline_"))
    original_dir = os.getcwd()
    os.chdir(str(work_dir))

    try:
        # Step 0: aggregate
        script_path = SP_SCRIPTS_DIR / sp_scripts[0][0]
        if not script_path.exists():
            print(f"\n  [ERROR] {sp_scripts[0][0]} not found in {SP_SCRIPTS_DIR}")
            return

        print(f"\n{'='*60}")
        print(f"  STAGE 3: {sp_scripts[0][1]}")
        print(f"{'='*60}")

        adapted_str = [str(p) for p in adapted_paths]
        result = subprocess.run(
            [sys.executable, str(script_path)] + adapted_str,
            capture_output=False
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  [ERROR] {sp_scripts[0][0]} failed:")
            print(f"  {result.stderr[:500]}")
            return

        # Steps 1-3
        for script_name, description in sp_scripts[1:]:
            script_path = SP_SCRIPTS_DIR / script_name
            if not script_path.exists():
                print(f"\n  [WARN] {script_name} not found, skipping")
                continue

            print(f"\n{'='*60}")
            print(f"  STAGE: {description}")
            print(f"{'='*60}")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"  [ERROR] {script_name} failed:")
                print(f"  {result.stderr[:500]}")
                return

        # Organize outputs into user's reports directory
        organize_outputs(work_dir, adapted_paths, user)

    finally:
        os.chdir(original_dir)
        # Clean up temp dir
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


def organize_outputs(work_dir: Path, adapted_paths: list[Path], user: UserProfile):
    """Move SP pipeline outputs into user's organized report directories."""
    date_prefix = get_output_prefix(adapted_paths)
    moved = 0

    # PNG classification
    fig_categories = {
        "1_smoothing": [
            "fig_moving_average_calories.png",
            "fig_all_nutrients_smoothed.png",
            "fig_window_size_ablation.png",
        ],
        "2_detection": [
            "fig_threshold_detection.png",
            "fig_raw_vs_filtered_detection.png",
            "fig_deficiency_summary.png",
        ],
        "3_fft_report": [
            "fig_fft_purchase_patterns.png",
            "fig_fft_regularity.png",
            "fig_nutrition_report_full.png",
        ],
    }

    file_to_dest = {}
    for subdir, files in fig_categories.items():
        for fname in files:
            short_name = fname.replace("fig_", "")
            new_name = f"{date_prefix}_{short_name}"
            file_to_dest[fname] = (subdir, new_name)

    # Move PNGs
    for f in work_dir.glob("*.png"):
        if f.name in file_to_dest:
            subdir, new_name = file_to_dest[f.name]
        else:
            subdir = "other"
            new_name = f"{date_prefix}_{f.name.replace('fig_', '')}"

        dest_dir = user.figures_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / new_name

        # Handle existing files: overwrite
        if dest.exists():
            dest.unlink()
        shutil.move(str(f), str(dest))
        moved += 1

    # Move CSVs
    for name in ["daily_nutrition.csv", "smoothed_nutrition.csv",
                 "threshold_detection_results.csv"]:
        f = work_dir / name
        if f.exists():
            dest = user.csv_output_dir / f"{date_prefix}_{name}"
            if dest.exists():
                dest.unlink()
            shutil.move(str(f), str(dest))
            moved += 1

    # Move purchase_history.json
    f = work_dir / "purchase_history.json"
    if f.exists():
        dest = user.csv_output_dir / f"{date_prefix}_purchase_history.json"
        if dest.exists():
            dest.unlink()
        shutil.move(str(f), str(dest))
        moved += 1

    if moved:
        print(f"\n  [CLEANUP] Organized {moved} files with prefix '{date_prefix}'")


# =============================== Summary ===============================

def print_summary(results: list[PipelineResult], user: UserProfile):
    """Print structured success/failure summary."""
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY for {user.name}")
    print(f"{'='*60}")
    print(f"  Processed: {len(successes)}/{len(results)} images")

    if successes:
        print(f"\n  Successful:")
        for r in successes:
            print(f"    [OK] {r.base_name} | {r.store or '?'} | {r.date or 'no date'} | {r.n_items} items")

    if failures:
        print(f"\n  Failed:")
        for r in failures:
            print(f"    [FAIL] {Path(r.image).name} | stage={r.stage} | {r.error[:80]}")

    # Print report files
    date_prefix = get_output_prefix(sorted(user.adapted_dir.glob("*.json")))
    receipt_range = get_receipt_date_range(sorted(user.adapted_dir.glob("*.json")))
    print(f"\n  Generated:     {date_prefix}")
    print(f"  Receipt range: {receipt_range}")
    print(f"  User directory: {user.root}")

    print(f"\n  Reports:")
    for subdir in ["1_smoothing", "2_detection", "3_fft_report"]:
        sd = user.figures_dir / subdir
        if sd.exists():
            files = sorted(sd.glob("*.png"))
            if files:
                print(f"    {subdir}/")
                for fp in files:
                    print(f"      [OK] {fp.name}")


# =============================== Main ===============================

def main():
    print("=" * 60)
    print("  RECEIPT NUTRITION ANALYSIS PIPELINE")
    print("  Image -> OCR -> Nutrition -> Signal Processing -> Report")
    print("=" * 60)

    # Parse --user argument
    if "--user" not in sys.argv:
        print(f"\nUsage:")
        print(f"  python {sys.argv[0]} --user <name> --all")
        print(f"  python {sys.argv[0]} --user <name> img1.jpg img2.jpg")
        print(f"  python {sys.argv[0]} --user <name> --append img1.jpg ...")
        print(f"  python {sys.argv[0]} --user <name> --sp-only")
        sys.exit(1)

    user_idx = sys.argv.index("--user")
    if user_idx + 1 >= len(sys.argv):
        print("[ERROR] --user requires a name")
        sys.exit(1)

    user_name = sys.argv[user_idx + 1]
    remaining_args = [a for i, a in enumerate(sys.argv[1:], 1)
                      if i != user_idx and i != user_idx + 1]

    if not remaining_args:
        print("[ERROR] No mode specified. Use --all, --append, --sp-only, or image paths.")
        sys.exit(1)

    # Load or create user profile
    user = UserProfile.load_or_create(user_name)
    mode = remaining_args[0]

    # ── --sp-only ──
    if mode == "--sp-only":
        print(f"\n  [SP-ONLY] Rerunning analysis for {user.name}...")
        adapted_paths = sorted(user.adapted_dir.glob("*.json"))

        if not adapted_paths:
            print(f"  No adapted JSONs found, adapting from receipt_info/...")
            adapted_paths = adapt_all_existing(user)

        if not adapted_paths:
            print("  [ERROR] No data found. Run full pipeline first.")
            sys.exit(1)

        print(f"  Using {len(adapted_paths)} receipts")
        run_sp_pipeline(adapted_paths, user)

        # Update profile
        user.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")
        user.save()
        return

    # ── --append ──
    if mode == "--append":
        image_paths = remaining_args[1:]
        if not image_paths:
            print("[ERROR] --append requires image paths")
            sys.exit(1)

        existing = sorted(user.adapted_dir.glob("*.json"))
        print(f"\n  User: {user.name}")
        print(f"  Existing receipts: {len(existing)}")
        print(f"  New images: {len(image_paths)}")

        # Stage 1
        print(f"\n{'='*60}")
        print(f"  STAGE 1: Processing {len(image_paths)} NEW receipt(s)")
        print(f"{'='*60}")
        results = run_receipt_pipeline(image_paths, user)

        # Stage 2
        successes = [r for r in results if r.success]
        if successes:
            print(f"\n{'='*60}")
            print(f"  STAGE 2: Adapting new receipts")
            print(f"{'='*60}")
            adapt_and_save(results, user)

        # Collect ALL adapted JSONs
        all_adapted = sorted(user.adapted_dir.glob("*.json"))
        print(f"\n  Total receipts for analysis: {len(all_adapted)}")

        # Stage 3-5
        run_sp_pipeline(all_adapted, user)

        # Update profile
        user.receipt_count = len(list(user.receipt_info_dir.glob("*.json")))
        user.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")
        user.save()

        print_summary(results, user)
        return

    # ── --all ──
    if mode == "--all":
        img_dir = PROJECT_ROOT / "data_Picture"
        image_paths = sorted(
            glob.glob(str(img_dir / "*.jpg")) +
            glob.glob(str(img_dir / "*.jpeg")) +
            glob.glob(str(img_dir / "*.png"))
        )
        if not image_paths:
            print(f"No images found in {img_dir}")
            sys.exit(1)
    else:
        # Specific image paths
        image_paths = remaining_args

    print(f"\n  User: {user.name}")
    print(f"  Images: {len(image_paths)}")
    for p in image_paths[:10]:
        print(f"    {Path(p).name}")
    if len(image_paths) > 10:
        print(f"    ... and {len(image_paths) - 10} more")

    # Stage 1
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Receipt OCR + Nutrition Lookup")
    print(f"{'='*60}")
    results = run_receipt_pipeline(image_paths, user)

    successes = [r for r in results if r.success]
    print(f"\n  Processed: {len(successes)}/{len(results)}")

    if not successes:
        print("[ERROR] No receipts processed")
        sys.exit(1)

    # Stage 2
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Adapting JSON format for SP pipeline")
    print(f"{'='*60}")
    adapted_paths = adapt_and_save(results, user)
    print(f"\n  Adapted: {len(adapted_paths)} receipts")

    if not adapted_paths:
        print("[ERROR] No receipts adapted")
        sys.exit(1)

    # Stage 3-5
    run_sp_pipeline(adapted_paths, user)

    # Update profile
    user.receipt_count = len(list(user.receipt_info_dir.glob("*.json")))
    user.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")
    user.save()

    print_summary(results, user)


if __name__ == "__main__":
    main()