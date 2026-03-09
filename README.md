# NutriLens - Smart Receipt Analyzer

A signal-processing pipeline that extracts nutritional data from grocery receipt images and generates health reports using classical DSP techniques.

## Overview

This project turns photos of grocery receipts into actionable nutrition insights. It combines OCR, machine learning, and signal processing to analyze purchasing patterns and nutritional intake over time.

**Pipeline:**
```
Receipt Image → OCR → Item Extraction → Nutrition Lookup → Signal Processing → Report
```

## Features

- **Multi-store receipt OCR** via Google Vision API with PaddleOCR fallback
- **ML line classifier** (Gradient Boosting, 84.6% accuracy) for item/noise/header/date classification
- **Dual nutrition database**: Open Food Facts (310K+ products) + USDA SR Legacy (7,800 foods)
- **Automatic store detection**: Target, H Mart, Uwajimaya, Trader Joe's, T&T, Whole Foods, and 20+ more
- **User profile system** with per-user receipt history and reports
- **9 auto-generated visualizations** including a comprehensive monthly nutrition report

## Project Structure

```
OFFAPI_test/
├── pipeline/
│   ├── run_receipt.py          # OCR + parsing + nutrition lookup
│   ├── run_full_pipeline.py    # Orchestrator with user profiles
│   └── annotate_lines.py       # Training data annotation tool
├── SP_Pipeline/
│   ├── 0_aggregate_receipts.py # Receipt → daily signal conversion
│   ├── 1_sp_moving_average.py  # Step 2: MA filter (convolution)
│   ├── 2_sp_threshold_detection.py  # Step 3: DRI detection
│   └── 3_sp_fft_and_report.py  # Step 4: FFT + report generation
├── data/
│   └── db/
│       ├── off_products.db     # Open Food Facts (313MB)
│       └── usda_sr.db          # USDA SR Legacy (2.6MB)
├── models/
│   └── line_classifier.pkl     # Trained ML classifier
├── users/                      # Per-user data & reports
│   └── <username>/
│       ├── profile.json
│       ├── receipt_info/       # Extracted receipt JSONs
│       ├── adapted/            # SP-format JSONs
│       └── reports/
│           ├── figures/
│           │   ├── 1_smoothing/
│           │   ├── 2_detection/
│           │   └── 3_fft_report/
│           └── csv_output/
├── data_Picture/               # Receipt images
├── annotations/                # ML training data
└── scripts/                    # Database build scripts
```

## Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud Vision API key (optional, falls back to PaddleOCR)

### Installation

```bash
pip install paddleocr paddlepaddle numpy pandas matplotlib scikit-learn google-cloud-vision
```

### Usage

**Process all receipts for a user:**
```bash
python pipeline/run_full_pipeline.py --user <name> --all
```

**Append new receipts:**
```bash
python pipeline/run_full_pipeline.py --user <name> --append receipt1.jpg receipt2.jpg
```

**Rerun SP analysis only (no OCR):**
```bash
python pipeline/run_full_pipeline.py --user <name> --sp-only
```

**Process a single image:**
```bash
python pipeline/run_receipt.py image.jpg
```

## Sample Output

The pipeline generates a comprehensive monthly nutrition report with 5 panels:

1. **Macronutrient Breakdown** — Pie chart of carbs/protein/fat ratios
2. **DRI Comparison** — How your intake compares to recommended levels
3. **Category Purchases** — What food groups you buy most
4. **Calorie Intake Trend** — Weekly calorie trend vs. healthy range (1600-2000 kcal)
5. **Health Insights** — Personalized recommendations based on FFT and threshold analysis

## Technical Details

### OCR Strategy

Google Vision `text_detection` provides full-text output with natural line breaks optimized for receipt layouts. Word-level fragments are also retained for potential geometric analysis.

### Nutrition Lookup Chain

```
Item name → OFF exact match (310K products)
         → OFF fuzzy match
         → USDA SR Legacy DB (7,800 foods)
         → Category defaults
```

### DRI Ranges

| Nutrient | Low | Target | High | Unit |
|----------|-----|--------|------|------|
| Calories | 1600 | 1800 | 2000 | kcal |
| Protein | 46 | 50 | 56 | g |
| Carbs | 225 | 275 | 325 | g |
| Fat | 65 | 78 | 97 | g |
| Fiber | 21 | 25 | 30 | g |

### Supported Stores

Target, H Mart, Uwajimaya, T&T Supermarket, Trader Joe's, Whole Foods, Asian Family Market, Costco, Walmart, Safeway, QFC, Fred Meyer, Albertsons, 99 Ranch Market, and more.

## Data Sources

- **[Open Food Facts](https://world.openfoodfacts.org/)** — Open food product database (ODbL license)
- **[USDA FoodData Central](https://fdc.nal.usda.gov/)** — SR Legacy dataset (public domain)

## License

MIT License
