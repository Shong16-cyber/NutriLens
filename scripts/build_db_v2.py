"""
build_db_v2.py — 从 Open Food Facts Parquet 构建 SQLite 数据库（含营养数据）
用法：
  1. pip install pandas pyarrow
  2. 确保 food.parquet 在当前目录
  3. python build_db_v2.py
  4. 生成 off_products.db
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np

PARQUET_FILE = "food.parquet"
SQLITE_FILE = "off_products.db"

COLUMNS = [
    "code", "product_name", "brands", "quantity",
    "categories_tags", "nutriments",
]

# 我们要从 nutriments 数组中提取的字段（per 100g）
NUTRIENT_NAMES = [
    "energy-kcal",
    "fat",
    "saturated-fat",
    "carbohydrates",
    "sugars",
    "fiber",
    "proteins",
    "salt",
    "sodium",
]


def extract_en_name(product_name_list):
    """从 list<struct<lang, text>> 中提取英文名"""
    if product_name_list is None:
        return None
    try:
        if len(product_name_list) == 0:
            return None
    except TypeError:
        return None
    for item in product_name_list:
        if isinstance(item, dict) and item.get("lang") == "en":
            return item.get("text", "").strip()
    for item in product_name_list:
        if isinstance(item, dict):
            text = item.get("text", "").strip()
            if text:
                return text
    return None


def extract_categories(categories_list):
    """categories_tags 是 list<string>，拼成逗号分隔字符串"""
    if categories_list is None:
        return None
    try:
        if len(categories_list) == 0:
            return None
    except TypeError:
        return None
    return ",".join(str(c) for c in categories_list if c)


def extract_nutriments(nutriments_arr):
    """
    从 nutriments 数组中提取指定营养素的 per-100g 值。
    返回字典 {nutrient_name: value_per_100g}
    """
    result = {n: None for n in NUTRIENT_NAMES}
    if nutriments_arr is None:
        return result
    try:
        if len(nutriments_arr) == 0:
            return result
    except TypeError:
        return result
    for item in nutriments_arr:
        if isinstance(item, dict):
            name = item.get("name", "")
            if name in result:
                val = item.get("100g")
                if val is not None:
                    try:
                        result[name] = round(float(val), 4)
                    except (ValueError, TypeError):
                        pass
    return result


def build():
    if not os.path.exists(PARQUET_FILE):
        print(f"错误: 找不到 {PARQUET_FILE}")
        print("请先下载:")
        print("https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet?download=true")
        sys.exit(1)

    if os.path.exists(SQLITE_FILE):
        os.remove(SQLITE_FILE)
        print(f"已删除旧的 {SQLITE_FILE}")

    print(f"正在读取 {PARQUET_FILE}（只读 {len(COLUMNS)} 列）...")
    df = pd.read_parquet(PARQUET_FILE, columns=COLUMNS)
    print(f"总商品数: {len(df):,}")

    # 提取英文商品名
    print("正在提取英文商品名...")
    df["product_name_en"] = df["product_name"].apply(extract_en_name)

    # 提取分类
    print("正在处理分类...")
    df["categories_str"] = df["categories_tags"].apply(extract_categories)

    # 提取营养数据
    print("正在提取营养数据（这一步较慢）...")
    nutrient_data = df["nutriments"].apply(extract_nutriments)
    nutrient_df = pd.DataFrame(nutrient_data.tolist())

    # 重命名列：把 "-" 换成 "_"，方便 SQL 查询
    col_rename = {n: n.replace("-", "_") for n in NUTRIENT_NAMES}
    nutrient_df = nutrient_df.rename(columns=col_rename)

    print("正在合并数据...")
    result = pd.concat([
        df[["code", "product_name_en", "brands", "quantity", "categories_str"]],
        nutrient_df,
    ], axis=1)

    result = result.rename(columns={
        "product_name_en": "product_name",
        "categories_str": "categories",
    })

    # 过滤：必须有商品名和 quantity
    print("正在过滤...")
    result = result.dropna(subset=["product_name", "quantity"])
    result = result[result["product_name"].str.strip() != ""]
    result = result[result["quantity"].str.strip() != ""]
    print(f"有效商品数: {len(result):,}")

    # 写入 SQLite
    print(f"正在写入 {SQLITE_FILE}...")
    conn = sqlite3.connect(SQLITE_FILE)
    result.to_sql("products", conn, index=False, if_exists="replace")

    print("正在创建索引...")
    conn.execute("CREATE INDEX idx_product_name ON products(product_name)")
    conn.execute("CREATE INDEX idx_brands ON products(brands)")
    conn.commit()

    # 验证
    count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    has_nutrition = conn.execute(
        "SELECT COUNT(*) FROM products WHERE energy_kcal IS NOT NULL"
    ).fetchone()[0]

    print(f"\n===== 样例 =====")
    rows = conn.execute("""
        SELECT product_name, quantity, energy_kcal, fat, proteins, carbohydrates
        FROM products
        WHERE energy_kcal IS NOT NULL
        AND brands LIKE '%Trader Joe%'
        LIMIT 5
    """).fetchall()
    for r in rows:
        print(f"  {r[0]} | {r[1]} | {r[2]}kcal | fat:{r[3]}g | prot:{r[4]}g | carb:{r[5]}g")

    # 表结构
    schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='products'"
    ).fetchone()[0]
    print(f"\n===== 表结构 =====")
    print(schema)

    conn.close()
    size_mb = os.path.getsize(SQLITE_FILE) / (1024 * 1024)
    print(f"\n完成! {SQLITE_FILE} ({size_mb:.1f} MB)")
    print(f"共 {count:,} 条有效商品记录")
    print(f"其中 {has_nutrition:,} 条有热量数据")


if __name__ == "__main__":
    build()