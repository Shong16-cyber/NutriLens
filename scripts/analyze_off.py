"""Quick analysis of OFF database categories and quantities."""
import sqlite3
from collections import Counter

conn = sqlite3.connect('off_products.db')

# Sample categories
print('=== Categories samples ===')
for row in conn.execute("SELECT product_name, categories, quantity FROM products WHERE categories != '' LIMIT 10"):
    name = (row[0] or '')[:40]
    cat = (row[1] or '')[:60]
    qty = row[2] or ''
    print(f'  {name:<42} cat={cat}')
    print(f'  {"":42} qty={qty}')

# Counts
total = conn.execute('SELECT COUNT(*) FROM products').fetchone()[0]
with_cat = conn.execute("SELECT COUNT(*) FROM products WHERE categories != ''").fetchone()[0]
with_qty = conn.execute("SELECT COUNT(*) FROM products WHERE quantity != ''").fetchone()[0]
both = conn.execute("SELECT COUNT(*) FROM products WHERE categories != '' AND quantity != ''").fetchone()[0]
print(f'\nTotal: {total}')
print(f'Has categories: {with_cat} ({with_cat/total*100:.1f}%)')
print(f'Has quantity: {with_qty} ({with_qty/total*100:.1f}%)')
print(f'Has both: {both} ({both/total*100:.1f}%)')

# Top category tags
print('\n=== Top 20 category tags ===')
tag_counts = Counter()
for row in conn.execute("SELECT categories FROM products WHERE categories != ''"):
    for tag in row[0].split(','):
        tag = tag.strip()
        if tag.startswith('en:'):
            tag_counts[tag] += 1
for tag, count in tag_counts.most_common(20):
    print(f'  {tag:<45} {count:>6}')

conn.close()