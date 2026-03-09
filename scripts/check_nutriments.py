import pandas as pd

df = pd.read_parquet("food.parquet", columns=["nutriments"])
sample = df["nutriments"].dropna().iloc[0]
print("Type:", type(sample))
print("Value:", sample)