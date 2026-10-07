import pandas as pd
import os

print("======================================")
print("COPERNICUS PRODUCT ANALYSIS")
print("======================================")

file = "data/products/copernicus_products.csv"

if not os.path.exists(file):
    print("ERROR: File not found:")
    print(file)
    exit()

df = pd.read_csv(file)

print("\nTotal matched products:", len(df))

print("\nColumns:")
for col in df.columns:
    print(" -", col)


# --------------------------------------
# Product names
# --------------------------------------

if "product_name" in df.columns:

    print("\nFirst 10 products:")

    for name in df["product_name"].head(10):
        print(name)


# --------------------------------------
# Product size
# --------------------------------------

if "content_length" in df.columns:

    df["content_length"] = pd.to_numeric(
        df["content_length"],
        errors="coerce"
    )

    total_bytes = df["content_length"].sum()

    total_gb = total_bytes / (1024 ** 3)

    print("\n--------------------------------------")
    print("STORAGE")
    print("--------------------------------------")

    print("Total size:", round(total_gb, 2), "GB")

    print(
        "Average product size:",
        round(df["content_length"].mean() / (1024 ** 2), 2),
        "MB"
    )


# --------------------------------------
# Online status
# --------------------------------------

if "online" in df.columns:

    print("\n--------------------------------------")
    print("ONLINE STATUS")
    print("--------------------------------------")

    print(df["online"].value_counts(dropna=False))


# --------------------------------------
# Save analysis copy
# --------------------------------------

output = "data/products/product_analysis.csv"

df.to_csv(output, index=False)

print("\nAnalysis saved to:")
print(output)

print("\n======================================")
print("ANALYSIS COMPLETED")
print("======================================")