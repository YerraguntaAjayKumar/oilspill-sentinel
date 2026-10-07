import pandas as pd
import os

# Input catalogue
input_file = "data/catalogue/dartis_catalogue.csv"

# Output folder
output_folder = "data/products"
os.makedirs(output_folder, exist_ok=True)

# Read catalogue
df = pd.read_csv(input_file)

print("Catalogue rows:", len(df))

# Get unique Sentinel-1 products
products = df[
    [
        "sentinel_id",
        "start_time",
        "end_time"
    ]
].drop_duplicates()

# Sort by acquisition time
products = products.sort_values("start_time")

# Save
output_file = os.path.join(
    output_folder,
    "sentinel_products.csv"
)

products.to_csv(
    output_file,
    index=False
)

print()
print("==============================")
print("SENTINEL PRODUCTS CREATED")
print("==============================")

print("Unique products:", len(products))

print()
print("First 10 products:")

print(products.head(10).to_string(index=False))

print()
print("Saved to:")
print(output_file)