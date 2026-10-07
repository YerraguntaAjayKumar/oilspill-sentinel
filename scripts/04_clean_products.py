import pandas as pd
import os

# Input
input_file = "data/catalogue/dartis_catalogue.csv"

# Output folder
output_folder = "data/products"
os.makedirs(output_folder, exist_ok=True)

# Read catalogue
df = pd.read_csv(input_file)

print("Catalogue rows:", len(df))

# Get Sentinel product column
products = df["sentinel_id"].dropna()

# Split rows containing multiple products
all_products = []

for value in products:
    parts = str(value).split(";")

    for product in parts:
        product = product.strip()

        if product.endswith(".SAFE"):
            all_products.append(product)

# Remove duplicates
all_products = sorted(set(all_products))

# Create dataframe
product_df = pd.DataFrame({
    "sentinel_id": all_products
})

# Save
output_file = os.path.join(
    output_folder,
    "clean_sentinel_products.csv"
)

product_df.to_csv(
    output_file,
    index=False
)

print()
print("==============================")
print("PRODUCT LIST CLEANED")
print("==============================")

print("Original catalogue rows:", len(df))
print("Unique individual products:", len(product_df))

print()
print("First 10 products:")

print(product_df.head(10).to_string(index=False))

print()
print("Saved to:")
print(output_file)