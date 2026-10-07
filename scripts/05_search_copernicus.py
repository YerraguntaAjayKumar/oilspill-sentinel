import requests
import pandas as pd
import os
import time

print("======================================")
print("SEARCHING COPERNICUS DATA SPACE")
print("======================================")

# --------------------------------------
# FILES
# --------------------------------------

catalogue_file = "data/products/clean_sentinel_products.csv"
token_file = "access_token.txt"
output_file = "data/products/copernicus_products.csv"


# --------------------------------------
# CHECK FILES
# --------------------------------------

if not os.path.exists(catalogue_file):
    print("\nERROR: Catalogue file not found:")
    print(catalogue_file)
    exit()

if not os.path.exists(token_file):
    print("\nERROR: access_token.txt not found.")
    exit()


# --------------------------------------
# LOAD TOKEN
# --------------------------------------

with open(token_file, "r") as f:
    token = f.read().strip()

if not token:
    print("\nERROR: access_token.txt is empty.")
    print("Run:")
    print("python scripts\\02_get_token.py")
    exit()


# --------------------------------------
# LOAD PRODUCT NAMES
# --------------------------------------

df = pd.read_csv(catalogue_file)

print("\nDARTIS products loaded:", len(df))

# Remove empty names
df = df.dropna(subset=["sentinel_id"])

# Remove duplicates
product_names = (
    df["sentinel_id"]
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

print("Unique products to search:", len(product_names))


# --------------------------------------
# CDSE API
# --------------------------------------

url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

headers = {
    "Authorization": f"Bearer {token}"
}


# --------------------------------------
# SEARCH PRODUCTS
# --------------------------------------

results = []
not_found = []

total = len(product_names)

for i, product_name in enumerate(product_names, start=1):

    print(f"\rSearching {i}/{total}: {product_name}", end="")

    params = {
        "$filter": f"Name eq '{product_name}'",
        "$top": "1"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        # Authentication failure
        if response.status_code == 401:
            print("\n\nERROR: Access token is invalid or expired.")
            print("Run:")
            print("python scripts\\02_get_token.py")
            exit()

        # Other API error
        if response.status_code != 200:
            not_found.append({
                "sentinel_name": product_name,
                "reason": f"HTTP {response.status_code}"
            })
            continue

        data = response.json()
        products = data.get("value", [])

        if products:

            product = products[0]

            results.append({
                "sentinel_name": product_name,
                "product_id": product.get("Id"),
                "product_name": product.get("Name"),
                "content_length": product.get("ContentLength"),
                "online": product.get("Online"),
                "s3_path": product.get("S3Path")
            })

        else:

            not_found.append({
                "sentinel_name": product_name,
                "reason": "Not found"
            })

    except requests.exceptions.RequestException as e:

        not_found.append({
            "sentinel_name": product_name,
            "reason": str(e)
        })

    # Small delay to avoid hammering the API
    time.sleep(0.1)


# --------------------------------------
# SAVE RESULTS
# --------------------------------------

print("\n\n======================================")
print("SEARCH COMPLETED")
print("======================================")

found_df = pd.DataFrame(results)
not_found_df = pd.DataFrame(not_found)

found_df.to_csv(
    output_file,
    index=False
)

not_found_file = "data/products/not_found_products.csv"

not_found_df.to_csv(
    not_found_file,
    index=False
)


# --------------------------------------
# SUMMARY
# --------------------------------------

print()
print("Total searched :", total)
print("Products found :", len(found_df))
print("Not found      :", len(not_found_df))

print()
print("Saved found products to:")
print(output_file)

print()
print("Saved missing products to:")
print(not_found_file)

print()
print("======================================")
print("DONE")
print("======================================")