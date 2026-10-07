import pandas as pd
import os


# --------------------------------------------------
# Paths
# --------------------------------------------------

catalogue_file = r"data\catalogue\dartis_catalogue.csv"
products_file = r"data\products\copernicus_products.csv"

output_dir = r"data\scene_selection"
output_file = os.path.join(output_dir, "scene_statistics.csv")


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------------
# Read files
# --------------------------------------------------

print("Reading DARTIS catalogue...")
catalogue = pd.read_csv(catalogue_file)

print("Reading Copernicus products...")
products = pd.read_csv(products_file)


print("\nCatalogue columns:")
print(catalogue.columns.tolist())

print("\nCopernicus columns:")
print(products.columns.tolist())


# --------------------------------------------------
# Rename catalogue columns if necessary
# --------------------------------------------------

print("\nCatalogue sample:")
print(catalogue.head())


# --------------------------------------------------
# Find Sentinel ID column
# --------------------------------------------------

sentinel_col = None

for col in catalogue.columns:
    if "sentinel" in col.lower():
        sentinel_col = col
        break

if sentinel_col is None:
    raise ValueError("Could not find Sentinel ID column in catalogue.")

print("\nUsing Sentinel column:", sentinel_col)


# --------------------------------------------------
# Find label column
# --------------------------------------------------

label_col = None

for col in catalogue.columns:
    if col.lower() == "label":
        label_col = col
        break

if label_col is None:
    raise ValueError("Could not find label column in catalogue.")

print("Using label column:", label_col)


# --------------------------------------------------
# Clean Sentinel IDs
# --------------------------------------------------

catalogue["sentinel_id_clean"] = (
    catalogue[sentinel_col]
    .astype(str)
    .str.split(";")
    .str[0]
    .str.strip()
)


# --------------------------------------------------
# Clean product names
# --------------------------------------------------

products["sentinel_name"] = (
    products["sentinel_name"]
    .astype(str)
    .str.strip()
)


# --------------------------------------------------
# Count labels for each Sentinel scene
# --------------------------------------------------

scene_counts = (
    catalogue
    .groupby(["sentinel_id_clean", label_col])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)


# Make sure both columns exist

if "oil" not in scene_counts.columns:
    scene_counts["oil"] = 0

if "no_oil" not in scene_counts.columns:
    scene_counts["no_oil"] = 0


scene_counts["total_patches"] = (
    scene_counts["oil"] +
    scene_counts["no_oil"]
)


# --------------------------------------------------
# Merge with Copernicus information
# --------------------------------------------------

scene_stats = products.merge(
    scene_counts,
    left_on="sentinel_name",
    right_on="sentinel_id_clean",
    how="left"
)


# --------------------------------------------------
# Fill missing counts
# --------------------------------------------------

scene_stats["oil"] = scene_stats["oil"].fillna(0).astype(int)
scene_stats["no_oil"] = scene_stats["no_oil"].fillna(0).astype(int)
scene_stats["total_patches"] = (
    scene_stats["total_patches"]
    .fillna(0)
    .astype(int)
)


# --------------------------------------------------
# Convert size to GB
# --------------------------------------------------

scene_stats["size_gb"] = (
    scene_stats["content_length"] /
    (1024 ** 3)
)


# --------------------------------------------------
# Extract date
# --------------------------------------------------

scene_stats["date"] = (
    scene_stats["sentinel_name"]
    .str.extract(r"_(\d{8})T")
)


# --------------------------------------------------
# Sort by labelled patches
# --------------------------------------------------

scene_stats = scene_stats.sort_values(
    by="total_patches",
    ascending=False
)


# --------------------------------------------------
# Save
# --------------------------------------------------

scene_stats.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n====================================")
print("SCENE ANALYSIS COMPLETE")
print("====================================")

print("Total Copernicus scenes:", len(products))

print(
    "Scenes with labelled patches:",
    (scene_stats["total_patches"] > 0).sum()
)

print(
    "Scenes without labelled patches:",
    (scene_stats["total_patches"] == 0).sum()
)

print("\nTop 20 scenes:")
print(
    scene_stats[
        [
            "sentinel_name",
            "date",
            "oil",
            "no_oil",
            "total_patches",
            "size_gb"
        ]
    ].head(20).to_string(index=False)
)

print("\nSaved:")
print(output_file)