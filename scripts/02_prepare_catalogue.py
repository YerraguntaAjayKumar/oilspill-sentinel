import pandas as pd
import os

input_file = "data/events/dartis_2019.txt"
output_folder = "data/catalogue"

os.makedirs(output_folder, exist_ok=True)

# Find the real table header
with open(input_file, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

header_line = None

for i, line in enumerate(lines):
    if line.startswith("Image set"):
        header_line = i
        break

if header_line is None:
    print("ERROR: Table header not found.")
    exit()

print("Table starts at line:", header_line + 1)

# Read the table
df = pd.read_csv(
    input_file,
    sep="\t",
    skiprows=header_line
)

print("Total rows:", len(df))

# Rename important columns
df = df.rename(columns={
    "Image set (subset; oc : oil/coast; ow : ...)": "subset",
    "IMAGE (jpg_file)": "image_file",
    "Binary (xml_file)": "xml_file",
    "ID (tag)": "tag",
    "ID (patch_name)": "patch_name",
    "Date/Time (start_time)": "start_time",
    "Date/Time (end_time)": "end_time",
    "ID (Sentinel_ID)": "sentinel_id",
    "Width [pixel] (patch_width)": "patch_width",
    "Height [pixel] (patch_height)": "patch_height",
    "Longitude (patch_ul_lon)": "patch_ul_lon",
    "Latitude (patch_ul_lat)": "patch_ul_lat",
    "Pos X [pixel] (obj_patchloc_xmin)": "obj_xmin",
    "Pos Y [pixel] (obj_patchloc_ymin)": "obj_ymin",
    "Pos X [pixel] (obj_patchloc_xmax)": "obj_xmax",
    "Pos Y [pixel] (obj_patchloc_ymax)": "obj_ymax",
    "Size [pixel] (label_size)": "label_size"
})

# Create simple label
df["label"] = df["subset"].map({
    "oc": "oil",
    "ow": "oil",
    "nc": "no_oil",
    "nw": "no_oil"
})

# Keep important information
catalogue = df[
    [
        "patch_name",
        "start_time",
        "end_time",
        "sentinel_id",
        "patch_width",
        "patch_height",
        "patch_ul_lon",
        "patch_ul_lat",
        "label",
        "subset",
        "obj_xmin",
        "obj_ymin",
        "obj_xmax",
        "obj_ymax",
        "label_size"
    ]
].copy()

# Remove duplicate patches
catalogue = catalogue.drop_duplicates(
    subset=["patch_name"]
)

# Save
output_file = os.path.join(
    output_folder,
    "dartis_catalogue.csv"
)

catalogue.to_csv(output_file, index=False)

print()
print("==============================")
print("CATALOGUE CREATED SUCCESSFULLY")
print("==============================")
print("Unique patches:", len(catalogue))
print()
print("Labels:")
print(catalogue["label"].value_counts())
print()
print("Unique Sentinel products:",
      catalogue["sentinel_id"].nunique())
print()
print("Saved to:", output_file)