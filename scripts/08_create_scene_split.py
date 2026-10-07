import pandas as pd
import os


input_file = r"data\scene_selection\scene_statistics.csv"

output_dir = r"data\scene_selection"

train_file = os.path.join(output_dir, "train_scenes.csv")
val_file = os.path.join(output_dir, "val_scenes.csv")
test_file = os.path.join(output_dir, "test_scenes.csv")


# --------------------------------------------------
# Read scene statistics
# --------------------------------------------------

df = pd.read_csv(input_file)


# --------------------------------------------------
# Keep only scenes containing labelled patches
# --------------------------------------------------

df = df[df["total_patches"] > 0].copy()


# --------------------------------------------------
# Require both classes when possible
# --------------------------------------------------

balanced = df[
    (df["oil"] > 0) &
    (df["no_oil"] > 0)
].copy()


print("Scenes with both oil and no_oil:", len(balanced))


# --------------------------------------------------
# Sort by useful labelled data
# --------------------------------------------------

balanced = balanced.sort_values(
    by="total_patches",
    ascending=False
)


# --------------------------------------------------
# Select scenes
# --------------------------------------------------

required = 25

if len(balanced) < required:
    print(
        "\nWARNING:",
        "Less than 25 balanced scenes available."
    )

selected = balanced.head(required).copy()


# --------------------------------------------------
# Split
# --------------------------------------------------

train = selected.iloc[:15].copy()

val = selected.iloc[15:20].copy()

test = selected.iloc[20:25].copy()


# --------------------------------------------------
# Add split column
# --------------------------------------------------

train["split"] = "train"
val["split"] = "val"
test["split"] = "test"


# --------------------------------------------------
# Save
# --------------------------------------------------

train.to_csv(train_file, index=False)

val.to_csv(val_file, index=False)

test.to_csv(test_file, index=False)


# --------------------------------------------------
# Print
# --------------------------------------------------

print("\n================================")
print("SCENE SPLIT CREATED")
print("================================")

print("\nTRAIN:", len(train))
print(train[
    ["sentinel_name", "date", "oil", "no_oil", "total_patches"]
].to_string(index=False))

print("\nVALIDATION:", len(val))
print(val[
    ["sentinel_name", "date", "oil", "no_oil", "total_patches"]
].to_string(index=False))

print("\nTEST:", len(test))
print(test[
    ["sentinel_name", "date", "oil", "no_oil", "total_patches"]
].to_string(index=False))

print("\nFiles created:")
print(train_file)
print(val_file)
print(test_file)