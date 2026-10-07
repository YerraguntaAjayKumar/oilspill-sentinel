import pandas as pd
import os
import sys


# --------------------------------------------------
# INPUT / OUTPUT PATHS
# --------------------------------------------------

train_file = r"data\scene_selection\train_scenes.csv"

output_dir = r"data\scene_selection\batches"

os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------------
# GET BATCH NUMBER
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage:")
    print("python scripts\\09_create_download_batch.py 1")
    print("python scripts\\09_create_download_batch.py 2")
    print("python scripts\\09_create_download_batch.py 3")
    sys.exit()


batch_number = int(sys.argv[1])


# --------------------------------------------------
# READ TRAIN SCENES
# --------------------------------------------------

df = pd.read_csv(train_file)


# --------------------------------------------------
# BATCH SIZE
# --------------------------------------------------

# Each Sentinel scene is about 1.6 GB.
# We use 2 scenes per batch to stay within
# the 5 GB storage limit.

batch_size = 2


# --------------------------------------------------
# CALCULATE BATCH RANGE
# --------------------------------------------------

start = (batch_number - 1) * batch_size
end = start + batch_size


batch = df.iloc[start:end].copy()


# --------------------------------------------------
# CHECK IF BATCH EXISTS
# --------------------------------------------------

if len(batch) == 0:

    print()
    print("================================")
    print("NO SCENES AVAILABLE")
    print("================================")

    print("Requested batch:", batch_number)

    print("Total training scenes:", len(df))

    sys.exit()


# --------------------------------------------------
# CALCULATE TOTAL SIZE
# --------------------------------------------------

total_size = batch["size_gb"].sum()


# --------------------------------------------------
# SAVE BATCH
# --------------------------------------------------

output_file = os.path.join(
    output_dir,
    f"train_batch_{batch_number}.csv"
)


batch.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print()
print("================================")
print("DOWNLOAD BATCH CREATED")
print("================================")

print("Batch number:", batch_number)

print("Scenes in batch:", len(batch))

print()


print("Selected scenes:")
print(
    batch[
        [
            "sentinel_name",
            "date",
            "oil",
            "no_oil",
            "total_patches",
            "size_gb"
        ]
    ].to_string(index=False)
)


print()
print("--------------------------------")
print("BATCH STORAGE")
print("--------------------------------")

print(
    "Total Sentinel size:",
    round(total_size, 2),
    "GB"
)


# --------------------------------------------------
# STORAGE WARNING
# --------------------------------------------------

if total_size > 4.5:

    print()
    print("WARNING:")
    print("This batch is larger than 4.5 GB.")
    print("Do NOT download it with the current")
    print("5 GB storage limit.")

elif total_size > 4:

    print()
    print("WARNING:")
    print("Batch is above 4 GB.")
    print("Check available disk space before downloading.")

else:

    print()
    print("Storage check: OK")


# --------------------------------------------------
# OUTPUT FILE
# --------------------------------------------------

print()
print("--------------------------------")
print("BATCH FILE")
print("--------------------------------")

print(output_file)

print()
print("Next command:")
print(
    f"python scripts\\10_download_batch.py {batch_number}"
)