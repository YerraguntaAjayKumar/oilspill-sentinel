# ============================================================
# 19_extract_dartis_patches.py
# ============================================================
# Extract 640x640 VV and VH patches using the validated
# DARTIS -> Sentinel-1 pixel mapping.
# ============================================================

import os
import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MAPPING_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "catalogue",
    "dartis_pixel_mapping_precise.csv"
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_batch_1"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "patches",
    "train_batch_1"
)


# ============================================================
# PATCH SIZE
# ============================================================

PATCH_WIDTH = 640
PATCH_HEIGHT = 640

HALF_WIDTH = PATCH_WIDTH // 2
HALF_HEIGHT = PATCH_HEIGHT // 2


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OIL_DIR = os.path.join(
    OUTPUT_DIR,
    "oil"
)

NO_OIL_DIR = os.path.join(
    OUTPUT_DIR,
    "no_oil"
)

os.makedirs(
    OIL_DIR,
    exist_ok=True
)

os.makedirs(
    NO_OIL_DIR,
    exist_ok=True
)


METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "patch_metadata.csv"
)


# ============================================================
# NORMALIZE SCENE NAME
# ============================================================

def normalize_scene(scene):

    scene = str(scene).strip()

    if scene.lower().endswith(".zip"):
        scene = scene[:-4]

    if not scene.lower().endswith(".safe"):
        scene += ".SAFE"

    return scene.lower()


# ============================================================
# FIND TIFF FILES
# ============================================================

def find_tiff(scene, polarization):

    scene_normalized = normalize_scene(
        scene
    )

    polarization = polarization.lower()

    print()
    print(
        f"Searching for {polarization.upper()} TIFF..."
    )

    # --------------------------------------------------------
    # List all TIFF files
    # --------------------------------------------------------

    tiff_files = []

    for filename in os.listdir(
        PROCESSED_DIR
    ):

        lower = filename.lower()

        if (
            lower.endswith(".tif")
            or
            lower.endswith(".tiff")
        ):

            tiff_files.append(
                filename
            )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    for filename in tiff_files:

        lower = filename.lower()

        # Scene must occur in filename
        if scene_normalized not in lower:
            continue

        # Polarization must occur
        # as -vv- or -vh-
        if f"-{polarization}-" not in lower:
            continue

        path = os.path.join(
            PROCESSED_DIR,
            filename
        )

        print(
            "FOUND:",
            filename
        )

        return path

    # --------------------------------------------------------
    # If exact search fails, print available files
    # --------------------------------------------------------

    print(
        "Exact filename match failed."
    )

    print(
        "Available TIFF files:"
    )

    for filename in tiff_files:

        print(
            "   ",
            filename
        )

    return None


# ============================================================
# EXTRACT PATCH
# ============================================================

def extract_patch(
    tif_path,
    center_row,
    center_pixel
):

    with rasterio.open(
        tif_path
    ) as src:

        width = src.width
        height = src.height

        # ----------------------------------------------------
        # Convert center to integer
        # ----------------------------------------------------

        center_row = int(
            round(center_row)
        )

        center_pixel = int(
            round(center_pixel)
        )

        # ----------------------------------------------------
        # Calculate crop
        # ----------------------------------------------------

        row_start = (
            center_row
            - HALF_HEIGHT
        )

        row_end = (
            row_start
            + PATCH_HEIGHT
        )

        col_start = (
            center_pixel
            - HALF_WIDTH
        )

        col_end = (
            col_start
            + PATCH_WIDTH
        )

        print(
            "Image size:",
            width,
            "x",
            height
        )

        print(
            "Crop:",
            f"rows {row_start}:{row_end},",
            f"cols {col_start}:{col_end}"
        )

        # ----------------------------------------------------
        # Boundary check
        # ----------------------------------------------------

        if row_start < 0:

            return None, "TOP_BOUNDARY"

        if col_start < 0:

            return None, "LEFT_BOUNDARY"

        if row_end > height:

            return None, "BOTTOM_BOUNDARY"

        if col_end > width:

            return None, "RIGHT_BOUNDARY"

        # ----------------------------------------------------
        # Read patch
        # ----------------------------------------------------

        window = rasterio.windows.Window(
            col_start,
            row_start,
            PATCH_WIDTH,
            PATCH_HEIGHT
        )

        patch = src.read(
            1,
            window=window
        )

        return patch, None


# ============================================================
# MAIN
# ============================================================

print("=" * 90)
print("DARTIS → SENTINEL-1 PATCH EXTRACTION")
print("=" * 90)


# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not os.path.exists(
    PROCESSED_DIR
):

    print()
    print(
        "ERROR: Processed directory not found:"
    )

    print(
        PROCESSED_DIR
    )

    raise SystemExit


print()
print(
    "Processed directory:"
)

print(
    PROCESSED_DIR
)


# ============================================================
# SHOW TIFF FILES
# ============================================================

print()
print(
    "Available TIFF files:"
)

available_tiffs = []

for filename in os.listdir(
    PROCESSED_DIR
):

    lower = filename.lower()

    if (
        lower.endswith(".tif")
        or
        lower.endswith(".tiff")
    ):

        available_tiffs.append(
            filename
        )

        print(
            "   ",
            filename
        )


print()
print(
    "Total TIFF files:",
    len(available_tiffs)
)


# ============================================================
# LOAD MAPPING
# ============================================================

if not os.path.exists(
    MAPPING_FILE
):

    print()
    print(
        "ERROR: Mapping file not found:"
    )

    print(
        MAPPING_FILE
    )

    raise SystemExit


df = pd.read_csv(
    MAPPING_FILE
)


print()
print(
    "Mapping records:",
    len(df)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "patch_name",
    "sentinel_id",
    "image_row",
    "image_pixel",
    "label"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if len(missing) > 0:

    print()
    print(
        "ERROR: Missing columns:"
    )

    print(
        missing
    )

    raise SystemExit


# ============================================================
# COUNTERS
# ============================================================

successful = 0
failed = 0

oil_count = 0
no_oil_count = 0

metadata = []


# ============================================================
# PROCESS PATCHES
# ============================================================

for index, record in df.iterrows():

    print()
    print("-" * 90)

    print(
        f"PATCH {index + 1}/{len(df)}"
    )

    patch_name = str(
        record["patch_name"]
    )

    scene = str(
        record["sentinel_id"]
    )

    label = str(
        record["label"]
    ).strip().lower()

    center_row = float(
        record["image_row"]
    )

    center_pixel = float(
        record["image_pixel"]
    )

    print(
        "Name   :",
        patch_name
    )

    print(
        "Scene  :",
        scene
    )

    print(
        "Label  :",
        label
    )

    print(
        "Center :",
        f"row={center_row:.2f},",
        f"pixel={center_pixel:.2f}"
    )


    # ========================================================
    # LABEL CHECK
    # ========================================================

    if label not in [
        "oil",
        "no_oil"
    ]:

        print(
            "ERROR: Invalid label"
        )

        failed += 1

        continue


    # ========================================================
    # FIND VV
    # ========================================================

    vv_file = find_tiff(
        scene,
        "vv"
    )

    if vv_file is None:

        print(
            "ERROR: VV TIFF not found"
        )

        failed += 1

        continue


    # ========================================================
    # FIND VH
    # ========================================================

    vh_file = find_tiff(
        scene,
        "vh"
    )

    if vh_file is None:

        print(
            "ERROR: VH TIFF not found"
        )

        failed += 1

        continue


    # ========================================================
    # EXTRACT VV
    # ========================================================

    print()
    print(
        "Extracting VV..."
    )

    vv_patch, vv_error = extract_patch(
        vv_file,
        center_row,
        center_pixel
    )

    if vv_patch is None:

        print(
            "ERROR: VV extraction failed:",
            vv_error
        )

        failed += 1

        continue


    # ========================================================
    # EXTRACT VH
    # ========================================================

    print()
    print(
        "Extracting VH..."
    )

    vh_patch, vh_error = extract_patch(
        vh_file,
        center_row,
        center_pixel
    )

    if vh_patch is None:

        print(
            "ERROR: VH extraction failed:",
            vh_error
        )

        failed += 1

        continue


    # ========================================================
    # VERIFY SHAPE
    # ========================================================

    if vv_patch.shape != (
        PATCH_HEIGHT,
        PATCH_WIDTH
    ):

        print(
            "ERROR: VV shape:",
            vv_patch.shape
        )

        failed += 1

        continue


    if vh_patch.shape != (
        PATCH_HEIGHT,
        PATCH_WIDTH
    ):

        print(
            "ERROR: VH shape:",
            vh_patch.shape
        )

        failed += 1

        continue


    # ========================================================
    # PATCH ID
    # ========================================================

    patch_id = (
        f"{index + 1:04d}_"
        f"{patch_name}"
    )


    # ========================================================
    # LABEL DIRECTORY
    # ========================================================

    if label == "oil":

        label_dir = OIL_DIR

    else:

        label_dir = NO_OIL_DIR


    # ========================================================
    # OUTPUT FILES
    # ========================================================

    vv_output = os.path.join(
        label_dir,
        patch_id + "_VV.npy"
    )

    vh_output = os.path.join(
        label_dir,
        patch_id + "_VH.npy"
    )


    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        vv_output,
        vv_patch
    )

    np.save(
        vh_output,
        vh_patch
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    vv_min = float(
        np.min(vv_patch)
    )

    vv_max = float(
        np.max(vv_patch)
    )

    vv_mean = float(
        np.mean(vv_patch)
    )

    vh_min = float(
        np.min(vh_patch)
    )

    vh_max = float(
        np.max(vh_patch)
    )

    vh_mean = float(
        np.mean(vh_patch)
    )

    vv_zero_percent = (
        np.sum(vv_patch == 0)
        /
        vv_patch.size
        *
        100
    )

    vh_zero_percent = (
        np.sum(vh_patch == 0)
        /
        vh_patch.size
        *
        100
    )


    # ========================================================
    # METADATA
    # ========================================================

    metadata.append({

        "patch_id":
            patch_id,

        "patch_name":
            patch_name,

        "sentinel_id":
            scene,

        "label":
            label,

        "image_row":
            center_row,

        "image_pixel":
            center_pixel,

        "patch_height":
            PATCH_HEIGHT,

        "patch_width":
            PATCH_WIDTH,

        "vv_file":
            vv_output,

        "vh_file":
            vh_output,

        "vv_min":
            vv_min,

        "vv_max":
            vv_max,

        "vv_mean":
            vv_mean,

        "vh_min":
            vh_min,

        "vh_max":
            vh_max,

        "vh_mean":
            vh_mean,

        "vv_zero_percent":
            vv_zero_percent,

        "vh_zero_percent":
            vh_zero_percent
    })


    # ========================================================
    # COUNTS
    # ========================================================

    successful += 1

    if label == "oil":

        oil_count += 1

    else:

        no_oil_count += 1


    print()
    print(
        "SUCCESS"
    )

    print(
        "VV shape:",
        vv_patch.shape
    )

    print(
        "VH shape:",
        vh_patch.shape
    )

    print(
        "VV saved:",
        vv_output
    )

    print(
        "VH saved:",
        vh_output
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(
    metadata
)

metadata_df.to_csv(
    METADATA_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 90)
print("PATCH EXTRACTION COMPLETE")
print("=" * 90)

print()

print(
    "Total mappings:",
    len(df)
)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

print()

print(
    "Oil patches:",
    oil_count
)

print(
    "No-oil patches:",
    no_oil_count
)

print()

print(
    "Patch size:",
    f"{PATCH_WIDTH} × {PATCH_HEIGHT}"
)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Metadata:"
)

print(
    METADATA_FILE
)

print()
print("=" * 90)