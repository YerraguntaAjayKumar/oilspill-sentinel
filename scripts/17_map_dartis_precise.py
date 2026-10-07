import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CATALOGUE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "catalogue",
    "dartis_catalogue.csv"
)

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "train_batch_1"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "catalogue",
    "dartis_pixel_mapping_precise.csv"
)


# ============================================================
# FIND DOWNLOADED SCENES
# ============================================================

def get_downloaded_scenes():

    scenes = []

    for filename in os.listdir(RAW_DIR):

        if not filename.lower().endswith(".zip"):
            continue

        # Example:
        # S1A_xxxxxxxxx.SAFE.zip
        scene = filename[:-4]

        scenes.append(scene)

    return scenes


# ============================================================
# FIND ZIP FOR SCENE
# ============================================================

def find_zip(scene):

    expected_filename = scene + ".zip"

    path = os.path.join(
        RAW_DIR,
        expected_filename
    )

    if os.path.exists(path):
        return path

    return None


# ============================================================
# FIND VV ANNOTATION XML
# ============================================================

def get_annotation_xml(zip_path):

    with zipfile.ZipFile(zip_path, "r") as z:

        for name in z.namelist():

            lower = name.lower()

            # IMPORTANT:
            # Do not accidentally select calibration/noise XML.
            if (
                "/annotation/" in lower
                and "/calibration/" not in lower
                and "/noise/" not in lower
                and "-vv-" in lower
                and lower.endswith(".xml")
            ):

                return name

    return None


# ============================================================
# READ SENTINEL-1 GEOLOCATION GRID
# ============================================================

def read_geolocation_grid(zip_path, xml_name):

    with zipfile.ZipFile(zip_path, "r") as z:

        xml_data = z.read(xml_name)

    root = ET.fromstring(xml_data)

    points = []

    for node in root.iter():

        if not node.tag.endswith(
            "geolocationGridPoint"
        ):
            continue

        line = None
        pixel = None
        latitude = None
        longitude = None

        for child in node:

            tag = child.tag.split("}")[-1]

            if tag == "line":

                line = float(child.text)

            elif tag == "pixel":

                pixel = float(child.text)

            elif tag == "latitude":

                latitude = float(child.text)

            elif tag == "longitude":

                longitude = float(child.text)

        if None not in (
            line,
            pixel,
            latitude,
            longitude
        ):

            points.append([
                line,
                pixel,
                latitude,
                longitude
            ])

    return np.array(
        points,
        dtype=float
    )


# ============================================================
# LOCAL LEAST-SQUARES GEO → PIXEL MAPPING
#
# latitude  = a + b*row + c*pixel
# longitude = d + e*row + f*pixel
# ============================================================

def map_geo_to_pixel(
    points,
    target_lat,
    target_lon,
    n_neighbors=12
):

    rows = points[:, 0]
    pixels = points[:, 1]

    lats = points[:, 2]
    lons = points[:, 3]

    # --------------------------------------------------------
    # Calculate approximate geographic distance
    # --------------------------------------------------------

    lon_scale = np.cos(
        np.radians(target_lat)
    )

    distances = np.sqrt(
        (lats - target_lat) ** 2
        +
        ((lons - target_lon) * lon_scale) ** 2
    )

    # --------------------------------------------------------
    # Select nearest geolocation points
    # --------------------------------------------------------

    indices = np.argsort(
        distances
    )[:n_neighbors]

    local_rows = rows[indices]
    local_pixels = pixels[indices]

    local_lats = lats[indices]
    local_lons = lons[indices]

    # --------------------------------------------------------
    # Normalize image coordinates
    # --------------------------------------------------------

    row_center = np.mean(
        local_rows
    )

    pixel_center = np.mean(
        local_pixels
    )

    row_scale = np.std(
        local_rows
    )

    pixel_scale = np.std(
        local_pixels
    )

    if row_scale == 0:

        row_scale = 1.0

    if pixel_scale == 0:

        pixel_scale = 1.0

    r = (
        local_rows - row_center
    ) / row_scale

    p = (
        local_pixels - pixel_center
    ) / pixel_scale

    # --------------------------------------------------------
    # Design matrix
    # --------------------------------------------------------

    A = np.column_stack([
        np.ones(len(indices)),
        r,
        p
    ])

    # --------------------------------------------------------
    # Least-squares latitude model
    # --------------------------------------------------------

    coef_lat, _, _, _ = np.linalg.lstsq(
        A,
        local_lats,
        rcond=None
    )

    # --------------------------------------------------------
    # Least-squares longitude model
    # --------------------------------------------------------

    coef_lon, _, _, _ = np.linalg.lstsq(
        A,
        local_lons,
        rcond=None
    )

    # --------------------------------------------------------
    # Solve for image coordinates
    # --------------------------------------------------------

    target_lat_centered = (
        target_lat - coef_lat[0]
    )

    target_lon_centered = (
        target_lon - coef_lon[0]
    )

    matrix = np.array([
        [
            coef_lat[1],
            coef_lat[2]
        ],
        [
            coef_lon[1],
            coef_lon[2]
        ]
    ])

    vector = np.array([
        target_lat_centered,
        target_lon_centered
    ])

    try:

        normalized_position = np.linalg.solve(
            matrix,
            vector
        )

    except np.linalg.LinAlgError:

        return None

    normalized_row = (
        normalized_position[0]
    )

    normalized_pixel = (
        normalized_position[1]
    )

    image_row = (
        normalized_row * row_scale
        + row_center
    )

    image_pixel = (
        normalized_pixel * pixel_scale
        + pixel_center
    )

    return (
        image_row,
        image_pixel
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 90)
print("PRECISE DARTIS → SENTINEL-1 PIXEL MAPPING")
print("=" * 90)


# ------------------------------------------------------------
# Load DARTIS catalogue
# ------------------------------------------------------------

df = pd.read_csv(
    CATALOGUE_FILE
)

print()
print(
    "Total DARTIS catalogue records:",
    len(df)
)


# ------------------------------------------------------------
# Get only scenes actually downloaded
# ------------------------------------------------------------

downloaded_scenes = get_downloaded_scenes()

print()
print(
    "Downloaded ZIP files:",
    len(downloaded_scenes)
)

for scene in downloaded_scenes:

    print(
        "  ",
        scene
    )


# ------------------------------------------------------------
# Keep only catalogue records belonging to downloaded scenes
# ------------------------------------------------------------

df = df[
    df["sentinel_id"].isin(
        downloaded_scenes
    )
].copy()


print()
print(
    "DARTIS records for downloaded scenes:",
    len(df)
)

print(
    "Scenes to process:",
    df["sentinel_id"].nunique()
)


# ------------------------------------------------------------
# Storage for results
# ------------------------------------------------------------

results = []


# ============================================================
# PROCESS EACH DOWNLOADED SCENE
# ============================================================

for scene in df["sentinel_id"].unique():

    print()
    print("=" * 90)
    print("SCENE")
    print(scene)
    print("=" * 90)


    # --------------------------------------------------------
    # Find ZIP
    # --------------------------------------------------------

    zip_path = find_zip(
        scene
    )

    if zip_path is None:

        print(
            "ERROR: ZIP not found"
        )

        continue


    print(
        "ZIP:",
        os.path.basename(zip_path)
    )


    # --------------------------------------------------------
    # Find VV annotation XML
    # --------------------------------------------------------

    xml_name = get_annotation_xml(
        zip_path
    )

    if xml_name is None:

        print(
            "ERROR: VV annotation XML not found"
        )

        continue


    print(
        "Annotation XML:"
    )

    print(
        xml_name
    )


    # --------------------------------------------------------
    # Read geolocation grid
    # --------------------------------------------------------

    points = read_geolocation_grid(
        zip_path,
        xml_name
    )


    print(
        "Geolocation points:",
        len(points)
    )


    if len(points) == 0:

        print(
            "ERROR: No geolocation points found"
        )

        continue


    # --------------------------------------------------------
    # Get DARTIS records for this scene
    # --------------------------------------------------------

    scene_df = df[
        df["sentinel_id"] == scene
    ]


    print()

    print(
        f"{'Patch':45s}"
        f"{'Lat':>11s}"
        f"{'Lon':>11s}"
        f"{'Row':>13s}"
        f"{'Pixel':>13s}"
    )

    print(
        "-" * 95
    )


    # ========================================================
    # PROCESS PATCHES
    # ========================================================

    for _, record in scene_df.iterrows():

        patch_name = record[
            "patch_name"
        ]

        target_lat = float(
            record[
                "patch_ul_lat"
            ]
        )

        target_lon = float(
            record[
                "patch_ul_lon"
            ]
        )


        # ----------------------------------------------------
        # Map geographic coordinate → image coordinate
        # ----------------------------------------------------

        mapped = map_geo_to_pixel(
            points,
            target_lat,
            target_lon,
            n_neighbors=12
        )


        if mapped is None:

            print(
                patch_name,
                "MAPPING FAILED"
            )

            continue


        image_row, image_pixel = mapped


        # ----------------------------------------------------
        # Print mapping
        # ----------------------------------------------------

        print(
            f"{patch_name:45s}"
            f"{target_lat:11.5f}"
            f"{target_lon:11.5f}"
            f"{image_row:13.2f}"
            f"{image_pixel:13.2f}"
        )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "patch_name":
                patch_name,

            "sentinel_id":
                scene,

            "label":
                record["label"],

            "subset":
                record["subset"],

            "patch_width":
                record["patch_width"],

            "patch_height":
                record["patch_height"],

            "patch_ul_lon":
                target_lon,

            "patch_ul_lat":
                target_lat,

            "image_row":
                image_row,

            "image_pixel":
                image_pixel,

            "obj_xmin":
                record["obj_xmin"],

            "obj_ymin":
                record["obj_ymin"],

            "obj_xmax":
                record["obj_xmax"],

            "obj_ymax":
                record["obj_ymax"]
        })


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(
    results
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 90)
print("MAPPING COMPLETE")
print("=" * 90)

print(
    "Mappings created:",
    len(result_df)
)

print()
print(
    "Output:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 90)