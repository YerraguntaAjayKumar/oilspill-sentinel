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

MAPPING_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "catalogue",
    "dartis_pixel_mapping_precise.csv"
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
    "dartis_precise_mapping_validation.csv"
)


# ============================================================
# FIND ZIP
# ============================================================

def find_zip(scene):

    filename = scene + ".zip"

    path = os.path.join(
        RAW_DIR,
        filename
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
# READ GEOLOCATION GRID
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
# BILINEAR INTERPOLATION
# ============================================================

def interpolate_geo(
    points,
    row,
    pixel
):

    rows = points[:, 0]
    pixels = points[:, 1]
    lats = points[:, 2]
    lons = points[:, 3]

    unique_rows = np.unique(rows)
    unique_pixels = np.unique(pixels)

    # --------------------------------------------------------
    # Find surrounding row values
    # --------------------------------------------------------

    row_low_candidates = unique_rows[
        unique_rows <= row
    ]

    row_high_candidates = unique_rows[
        unique_rows >= row
    ]

    if len(row_low_candidates) == 0:
        return None

    if len(row_high_candidates) == 0:
        return None

    row_low = row_low_candidates[-1]
    row_high = row_high_candidates[0]

    # --------------------------------------------------------
    # Find surrounding pixel values
    # --------------------------------------------------------

    pixel_low_candidates = unique_pixels[
        unique_pixels <= pixel
    ]

    pixel_high_candidates = unique_pixels[
        unique_pixels >= pixel
    ]

    if len(pixel_low_candidates) == 0:
        return None

    if len(pixel_high_candidates) == 0:
        return None

    pixel_low = pixel_low_candidates[-1]
    pixel_high = pixel_high_candidates[0]

    # --------------------------------------------------------
    # Get four surrounding points
    # --------------------------------------------------------

    def get_point(r, p):

        mask = (
            (rows == r)
            &
            (pixels == p)
        )

        indices = np.where(mask)[0]

        if len(indices) == 0:
            return None

        idx = indices[0]

        return (
            lats[idx],
            lons[idx]
        )

    p11 = get_point(
        row_low,
        pixel_low
    )

    p12 = get_point(
        row_low,
        pixel_high
    )

    p21 = get_point(
        row_high,
        pixel_low
    )

    p22 = get_point(
        row_high,
        pixel_high
    )

    if None in (
        p11,
        p12,
        p21,
        p22
    ):

        return None

    lat11, lon11 = p11
    lat12, lon12 = p12
    lat21, lon21 = p21
    lat22, lon22 = p22

    # --------------------------------------------------------
    # Handle exact row/pixel cases
    # --------------------------------------------------------

    if row_high == row_low:

        row_fraction = 0.0

    else:

        row_fraction = (
            row - row_low
        ) / (
            row_high - row_low
        )

    if pixel_high == pixel_low:

        pixel_fraction = 0.0

    else:

        pixel_fraction = (
            pixel - pixel_low
        ) / (
            pixel_high - pixel_low
        )

    # --------------------------------------------------------
    # Bilinear interpolation
    # --------------------------------------------------------

    lat_top = (
        lat11
        +
        pixel_fraction
        *
        (lat12 - lat11)
    )

    lat_bottom = (
        lat21
        +
        pixel_fraction
        *
        (lat22 - lat21)
    )

    latitude = (
        lat_top
        +
        row_fraction
        *
        (lat_bottom - lat_top)
    )

    lon_top = (
        lon11
        +
        pixel_fraction
        *
        (lon12 - lon11)
    )

    lon_bottom = (
        lon21
        +
        pixel_fraction
        *
        (lon22 - lon21)
    )

    longitude = (
        lon_top
        +
        row_fraction
        *
        (lon_bottom - lon_top)
    )

    return latitude, longitude


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_meters(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371000.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = np.radians(
        lat2 - lat1
    )

    dlon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    c = (
        2
        *
        np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a)
        )
    )

    return earth_radius * c


# ============================================================
# MAIN
# ============================================================

print("=" * 90)
print("VALIDATING PRECISE DARTIS PIXEL MAPPING")
print("=" * 90)


# ------------------------------------------------------------
# Load mapping
# ------------------------------------------------------------

df = pd.read_csv(
    MAPPING_FILE
)

print()
print(
    "Mappings:",
    len(df)
)


results = []


# ============================================================
# PROCESS EACH SCENE
# ============================================================

for scene in df[
    "sentinel_id"
].unique():

    print()
    print("=" * 90)
    print("SCENE")
    print(scene)
    print("=" * 90)


    zip_path = find_zip(
        scene
    )

    if zip_path is None:

        print(
            "ERROR: ZIP not found"
        )

        continue


    xml_name = get_annotation_xml(
        zip_path
    )

    if xml_name is None:

        print(
            "ERROR: annotation XML not found"
        )

        continue


    points = read_geolocation_grid(
        zip_path,
        xml_name
    )


    print(
        "Geolocation points:",
        len(points)
    )


    scene_df = df[
        df["sentinel_id"] == scene
    ]


    print()

    print(
        f"{'Patch':45s}"
        f"{'Error(m)':>12s}"
        f"{'Original Lat':>14s}"
        f"{'Calculated Lat':>16s}"
        f"{'Original Lon':>14s}"
        f"{'Calculated Lon':>16s}"
    )

    print(
        "-" * 120
    )


    # ========================================================
    # VALIDATE EACH PATCH
    # ========================================================

    for _, record in scene_df.iterrows():

        patch_name = record[
            "patch_name"
        ]

        original_lat = float(
            record[
                "patch_ul_lat"
            ]
        )

        original_lon = float(
            record[
                "patch_ul_lon"
            ]
        )

        row = float(
            record[
                "image_row"
            ]
        )

        pixel = float(
            record[
                "image_pixel"
            ]
        )


        calculated = interpolate_geo(
            points,
            row,
            pixel
        )


        if calculated is None:

            print(
                patch_name,
                "VALIDATION FAILED"
            )

            continue


        calculated_lat = calculated[0]
        calculated_lon = calculated[1]


        # ----------------------------------------------------
        # Geographic error
        # ----------------------------------------------------

        error_m = haversine_meters(
            original_lat,
            original_lon,
            calculated_lat,
            calculated_lon
        )


        print(
            f"{patch_name:45s}"
            f"{error_m:12.2f}"
            f"{original_lat:14.6f}"
            f"{calculated_lat:16.6f}"
            f"{original_lon:14.6f}"
            f"{calculated_lon:16.6f}"
        )


        results.append({

            "patch_name":
                patch_name,

            "sentinel_id":
                scene,

            "image_row":
                row,

            "image_pixel":
                pixel,

            "original_lat":
                original_lat,

            "original_lon":
                original_lon,

            "calculated_lat":
                calculated_lat,

            "calculated_lon":
                calculated_lon,

            "error_meters":
                error_m
        })


# ============================================================
# SAVE VALIDATION
# ============================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("VALIDATION SUMMARY")
print("=" * 90)


if len(result_df) > 0:

    errors = result_df[
        "error_meters"
    ]


    print(
        "Mappings validated:",
        len(errors)
    )

    print(
        "Mean error:",
        f"{errors.mean():.2f} m"
    )

    print(
        "Median error:",
        f"{errors.median():.2f} m"
    )

    print(
        "Minimum error:",
        f"{errors.min():.2f} m"
    )

    print(
        "Maximum error:",
        f"{errors.max():.2f} m"
    )


    print()
    print(
        "Within 100 m:",
        (errors <= 100).sum()
    )

    print(
        "100–500 m:",
        (
            (errors > 100)
            &
            (errors <= 500)
        ).sum()
    )

    print(
        "Above 500 m:",
        (errors > 500).sum()
    )


print()
print(
    "Validation output:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 90)