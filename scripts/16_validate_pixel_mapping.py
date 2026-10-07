import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAPPING_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "catalogue",
    "dartis_pixel_mapping.csv"
)

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "train_batch_1"
)


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


def read_geolocation_grid(zip_path, xml_name):
    with zipfile.ZipFile(zip_path, "r") as z:
        xml_data = z.read(xml_name)

    root = ET.fromstring(xml_data)

    points = []

    for node in root.iter():
        if node.tag.endswith("geolocationGridPoint"):
            line = None
            pixel = None
            lat = None
            lon = None

            for child in node:
                tag = child.tag.split("}")[-1]

                if tag == "line":
                    line = float(child.text)

                elif tag == "pixel":
                    pixel = float(child.text)

                elif tag == "latitude":
                    lat = float(child.text)

                elif tag == "longitude":
                    lon = float(child.text)

            if None not in (line, pixel, lat, lon):
                points.append((line, pixel, lat, lon))

    return np.array(points)


def bilinear_value(x, y, x0, x1, y0, y1, v00, v01, v10, v11):

    if x1 == x0 or y1 == y0:
        return None

    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)

    value = (
        (1 - tx) * (1 - ty) * v00
        + tx * (1 - ty) * v01
        + (1 - tx) * ty * v10
        + tx * ty * v11
    )

    return value


def pixel_to_geo(points, row, col):

    lines = np.unique(points[:, 0])
    pixels = np.unique(points[:, 1])

    lat_grid = np.zeros((len(lines), len(pixels)))
    lon_grid = np.zeros((len(lines), len(pixels)))

    for p in points:
        i = np.where(lines == p[0])[0][0]
        j = np.where(pixels == p[1])[0][0]

        lat_grid[i, j] = p[2]
        lon_grid[i, j] = p[3]

    # Find surrounding line indices
    i = np.searchsorted(lines, row) - 1
    j = np.searchsorted(pixels, col) - 1

    i = max(0, min(i, len(lines) - 2))
    j = max(0, min(j, len(pixels) - 2))

    y0 = lines[i]
    y1 = lines[i + 1]

    x0 = pixels[j]
    x1 = pixels[j + 1]

    lat = bilinear_value(
        col,
        row,
        x0,
        x1,
        y0,
        y1,
        lat_grid[i, j],
        lat_grid[i, j + 1],
        lat_grid[i + 1, j],
        lat_grid[i + 1, j + 1]
    )

    lon = bilinear_value(
        col,
        row,
        x0,
        x1,
        y0,
        y1,
        lon_grid[i, j],
        lon_grid[i, j + 1],
        lon_grid[i + 1, j],
        lon_grid[i + 1, j + 1]
    )

    return lat, lon


def find_zip(scene):

    path = os.path.join(RAW_DIR, scene.replace(".SAFE", ".zip"))

    if os.path.exists(path):
        return path

    # fallback: search by scene ID
    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith(".zip"):
            if scene.replace(".SAFE", "").lower() in filename.lower():
                return os.path.join(RAW_DIR, filename)

    return None


print("=" * 90)
print("DARTIS → SENTINEL-1 PIXEL MAPPING VALIDATION")
print("=" * 90)

df = pd.read_csv(MAPPING_FILE)

print()
print("Mapping records:", len(df))
print("Columns:")
print(df.columns.tolist())

print()


# Automatically find important columns
def find_column(possible):
    for col in df.columns:
        if col.lower() in possible:
            return col

    for col in df.columns:
        low = col.lower()

        for name in possible:
            if name in low:
                return col

    return None


scene_col = find_column(["sentinel_id", "scene"])
lon_col = find_column(["dartis_lon", "patch_ul_lon"])
lat_col = find_column(["dartis_lat", "patch_ul_lat"])
row_col = find_column(["image_row", "row"])
pixel_col = find_column(["image_pixel", "pixel", "column"])

print("Detected columns:")
print("Scene :", scene_col)
print("Lon   :", lon_col)
print("Lat   :", lat_col)
print("Row   :", row_col)
print("Pixel :", pixel_col)

if None in (scene_col, lon_col, lat_col, row_col, pixel_col):
    raise RuntimeError(
        "Could not automatically identify required columns. "
        "Check the column names printed above."
    )


results = []

for scene in df[scene_col].unique():

    print()
    print("=" * 90)
    print("SCENE")
    print(scene)
    print("=" * 90)

    zip_path = find_zip(scene)

    if zip_path is None:
        print("ERROR: ZIP not found")
        continue

    print("ZIP:", os.path.basename(zip_path))

    xml_name = get_annotation_xml(zip_path)

    if xml_name is None:
        print("ERROR: VV annotation XML not found")
        continue

    print("Annotation XML:")
    print(xml_name)

    points = read_geolocation_grid(zip_path, xml_name)

    print("Geolocation points:", len(points))

    scene_df = df[df[scene_col] == scene]

    for _, record in scene_df.iterrows():

        dartis_lon = float(record[lon_col])
        dartis_lat = float(record[lat_col])

        row = float(record[row_col])
        pixel = float(record[pixel_col])

        calculated_lat, calculated_lon = pixel_to_geo(
            points,
            row,
            pixel
        )

        lat_error = calculated_lat - dartis_lat
        lon_error = calculated_lon - dartis_lon

        # Approximate distance in meters
        lat_m = lat_error * 111320
        lon_m = lon_error * 111320 * np.cos(
            np.radians(dartis_lat)
        )

        error_m = np.sqrt(lat_m ** 2 + lon_m ** 2)

        results.append({
            "scene": scene,
            "patch_name": record.get("patch_name", ""),
            "dartis_lat": dartis_lat,
            "dartis_lon": dartis_lon,
            "image_row": row,
            "image_pixel": pixel,
            "calculated_lat": calculated_lat,
            "calculated_lon": calculated_lon,
            "lat_error_deg": lat_error,
            "lon_error_deg": lon_error,
            "error_meters": error_m
        })


result_df = pd.DataFrame(results)

print()
print("=" * 90)
print("VALIDATION RESULTS")
print("=" * 90)

if len(result_df) > 0:

    print(
        result_df[
            [
                "patch_name",
                "image_row",
                "image_pixel",
                "error_meters"
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 90)
    print("ERROR SUMMARY")
    print("=" * 90)

    print("Number of mappings:", len(result_df))
    print("Mean error       :", round(result_df["error_meters"].mean(), 2), "m")
    print("Median error     :", round(result_df["error_meters"].median(), 2), "m")
    print("Maximum error    :", round(result_df["error_meters"].max(), 2), "m")
    print("Minimum error    :", round(result_df["error_meters"].min(), 2), "m")

    print()

    good = result_df[result_df["error_meters"] <= 100]
    warning = result_df[
        (result_df["error_meters"] > 100)
        & (result_df["error_meters"] <= 500)
    ]
    bad = result_df[result_df["error_meters"] > 500]

    print("≤ 100 m     :", len(good))
    print("100–500 m   :", len(warning))
    print("> 500 m     :", len(bad))

    output_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "catalogue",
        "dartis_mapping_validation.csv"
    )

    result_df.to_csv(output_file, index=False)

    print()
    print("Validation file:")
    print(output_file)

else:
    print("No validation results were generated.")

print()
print("=" * 90)
print("VALIDATION COMPLETE")
print("=" * 90)