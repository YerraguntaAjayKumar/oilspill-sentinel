import zipfile
import glob
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

CATALOGUE = r"data\catalogue\dartis_catalogue.csv"
RAW_DIR = r"data\raw\train_batch_1"
OUTPUT = r"data\catalogue\dartis_pixel_mapping.csv"

TARGET_IDS = [
    "S1A_IW_GRDH_1SDV_20190925T155712_20190925T155737_029180_035045_0057.SAFE",
    "S1B_IW_GRDH_1SDV_20190306T154810_20190306T154835_015236_01C814_5AC6.SAFE",
]


def read_geolocation_grid(zip_path):

    with zipfile.ZipFile(zip_path, "r") as z:

        xml_name = None

        # Find the VV annotation XML, NOT calibration/noise XML
        for name in z.namelist():

            lower = name.lower()

            if (
                "/annotation/" in lower
                and "/calibration/" not in lower
                and "/noise/" not in lower
                and "-vv-" in lower
                and lower.endswith(".xml")
            ):
                xml_name = name
                break

        if xml_name is None:

            print("ERROR: VV annotation XML not found")

            print("\nAvailable annotation XML files:")

            for name in z.namelist():

                lower = name.lower()

                if "/annotation/" in lower and lower.endswith(".xml"):
                    print(name)

            raise RuntimeError(
                "Could not find Sentinel-1 VV annotation XML"
            )

        print("Annotation XML:")
        print(xml_name)

        xml_data = z.read(xml_name)

    root = ET.fromstring(xml_data)

    # Remove XML namespaces
    for element in root.iter():

        if "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]

    # Find geolocation grid
    grid_nodes = root.findall(".//geolocationGridPoint")

    print("Raw geolocation nodes:", len(grid_nodes))

    rows = []

    for point in grid_nodes:

        values = {}

        for child in point:
            values[child.tag] = child.text

        required = [
            "line",
            "pixel",
            "latitude",
            "longitude"
        ]

        if not all(x in values for x in required):
            continue

        rows.append({
            "line": float(values["line"]),
            "pixel": float(values["pixel"]),
            "lat": float(values["latitude"]),
            "lon": float(values["longitude"]),
        })

    return pd.DataFrame(rows)


def prepare_grid(grid):

    lines = np.sort(
        grid["line"].unique()
    )

    pixels = np.sort(
        grid["pixel"].unique()
    )

    print()
    print("Unique lines :", len(lines))
    print("Unique pixels:", len(pixels))
    print(
        "Grid size    :",
        len(lines),
        "x",
        len(pixels)
    )

    lat_grid = np.zeros(
        (len(lines), len(pixels))
    )

    lon_grid = np.zeros(
        (len(lines), len(pixels))
    )

    line_index = {
        value: i
        for i, value in enumerate(lines)
    }

    pixel_index = {
        value: i
        for i, value in enumerate(pixels)
    }

    for _, row in grid.iterrows():

        i = line_index[row["line"]]
        j = pixel_index[row["pixel"]]

        lat_grid[i, j] = row["lat"]
        lon_grid[i, j] = row["lon"]

    return lines, pixels, lat_grid, lon_grid


def find_surrounding_cell(
    lines,
    pixels,
    lat_grid,
    lon_grid,
    target_lat,
    target_lon
):

    best_i = None
    best_j = None
    best_distance = float("inf")

    for i in range(len(lines) - 1):

        for j in range(len(pixels) - 1):

            cell_lat = [
                lat_grid[i, j],
                lat_grid[i + 1, j],
                lat_grid[i, j + 1],
                lat_grid[i + 1, j + 1]
            ]

            cell_lon = [
                lon_grid[i, j],
                lon_grid[i + 1, j],
                lon_grid[i, j + 1],
                lon_grid[i + 1, j + 1]
            ]

            center_lat = np.mean(cell_lat)
            center_lon = np.mean(cell_lon)

            distance = (
                (center_lat - target_lat) ** 2
                +
                (center_lon - target_lon) ** 2
            )

            if distance < best_distance:

                best_distance = distance
                best_i = i
                best_j = j

    return best_i, best_j


def bilinear_mapping(
    lines,
    pixels,
    lat_grid,
    lon_grid,
    i,
    j,
    target_lat,
    target_lon
):

    lat00 = lat_grid[i, j]
    lat10 = lat_grid[i + 1, j]
    lat01 = lat_grid[i, j + 1]
    lat11 = lat_grid[i + 1, j + 1]

    lon00 = lon_grid[i, j]
    lon10 = lon_grid[i + 1, j]
    lon01 = lon_grid[i, j + 1]
    lon11 = lon_grid[i + 1, j + 1]

    line00 = lines[i]
    line10 = lines[i + 1]

    pixel00 = pixels[j]
    pixel01 = pixels[j + 1]

    lat_min = min(
        lat00,
        lat10,
        lat01,
        lat11
    )

    lat_max = max(
        lat00,
        lat10,
        lat01,
        lat11
    )

    lon_min = min(
        lon00,
        lon10,
        lon01,
        lon11
    )

    lon_max = max(
        lon00,
        lon10,
        lon01,
        lon11
    )

    if lat_max == lat_min:
        return None

    if lon_max == lon_min:
        return None

    u = (
        target_lat - lat_min
    ) / (
        lat_max - lat_min
    )

    v = (
        target_lon - lon_min
    ) / (
        lon_max - lon_min
    )

    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    image_line = (
        line00
        +
        u * (line10 - line00)
    )

    image_pixel = (
        pixel00
        +
        v * (pixel01 - pixel00)
    )

    return image_line, image_pixel


def main():

    print("=" * 90)
    print("DARTIS → SENTINEL-1 PIXEL MAPPING")
    print("=" * 90)

    df = pd.read_csv(CATALOGUE)

    df = df[
        df["sentinel_id"].isin(TARGET_IDS)
    ].copy()

    print()
    print("DARTIS records:", len(df))

    zip_files = glob.glob(
        os.path.join(
            RAW_DIR,
            "*.zip"
        )
    )

    zip_map = {}

    for zip_path in zip_files:

        name = os.path.basename(zip_path)

        if name.endswith(".SAFE.zip"):

            safe_name = name[:-4]

            zip_map[safe_name] = zip_path

    results = []

    for sentinel_id in TARGET_IDS:

        print()
        print("=" * 90)
        print("SCENE")
        print(sentinel_id)
        print("=" * 90)

        if sentinel_id not in zip_map:

            print("ZIP not found")
            continue

        zip_path = zip_map[sentinel_id]

        print("Reading geolocation grid...")

        grid = read_geolocation_grid(
            zip_path
        )

        print(
            "Geolocation points:",
            len(grid)
        )

        if len(grid) == 0:

            print(
                "ERROR: No geolocation points found."
            )

            continue

        lines, pixels, lat_grid, lon_grid = prepare_grid(
            grid
        )

        scene_df = df[
            df["sentinel_id"] == sentinel_id
        ]

        print()
        print(
            f"{'Patch':<45}"
            f"{'DARTIS Lon':>12}"
            f"{'DARTIS Lat':>12}"
            f"{'Image Row':>14}"
            f"{'Image Pixel':>15}"
        )

        print("-" * 100)

        for _, row in scene_df.iterrows():

            patch_name = row["patch_name"]

            target_lon = float(
                row["patch_ul_lon"]
            )

            target_lat = float(
                row["patch_ul_lat"]
            )

            i, j = find_surrounding_cell(
                lines,
                pixels,
                lat_grid,
                lon_grid,
                target_lat,
                target_lon
            )

            mapped = bilinear_mapping(
                lines,
                pixels,
                lat_grid,
                lon_grid,
                i,
                j,
                target_lat,
                target_lon
            )

            if mapped is None:

                print(
                    "Mapping failed:",
                    patch_name
                )

                continue

            image_line, image_pixel = mapped

            print(
                f"{patch_name:<45}"
                f"{target_lon:>12.5f}"
                f"{target_lat:>12.5f}"
                f"{image_line:>14.1f}"
                f"{image_pixel:>15.1f}"
            )

            results.append({

                "patch_name": patch_name,

                "sentinel_id": sentinel_id,

                "label": row["label"],

                "subset": row["subset"],

                "patch_width": row["patch_width"],

                "patch_height": row["patch_height"],

                "patch_ul_lon": target_lon,

                "patch_ul_lat": target_lat,

                "image_row": image_line,

                "image_pixel": image_pixel,

                "obj_xmin": row["obj_xmin"],

                "obj_ymin": row["obj_ymin"],

                "obj_xmax": row["obj_xmax"],

                "obj_ymax": row["obj_ymax"]

            })

    result_df = pd.DataFrame(results)

    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("=" * 90)
    print("MAPPING COMPLETE")
    print("=" * 90)

    print(
        "Mappings created:",
        len(result_df)
    )

    print()
    print("Output:")
    print(OUTPUT)


if __name__ == "__main__":
    main()