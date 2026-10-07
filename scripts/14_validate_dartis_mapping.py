import zipfile
import glob
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


CATALOGUE = r"data\catalogue\dartis_catalogue.csv"
RAW_DIR = r"data\raw\train_batch_1"


TARGET_IDS = [
    "S1A_IW_GRDH_1SDV_20190925T155712_20190925T155737_029180_035045_0057.SAFE",
    "S1B_IW_GRDH_1SDV_20190306T154810_20190306T154835_015236_01C814_5AC6.SAFE",
]


def read_geolocation_grid(zip_path):
    """
    Read Sentinel-1 VV geolocation grid directly from ZIP.
    No extraction of the large SAFE product is required.
    """

    with zipfile.ZipFile(zip_path, "r") as z:

        xml_name = None

        for name in z.namelist():
            lower = name.lower()

            if (
                "/annotation/" in lower
                and "/calibration/" not in lower
                and "-vv-" in lower
                and lower.endswith(".xml")
            ):
                xml_name = name
                break

        if xml_name is None:
            raise RuntimeError("VV annotation XML not found")

        xml_data = z.read(xml_name)

    root = ET.fromstring(xml_data)

    # Remove XML namespaces
    for element in root.iter():
        if "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]

    points = root.findall(".//geolocationGridPoint")

    rows = []

    for point in points:

        values = {}

        for child in point:
            values[child.tag] = child.text

        rows.append({
            "line": float(values["line"]),
            "pixel": float(values["pixel"]),
            "lat": float(values["latitude"]),
            "lon": float(values["longitude"]),
        })

    return pd.DataFrame(rows)


def nearest_geolocation_point(grid, lat, lon):
    """
    Find the nearest Sentinel-1 geolocation tie point.
    Used only as a first validation.
    """

    distance = (
        (grid["lat"].values - lat) ** 2
        +
        (grid["lon"].values - lon) ** 2
    )

    index = np.argmin(distance)

    return grid.iloc[index], np.sqrt(distance[index])


def main():

    print("=" * 80)
    print("DARTIS → SENTINEL-1 LOCATION VALIDATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load catalogue
    # ---------------------------------------------------------

    df = pd.read_csv(CATALOGUE)

    df = df[df["sentinel_id"].isin(TARGET_IDS)].copy()

    print()
    print("DARTIS records:", len(df))

    # ---------------------------------------------------------
    # Find ZIP files
    # ---------------------------------------------------------

    zip_files = glob.glob(os.path.join(RAW_DIR, "*.zip"))

    zip_map = {}

    for zip_path in zip_files:

        name = os.path.basename(zip_path)

        if name.endswith(".SAFE.zip"):
            safe_name = name[:-4]
            zip_map[safe_name] = zip_path

    # ---------------------------------------------------------
    # Process each Sentinel scene
    # ---------------------------------------------------------

    for sentinel_id in TARGET_IDS:

        print()
        print("=" * 80)
        print("SCENE")
        print(sentinel_id)
        print("=" * 80)

        if sentinel_id not in zip_map:
            print("ZIP not found")
            continue

        zip_path = zip_map[sentinel_id]

        print("Reading geolocation grid...")

        grid = read_geolocation_grid(zip_path)

        print("Geolocation points:", len(grid))

        print(
            "Latitude range:",
            f"{grid['lat'].min():.6f}",
            "to",
            f"{grid['lat'].max():.6f}"
        )

        print(
            "Longitude range:",
            f"{grid['lon'].min():.6f}",
            "to",
            f"{grid['lon'].max():.6f}"
        )

        # -----------------------------------------------------
        # DARTIS patches for this scene
        # -----------------------------------------------------

        scene_df = df[df["sentinel_id"] == sentinel_id]

        print()
        print("DARTIS patches:", len(scene_df))

        print()

        print(
            f"{'Patch':<45}"
            f"{'DARTIS Lon':>12}"
            f"{'DARTIS Lat':>12}"
            f"{'Nearest Line':>14}"
            f"{'Nearest Pixel':>15}"
            f"{'Geo Error':>12}"
        )

        print("-" * 115)

        for _, row in scene_df.iterrows():

            lon = float(row["patch_ul_lon"])
            lat = float(row["patch_ul_lat"])

            nearest, error = nearest_geolocation_point(
                grid,
                lat,
                lon
            )

            patch_name = row["patch_name"]

            print(
                f"{patch_name:<45}"
                f"{lon:>12.5f}"
                f"{lat:>12.5f}"
                f"{nearest['line']:>14.0f}"
                f"{nearest['pixel']:>15.0f}"
                f"{error:>12.5f}"
            )

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()