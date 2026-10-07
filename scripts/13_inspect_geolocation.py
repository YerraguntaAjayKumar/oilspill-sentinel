import zipfile
import glob
import os
import xml.etree.ElementTree as ET

zip_files = glob.glob(r"data\raw\train_batch_1\*.zip")

print("=" * 70)
print("SENTINEL-1 GEOLOCATION XML INSPECTION")
print("=" * 70)

for zip_path in zip_files:

    print()
    print("=" * 70)
    print(os.path.basename(zip_path))
    print("=" * 70)

    with zipfile.ZipFile(zip_path, "r") as z:

        # Find VV annotation XML
        vv_xml = None

        for name in z.namelist():
            lower = name.lower()

            if (
                "/annotation/" in lower
                and "/calibration/" not in lower
                and "-vv-" in lower
                and lower.endswith(".xml")
            ):
                vv_xml = name
                break

        if vv_xml is None:
            print("VV annotation XML not found")
            continue

        print("VV annotation:")
        print(vv_xml)

        # Read XML directly from ZIP
        xml_data = z.read(vv_xml)

        root = ET.fromstring(xml_data)

        # Remove XML namespace from tags
        for element in root.iter():
            if "}" in element.tag:
                element.tag = element.tag.split("}", 1)[1]

        # Find geolocationGrid
        geolocation = root.find(".//geolocationGrid")

        if geolocation is None:
            print("Geolocation grid NOT found")
            continue

        points = geolocation.findall(".//geolocationGridPoint")

        print()
        print("Geolocation grid found!")
        print("Number of grid points:", len(points))
        print()

        # Print first 5 points
        print("First geolocation points:")
        print("-" * 70)

        for point in points[:5]:

            values = {}

            for child in point:
                values[child.tag] = child.text

            print(values)

print()
print("=" * 70)
print("DONE")
print("=" * 70)