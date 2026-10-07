import zipfile
import glob
import os

zip_files = glob.glob(r"data\raw\train_batch_1\*.zip")

print("=" * 60)
print("SENTINEL-1 ZIP METADATA INSPECTION")
print("=" * 60)

for zip_path in zip_files:

    print()
    print("=" * 60)
    print(os.path.basename(zip_path))
    print("=" * 60)

    with zipfile.ZipFile(zip_path, "r") as z:

        xml_files = []

        for name in z.namelist():
            if name.lower().endswith(".xml"):
                xml_files.append(name)

        print("XML files found:", len(xml_files))
        print()

        for name in xml_files[:30]:
            print(name)

        print()
        print("Important files:")

        for name in z.namelist():
            lower_name = name.lower()

            if (
                "manifest.safe" in lower_name
                or "/annotation/" in lower_name
            ):
                print(name)

print()
print("=" * 60)
print("DONE")
print("=" * 60)