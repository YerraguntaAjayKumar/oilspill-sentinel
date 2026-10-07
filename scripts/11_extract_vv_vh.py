import zipfile
import os
import glob
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_NUMBER = 1

RAW_DIR = os.path.join(
    "data",
    "raw",
    f"train_batch_{BATCH_NUMBER}"
)

OUTPUT_DIR = os.path.join(
    "data",
    "processed",
    f"train_batch_{BATCH_NUMBER}"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 11 - SENTINEL-1 VV + VH EXTRACTION")
print("=" * 70)


# ============================================================
# CHECK RAW DIRECTORY
# ============================================================

if not os.path.exists(RAW_DIR):

    print()
    print("ERROR: Raw directory not found:")
    print(os.path.abspath(RAW_DIR))
    print()

    raise SystemExit(1)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIND ZIP FILES
# ============================================================

zip_files = glob.glob(
    os.path.join(
        RAW_DIR,
        "*.zip"
    )
)


print()
print(
    "Found",
    len(zip_files),
    "Sentinel-1 ZIP files."
)


if len(zip_files) == 0:

    print()
    print("ERROR: No ZIP files found.")
    print()

    raise SystemExit(1)


# ============================================================
# COUNTERS
# ============================================================

scenes_processed = 0
vv_count = 0
vh_count = 0


# ============================================================
# PROCESS EACH ZIP
# ============================================================

for zip_path in zip_files:

    zip_name = os.path.basename(
        zip_path
    )

    print()
    print("-" * 70)
    print(
        "SCENE:",
        zip_name
    )
    print("-" * 70)


    # --------------------------------------------------------
    # OPEN ZIP
    # --------------------------------------------------------

    try:

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as z:

            print(
                "Inspecting ZIP contents..."
            )


            # ------------------------------------------------
            # FIND TIFF MEASUREMENT FILES
            # ------------------------------------------------

            measurement_files = []

            for name in z.namelist():

                name_lower = name.lower()

                if (
                    "/measurement/" in name_lower
                    and
                    name_lower.endswith(".tiff")
                ):

                    measurement_files.append(
                        name
                    )


            # ------------------------------------------------
            # FIND VV
            # ------------------------------------------------

            vv_files = []

            for name in measurement_files:

                name_lower = name.lower()

                if "-vv-" in name_lower:

                    vv_files.append(
                        name
                    )


            # ------------------------------------------------
            # FIND VH
            # ------------------------------------------------

            vh_files = []

            for name in measurement_files:

                name_lower = name.lower()

                if "-vh-" in name_lower:

                    vh_files.append(
                        name
                    )


            print(
                "Measurement TIFF files:",
                len(measurement_files)
            )

            print(
                "VV files found:",
                len(vv_files)
            )

            print(
                "VH files found:",
                len(vh_files)
            )


            # ------------------------------------------------
            # OUTPUT SCENE NAME
            # ------------------------------------------------

            scene_name = zip_name

            if scene_name.lower().endswith(".zip"):

                scene_name = scene_name[:-4]


            # ------------------------------------------------
            # CREATE SCENE OUTPUT DIRECTORY
            # ------------------------------------------------

            scene_output_dir = os.path.join(
                OUTPUT_DIR,
                scene_name
            )

            os.makedirs(
                scene_output_dir,
                exist_ok=True
            )


            # ------------------------------------------------
            # EXTRACT VV
            # ------------------------------------------------

            for vv_file in vv_files:

                output_file = os.path.join(
                    scene_output_dir,
                    os.path.basename(vv_file)
                )

                print()
                print(
                    "Extracting VV:"
                )

                print(
                    os.path.basename(vv_file)
                )


                with z.open(vv_file) as source:

                    with open(
                        output_file,
                        "wb"
                    ) as target:

                        shutil.copyfileobj(
                            source,
                            target
                        )


                file_size_mb = (
                    os.path.getsize(output_file)
                    / (1024 ** 2)
                )


                print(
                    "Saved:",
                    output_file
                )

                print(
                    "Size:",
                    round(file_size_mb, 2),
                    "MB"
                )


                vv_count += 1


            # ------------------------------------------------
            # EXTRACT VH
            # ------------------------------------------------

            for vh_file in vh_files:

                output_file = os.path.join(
                    scene_output_dir,
                    os.path.basename(vh_file)
                )

                print()
                print(
                    "Extracting VH:"
                )

                print(
                    os.path.basename(vh_file)
                )


                with z.open(vh_file) as source:

                    with open(
                        output_file,
                        "wb"
                    ) as target:

                        shutil.copyfileobj(
                            source,
                            target
                        )


                file_size_mb = (
                    os.path.getsize(output_file)
                    / (1024 ** 2)
                )


                print(
                    "Saved:",
                    output_file
                )

                print(
                    "Size:",
                    round(file_size_mb, 2),
                    "MB"
                )


                vh_count += 1


            # ------------------------------------------------
            # WARNING IF MISSING
            # ------------------------------------------------

            if len(vv_files) == 0:

                print()
                print(
                    "WARNING: No VV measurement file found."
                )


            if len(vh_files) == 0:

                print()
                print(
                    "WARNING: No VH measurement file found."
                )


            # ------------------------------------------------
            # SCENE SUCCESS
            # ------------------------------------------------

            if (
                len(vv_files) > 0
                and
                len(vh_files) > 0
            ):

                print()
                print(
                    "VV + VH extraction successful."
                )


            scenes_processed += 1


    except zipfile.BadZipFile:

        print()
        print(
            "ERROR: ZIP file is corrupted."
        )

    except Exception as e:

        print()
        print(
            "ERROR while processing ZIP:"
        )

        print(e)


# ============================================================
# EXTRACTION SUMMARY
# ============================================================

print()
print("=" * 70)
print("EXTRACTION SUMMARY")
print("=" * 70)

print(
    "Scenes processed :",
    scenes_processed
)

print(
    "VV files         :",
    vv_count
)

print(
    "VH files         :",
    vh_count
)

print()
print(
    "Output directory:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)


# ============================================================
# NEXT STEP
# ============================================================

print()
print("IMPORTANT:")
print(
    "The original ZIP files have NOT been deleted."
)

print(
    "We will verify the extracted GeoTIFFs before deleting anything."
)

print()
print("Next step:")

print(
    "Inspect the dimensions, CRS and georeferencing "
    "of the VV/VH TIFFs."
)