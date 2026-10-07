import pandas as pd
import requests
import os
import sys
import shutil
import time
import zipfile


# ============================================================
# CONFIGURATION
# ============================================================

TOKEN_FILE = "access_token.txt"

BATCH_DIR = r"data\scene_selection\batches"

RAW_BASE_DIR = r"data\raw"

DOWNLOAD_BASE_URL = (
    "https://download.dataspace.copernicus.eu"
    "/odata/v1/Products"
)

MAX_DOWNLOAD_GB = 4.0

MAX_RETRIES = 3


# ============================================================
# GET BATCH NUMBER
# ============================================================

if len(sys.argv) < 2:
    print()
    print("Usage:")
    print("python scripts\\10_download_batch.py 1")
    print()
    sys.exit(1)

batch_number = int(sys.argv[1])


# ============================================================
# BATCH FILE
# ============================================================

batch_file = os.path.join(
    BATCH_DIR,
    f"train_batch_{batch_number}.csv"
)

if not os.path.exists(batch_file):
    print()
    print("ERROR: Batch file not found:")
    print(batch_file)
    print()
    sys.exit(1)


# ============================================================
# READ BATCH
# ============================================================

batch = pd.read_csv(batch_file)

if len(batch) == 0:
    print("ERROR: Batch contains no scenes.")
    sys.exit(1)


# ============================================================
# CALCULATE SIZE
# ============================================================

total_size_gb = batch["size_gb"].sum()


print()
print("==============================================")
print("CDSE BATCH DOWNLOAD")
print("==============================================")
print("Batch:", batch_number)
print("Number of scenes:", len(batch))
print(
    "Expected download size:",
    round(total_size_gb, 2),
    "GB"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if total_size_gb > MAX_DOWNLOAD_GB:
    print()
    print("ERROR: Batch is larger than")
    print(MAX_DOWNLOAD_GB, "GB.")
    print("Download cancelled.")
    sys.exit(1)


# ============================================================
# CHECK TOKEN
# ============================================================

if not os.path.exists(TOKEN_FILE):
    print()
    print("ERROR: access_token.txt was not found.")
    print("Expected:")
    print(os.path.abspath(TOKEN_FILE))
    sys.exit(1)


with open(TOKEN_FILE, "r") as f:
    token = f.read().strip()


if not token:
    print("ERROR: access_token.txt is empty.")
    sys.exit(1)


# ============================================================
# CHECK DISK SPACE
# ============================================================

total, used, free = shutil.disk_usage(".")

free_gb = free / (1024 ** 3)

print()
print("----------------------------------------------")
print("DISK SPACE")
print("----------------------------------------------")
print("Free space:", round(free_gb, 2), "GB")

required_space = total_size_gb + 1.0

if free_gb < required_space:
    print()
    print("ERROR: Not enough free disk space.")
    print("Required:", round(required_space, 2), "GB")
    print("Available:", round(free_gb, 2), "GB")
    sys.exit(1)


# ============================================================
# CREATE RAW DIRECTORY
# ============================================================

raw_dir = os.path.join(
    RAW_BASE_DIR,
    f"train_batch_{batch_number}"
)

os.makedirs(raw_dir, exist_ok=True)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {token}"
})


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_product(product_id, product_name, output_file):

    url = f"{DOWNLOAD_BASE_URL}({product_id})/$value"

    for attempt in range(1, MAX_RETRIES + 1):

        print()
        print("----------------------------------------------")
        print(
            f"DOWNLOAD ATTEMPT {attempt}/{MAX_RETRIES}"
        )
        print("----------------------------------------------")
        print("Product:", product_name)
        print("Product ID:", product_id)
        print()

        # Remove old incomplete file
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception as e:
                print("ERROR removing old file:")
                print(e)
                return False

        try:

            # ------------------------------------------------
            # REQUEST
            # ------------------------------------------------

            response = session.get(
                url,
                stream=True,
                timeout=120
            )

            print("HTTP status:", response.status_code)

            # ------------------------------------------------
            # HTTP ERROR
            # ------------------------------------------------

            if response.status_code != 200:

                print()
                print("DOWNLOAD FAILED")

                try:
                    print(response.text[:1000])
                except Exception:
                    pass

                response.close()

                time.sleep(5)
                continue

            # ------------------------------------------------
            # EXPECTED SIZE
            # ------------------------------------------------

            content_length = response.headers.get(
                "Content-Length"
            )

            if content_length:
                total_bytes = int(content_length)
            else:
                total_bytes = 0

            print(
                "Expected response size:",
                round(
                    total_bytes / (1024 ** 3),
                    2
                ),
                "GB"
            )

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            downloaded = 0
            start_time = time.time()

            with open(output_file, "wb") as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue

                    f.write(chunk)
                    downloaded += len(chunk)

                    elapsed = time.time() - start_time

                    if elapsed > 0:
                        speed = (
                            downloaded
                            / elapsed
                            / (1024 * 1024)
                        )
                    else:
                        speed = 0

                    downloaded_gb = (
                        downloaded / (1024 ** 3)
                    )

                    if total_bytes > 0:

                        percent = (
                            downloaded
                            / total_bytes
                            * 100
                        )

                        print(
                            f"\rProgress: "
                            f"{percent:6.2f}% | "
                            f"{downloaded_gb:.2f} GB | "
                            f"{speed:.2f} MB/s",
                            end=""
                        )

                    else:

                        print(
                            f"\rDownloaded: "
                            f"{downloaded_gb:.2f} GB | "
                            f"{speed:.2f} MB/s",
                            end=""
                        )

            response.close()

            print()

            # ------------------------------------------------
            # CHECK ACTUAL SIZE
            # ------------------------------------------------

            actual_size = os.path.getsize(output_file)

            print(
                "Actual file size:",
                round(
                    actual_size / (1024 ** 3),
                    2
                ),
                "GB"
            )

            # ------------------------------------------------
            # CHECK CONTENT LENGTH
            # ------------------------------------------------

            if (
                total_bytes > 0
                and actual_size != total_bytes
            ):

                print()
                print("ERROR: Download is incomplete.")
                print("Expected:", total_bytes, "bytes")
                print("Received:", actual_size, "bytes")
                print("Retrying...")

                time.sleep(5)
                continue

            # ------------------------------------------------
            # CHECK ZIP
            # ------------------------------------------------

            print()
            print("Checking ZIP integrity...")

            if not zipfile.is_zipfile(output_file):

                print(
                    "ERROR: Downloaded file is NOT a valid ZIP."
                )

                print("Retrying...")

                time.sleep(5)
                continue

            # ------------------------------------------------
            # TEST ZIP CONTENT
            # ------------------------------------------------

            print("Testing ZIP contents...")

            try:

                with zipfile.ZipFile(
                    output_file,
                    "r"
                ) as z:

                    bad_file = z.testzip()

                if bad_file is not None:

                    print()
                    print(
                        "ERROR: ZIP contains corrupted data."
                    )

                    print(
                        "Corrupted file:",
                        bad_file
                    )

                    print("Retrying...")

                    time.sleep(5)
                    continue

            except Exception as e:

                print()
                print("ERROR while testing ZIP:")
                print(e)
                print("Retrying...")

                time.sleep(5)
                continue

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            print()
            print("==============================================")
            print("DOWNLOAD VERIFIED SUCCESSFULLY")
            print("==============================================")
            print("Valid ZIP: YES")
            print("File:", output_file)

            return True

        except Exception as e:

            print()
            print("ERROR while downloading:")
            print(e)
            print("Retrying...")

            time.sleep(5)

    # --------------------------------------------------------
    # ALL RETRIES FAILED
    # --------------------------------------------------------

    print()
    print("==============================================")
    print("DOWNLOAD FAILED")
    print("==============================================")
    print(
        "Maximum retries reached:",
        MAX_RETRIES
    )

    if os.path.exists(output_file):

        try:
            os.remove(output_file)
            print("Incomplete file removed.")
        except Exception:
            pass

    return False


# ============================================================
# DOWNLOAD EACH SCENE
# ============================================================

successful = []
failed = []

for index, row in batch.iterrows():

    product_id = row["product_id"]
    product_name = row["product_name"]
    sentinel_name = row["sentinel_name"]

    output_file = os.path.join(
        raw_dir,
        sentinel_name + ".zip"
    )

    print()
    print("==============================================")
    print(
        f"SCENE {index + 1} / {len(batch)}"
    )
    print("==============================================")

    # --------------------------------------------------------
    # CHECK EXISTING FILE
    # --------------------------------------------------------

    if os.path.exists(output_file):

        print("Existing file found:")
        print(output_file)

        print()
        print("Checking existing ZIP...")

        if zipfile.is_zipfile(output_file):

            print("Existing ZIP is valid.")

            successful.append(
                sentinel_name
            )

            continue

        else:

            print("Existing ZIP is INVALID.")
            print("It will be downloaded again.")

            try:
                os.remove(output_file)
            except Exception as e:

                print(
                    "ERROR removing invalid file:"
                )

                print(e)

                failed.append(
                    sentinel_name
                )

                continue

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    success = download_product(
        product_id,
        product_name,
        output_file
    )

    if success:

        successful.append(
            sentinel_name
        )

    else:

        failed.append(
            sentinel_name
        )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print()
print("==============================================")
print("DOWNLOAD SUMMARY")
print("==============================================")

print(
    "Successful:",
    len(successful)
)

print(
    "Failed:",
    len(failed)
)


# ============================================================
# SUCCESSFUL
# ============================================================

if successful:

    print()
    print("Successfully downloaded:")

    for name in successful:
        print("   ", name)


# ============================================================
# FAILED
# ============================================================

if failed:

    print()
    print("Failed downloads:")

    for name in failed:
        print("   ", name)


# ============================================================
# RAW DIRECTORY
# ============================================================

print()
print("Raw data directory:")
print(os.path.abspath(raw_dir))


# ============================================================
# NEXT STEP
# ============================================================

print()
print("==============================================")
print("NEXT STEP")
print("==============================================")

if failed:

    print("Some downloads failed.")
    print("Do NOT run VV/VH extraction yet.")

else:

    print("Batch download completed successfully.")
    print("All ZIP files passed integrity checks.")
    print()
    print(
        "Next: extract VV + VH from "
        "the Sentinel-1 data."
    )