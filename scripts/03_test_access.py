import requests
import os

print("======================================")
print("COPERNICUS DATA SPACE ACCESS TEST")
print("======================================")

# --------------------------------------
# 1. Check token file
# --------------------------------------

token_file = "access_token.txt"

if not os.path.exists(token_file):
    print("\nERROR: access_token.txt not found.")
    print("Run: python scripts\\02_get_token.py")
    exit()

with open(token_file, "r") as f:
    token = f.read().strip()

if not token:
    print("\nERROR: access_token.txt is empty.")
    print("Run: python scripts\\02_get_token.py")
    exit()

print("\nAccess token found.")
print("Testing CDSE catalogue access...")


# --------------------------------------
# 2. CDSE Catalogue API
# --------------------------------------

url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

headers = {
    "Authorization": f"Bearer {token}"
}

params = {
    "$top": "1"
}


# --------------------------------------
# 3. Send request
# --------------------------------------

try:

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    print("Status:", response.status_code)

except requests.exceptions.RequestException as e:

    print("\nERROR: Could not connect to CDSE.")
    print(e)
    exit()


# --------------------------------------
# 4. Check result
# --------------------------------------

if response.status_code == 200:

    print()
    print("======================================")
    print("COPERNICUS ACCESS SUCCESSFUL!")
    print("======================================")

    data = response.json()

    products = data.get("value", [])

    if products:

        product = products[0]

        print("\nTest product:")
        print("Name:", product.get("Name"))
        print("ID:", product.get("Id"))

    else:

        print("\nConnection successful, but no products were returned.")


elif response.status_code == 401:

    print()
    print("======================================")
    print("AUTHENTICATION FAILED")
    print("======================================")

    print("Your access token is invalid or expired.")
    print("Run:")
    print("python scripts\\02_get_token.py")


elif response.status_code == 403:

    print()
    print("======================================")
    print("ACCESS FORBIDDEN")
    print("======================================")

    print("Your account/token does not have permission.")


else:

    print()
    print("======================================")
    print("CDSE REQUEST FAILED")
    print("======================================")

    print("Status:", response.status_code)
    print(response.text)