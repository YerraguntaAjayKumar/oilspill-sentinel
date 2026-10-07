import requests
import getpass

print("================================")
print("COPERNICUS DATA SPACE LOGIN")
print("================================")

username = input("Copernicus email: ")
password = getpass.getpass("Copernicus password: ")

url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

data = {
    "grant_type": "password",
    "client_id": "cdse-public",
    "username": username,
    "password": password
}

print("\nRequesting access token...")

response = requests.post(
    url,
    data=data,
    timeout=30
)

print("Status:", response.status_code)

if response.status_code == 200:

    result = response.json()

    token = result.get("access_token")

    if not token:
        print("ERROR: No access token returned.")
        print(result)
        exit()

    with open("access_token.txt", "w") as f:
        f.write(token)

    print()
    print("==============================")
    print("TOKEN CREATED SUCCESSFULLY")
    print("==============================")

else:

    print()
    print("TOKEN REQUEST FAILED")
    print(response.text)