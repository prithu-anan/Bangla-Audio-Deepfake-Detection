import requests
import os

API_KEY = os.getenv("ELEVENLABS_API_KEY")

resp = requests.get(
    "https://api.elevenlabs.io/v1/voices",
    headers={"xi-api-key": API_KEY}
)

for v in resp.json()["voices"]:
    print(f"\"{v['name']}\",")


