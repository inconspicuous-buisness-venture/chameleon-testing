import requests

url = "https://api.gptzero.me/v2/predict/text"

payload = {
    "document": "Located in East Africa, Uganda is a landlocked nation known for its stunning natural beauty, diverse wildlife, and rich cultural heritage. With its lush landscapes, vibrant cities, and warm-hearted people, Uganda has rightfully earned the moniker The Pearl of Africa. This informational paper aims to provide an overview of Uganda, exploring its geography, history, culture, economy, and tourist attractions.",
    "multilingual": False
}
headers = {
    "x-api-key": "",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())