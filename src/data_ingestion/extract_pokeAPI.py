import requests
import json
import os

OUTPUT_DIR = "./vgc_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pokeapi_base.json")
URL = "https://pokeapi.co/api/v2/pokemon?limit=10000"
HEADERS = {"User-Agent": "vgc-mlops-engine/1.0 (Data Engineering Portfolio Project)"}

def execute_extraction() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    response = requests.get(URL, headers=HEADERS)
    
    if response.status_code == 200:
        with open(OUTPUT_FILE, "w") as f:   
            json.dump(response.json(), f)
    else:
        raise ConnectionError(f"Extraction failed: {response.status_code}")

if __name__ == "__main__":
    execute_extraction()