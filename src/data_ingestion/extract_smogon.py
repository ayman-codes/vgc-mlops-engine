import os
import json
import requests
from urllib.error import HTTPError

# Configuration targets. Modify TARGET_MONTH and FORMAT_TAG to isolate specific rulesets.
BASE_URL = "https://www.smogon.com/stats/"
TARGET_MONTH = "2024-04" 
FORMAT_TAG = "gen9vgc2024regf-1760" 
OUTPUT_DIR = "data/raw/smogon"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{FORMAT_TAG}.json")

def execute_extraction() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_url = f"{BASE_URL}{TARGET_MONTH}/chaos/{FORMAT_TAG}.json"
    
    response = requests.get(target_url)
    if response.status_code != 200:
        raise HTTPError(target_url, response.status_code, f"Extraction failed. Target missing: {target_url}", response.headers, None) # type: ignore
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(response.json(), f)

if __name__ == "__main__":
    execute_extraction()