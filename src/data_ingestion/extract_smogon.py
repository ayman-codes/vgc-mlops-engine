import os
import json
import time
import requests

BASE_URL = "https://www.smogon.com/stats/"
FORMAT_TAG = "gen9championsvgc2026regmabo3-1760"
TARGET_MONTHS = ["2026-05"]
OUTPUT_DIR = "data/raw/smogon"

def execute_longitudinal_extraction() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for month in TARGET_MONTHS:
        target_url = f"{BASE_URL}{month}/chaos/{FORMAT_TAG}.json"
        output_file = os.path.join(OUTPUT_DIR, f"{FORMAT_TAG}_{month}.json")
        
        response = requests.get(target_url)
        if response.status_code == 200:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(response.json(), f)
            print(f"Extraction successful: {month}")
        elif response.status_code == 404:
            print(f"Data missing for {month}. Target URL returned 404: {target_url}")
        else:
            response.raise_for_status()
            
        time.sleep(1.0)

if __name__ == "__main__":
    execute_longitudinal_extraction()