import os
import json
import time
import requests

# The 'championspreview' tag is the verified active format on Pikalytics servers.
# Revert to 'gen9championsvgc2026regma' once the explicit ruleset tag is pushed to their API.
FORMAT_TAG = "championspreview" 
USAGE_API = f"https://pikalytics.com/api/p/usage?p={FORMAT_TAG}"
DETAILS_API = "https://pikalytics.com/api/p/pokemon"
OUTPUT_DIR = "data/raw/pikalytics"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{FORMAT_TAG}_matrix.json")

def execute_extraction() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    usage_response = requests.get(USAGE_API)
    usage_response.raise_for_status()
    usage_data = usage_response.json()
    
    # Restrict to Top 50 to prevent rate-limit bans during initial pipeline validation
    top_meta = usage_data[:50] 
    compiled_matrix = {}

    for entity in top_meta:
        species_name = entity.get("name")
        params = {"l": "en", "p": FORMAT_TAG, "t": species_name}
        
        detail_response = requests.get(DETAILS_API, params=params)
        if detail_response.status_code == 200:
            compiled_matrix[species_name] = detail_response.json()
        else:
            print(f"Failed entity extraction: {species_name} | Status: {detail_response.status_code}")
            
        time.sleep(1.0) 

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(compiled_matrix, f)

    return OUTPUT_FILE

def analyze_extracted_data(filepath: str) -> None:
    print("--- Pikalytics Bronze Extraction Analysis ---")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError("Extraction failed. File not found.")
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Data Volume Validation
    entity_count = len(data.keys())
    print(f"Entities Extracted: {entity_count}")
    if entity_count < 20:
        print("CRITICAL FAILURE: Insufficient meta-game volume. N < 20.")
        return

    # Feature & Applicability Validation
    missing_evs = 0
    missing_teammates = 0
    
    for species, stats in data.items():
        # Validate Cross-Join Requirements
        spreads = stats.get("spreads", [])
        if not spreads or len(spreads) == 0:
            missing_evs += 1
            
        # Validate Bayesian Inference Requirements
        teammates = stats.get("teammates", [])
        if not teammates or len(teammates) == 0:
            missing_teammates += 1

    ev_deficit_rate = (missing_evs / entity_count) * 100
    teammate_deficit_rate = (missing_teammates / entity_count) * 100

    print(f"Entities lacking continuous EV distributions: {missing_evs} ({ev_deficit_rate:.2f}%)")
    print(f"Entities lacking adjacency matrices (Teammates): {missing_teammates} ({teammate_deficit_rate:.2f}%)")

    # Null & Threshold Check
    if ev_deficit_rate > 40.0:
        print("CRITICAL FAILURE: Silver ETL Relational Cross-Join blocked. Heuristic fallback limit exceeded.")
    elif teammate_deficit_rate > 40.0:
        print("CRITICAL FAILURE: Bayesian Inference Matrix blocked. Insufficient co-occurrence data.")
    else:
        print("VALIDATION SUCCESS: Payload structural integrity verified. Matrix cleared for Phase 1 Silver ETL.")

if __name__ == "__main__":
    matrix_path = execute_extraction()
    analyze_extracted_data(matrix_path)