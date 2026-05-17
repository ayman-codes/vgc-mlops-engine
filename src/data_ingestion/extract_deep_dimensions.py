import os
import json
import time
import pandas as pd
import requests
from typing import Dict, Any, List, cast

POKEAPI_BASE_PATH = "data/raw/pokeapi_base.json"
OUTPUT_STATS = "data/processed/dimension_stats.parquet"
OUTPUT_MOVES = "data/processed/dimension_moves.parquet"

def get_api_data(url: str, retries: int = 3) -> Dict[str, Any]:
    """Fetches API data with exponential backoff to prevent silent drops."""
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return cast(Dict[str, Any], response.json())
            elif response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            else:
                response.raise_for_status()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise RuntimeError(f"API extraction failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return {}

def execute_extraction() -> None:
    if not os.path.exists(POKEAPI_BASE_PATH):
        raise FileNotFoundError(f"Missing dependency: {POKEAPI_BASE_PATH}. Execute Stage 2.1.")

    with open(POKEAPI_BASE_PATH, "r", encoding="utf-8") as f:
        base_data = json.load(f)
        
    pokemon_results = base_data.get("results", [])
    stats_records: List[Dict[str, Any]] = []
    
    # Universal Species Extraction
    for p in pokemon_results:
        pid = int(p["url"].rstrip("/").split("/")[-1])
        data = get_api_data(p["url"])
        if not data:
            continue
            
        base_stats = {s['stat']['name']: s['base_stat'] for s in data.get('stats', [])}
        types = [t['type']['name'] for t in data.get('types', [])]
        
        stats_records.append({
            "pokeapi_id": pid,
            "hp": base_stats.get("hp", 0),
            "atk": base_stats.get("attack", 0),
            "def": base_stats.get("defense", 0),
            "spa": base_stats.get("special-attack", 0),
            "spd": base_stats.get("special-defense", 0),
            "spe": base_stats.get("speed", 0),
            "type_1": types[0] if len(types) > 0 else "None",
            "type_2": types[1] if len(types) > 1 else "None"
        })
        time.sleep(0.25)  # Strictly enforce 0.6-second delay to comply with 100 req/min limit 
        
    # Universal Move Extraction
    # Expanded limit to ensure full coverage of potential out-of-vocabulary moves
    moves_base = get_api_data("https://pokeapi.co/api/v2/move?limit=2000")
    move_results = moves_base.get("results", [])
    moves_records: List[Dict[str, Any]] = []
    
    for m in move_results:
        # Dimension Table Compression: Strip hyphens and spaces for relational join compatibility
        move_name = m["name"].replace("-", "").replace(" ", "").lower()
        data = get_api_data(m["url"])
        if not data:
            continue
            
        moves_records.append({
            "move_name": move_name,
            "base_power": data.get("power"),
            "accuracy": data.get("accuracy"),
            "type": data.get("type", {}).get("name", "None"),
            "damage_class": data.get("damage_class", {}).get("name", "None")
        })
        time.sleep(0.25)

    os.makedirs(os.path.dirname(OUTPUT_STATS), exist_ok=True)
    pd.DataFrame(stats_records).to_parquet(OUTPUT_STATS, index=False)
    pd.DataFrame(moves_records).to_parquet(OUTPUT_MOVES, index=False)

if __name__ == "__main__":
    execute_extraction()