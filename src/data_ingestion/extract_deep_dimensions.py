import os
import pandas as pd
import requests
import time
from typing import Dict, Any, List

SILVER_PATH = "data/processed/silver_standings.parquet"
OUTPUT_STATS = "data/processed/dimension_stats.parquet"
OUTPUT_MOVES = "data/processed/dimension_moves.parquet"

def get_pokeapi_data(endpoint: str, item_id: str | int) -> Dict[str, Any]:
    url = f"https://pokeapi.co/api/v2/{endpoint}/{item_id}/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {}

def execute_extraction() -> None:
    if not os.path.exists(SILVER_PATH):
        raise FileNotFoundError(f"Dependency missing: {SILVER_PATH}. Execute Stage 3 normalization prior to dimension extraction.")

    df = pd.read_parquet(SILVER_PATH)
    
    unique_pokemon = df['pokeapi_id'].dropna().unique()
    move_cols = ['move_1', 'move_2', 'move_3', 'move_4']
    unique_moves = pd.unique(df[move_cols].values.ravel('K'))
    unique_moves = [m for m in unique_moves if m and str(m).lower() != 'none']

    stats_records: List[Dict[str, Any]] = []
    for pid in unique_pokemon:
        if pid == -1: 
            continue
        data = get_pokeapi_data("pokemon", pid)
        if not data: 
            continue
        
        base_stats = {stat['stat']['name']: stat['base_stat'] for stat in data.get('stats', [])}
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
        time.sleep(0.05) 

    moves_records: List[Dict[str, Any]] = []
    for move in unique_moves:
        move_formatted = str(move).lower().replace(" ", "-")
        data = get_pokeapi_data("move", move_formatted)
        if not data: 
            continue
        
        moves_records.append({
            "move_name": move,
            "base_power": data.get("power"),
            "accuracy": data.get("accuracy"),
            "type": data.get("type", {}).get("name", "None"),
            "damage_class": data.get("damage_class", {}).get("name", "None")
        })
        time.sleep(0.05)

    pd.DataFrame(stats_records).to_parquet(OUTPUT_STATS, index=False)
    pd.DataFrame(moves_records).to_parquet(OUTPUT_MOVES, index=False)

if __name__ == "__main__":
    execute_extraction()