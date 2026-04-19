import json
import pandas as pd
import os
import glob
from typing import Any, Dict, List
from thefuzz import fuzz

POKEAPI_PATH = "data/raw/pokeapi_base.json"
LIMITLESS_DIR = "data/raw/limitless/"
OUTPUT_PATH = "data/processed/limitless_validation.parquet"

COLUMNS = [
    "player", "placing", "wins", "losses", "ties", "slot", 
    "limitless_species_id", "pokeapi_id", "item", "ability", 
    "tera_type", "move_1", "move_2", "move_3", "move_4"
]

STATIC_OVERRIDE = {
    "basculegion": 902,
    "urshifu": 892,
    "urshifu-rapid-strike": 10191,
    "tornadus": 641,
    "thundurus": 642,
    "landorus": 645,
    "enamorus": 905,
    "indeedee": 876,
    "ogerpon": 1011,
    "calyrex": 898,
    "calyrex-shadow": 10194,
    "calyrex-ice": 10193
}

FUZZY_THRESHOLD = 85

def resolve_entity(limitless_id: str, poke_map: Dict[str, int], poke_keys: List[str]) -> int:
    if limitless_id in STATIC_OVERRIDE:
        return STATIC_OVERRIDE[limitless_id]
        
    query = limitless_id.replace("-", " ")
    best_match = None
    best_score = -1
    
    for key in poke_keys:
        target = key.replace("-", " ")
        score = fuzz.token_set_ratio(query, target)
        if score > best_score:
            best_score = score
            best_match = key
            
    if best_score >= FUZZY_THRESHOLD and best_match:
        return poke_map[best_match]
        
    return -1

def extract_players(data_node: Any) -> List[Dict[str, Any]]:
    players = []
    for obj in data_node:
        for p in obj.get("data", []):
            players.append(p)
    return players

def execute_normalization() -> None:
    if not os.path.exists(POKEAPI_PATH):
        raise FileNotFoundError(f"Missing PokeAPI dependency: {POKEAPI_PATH}")
        
    with open(POKEAPI_PATH, "r", encoding="utf-8") as f:
        pokeapi_data = json.load(f)
        
    poke_map = {p["name"]: int(p["url"].rstrip("/").split("/")[-1]) for p in pokeapi_data.get("results", [])}
    poke_keys = list(poke_map.keys())
    
    normalized_records = []
    files = glob.glob(os.path.join(LIMITLESS_DIR, "*.json"))
    
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        players = extract_players(raw_data)
        
        for player in players:
            player_name = player.get("player")
            placing = player.get("placing")
            record = player.get("record", {})
            decklist = player.get("decklist", [])
            
            for slot, pkmn in enumerate(decklist):
                limitless_id = pkmn.get("id", "")
                pokeapi_id = resolve_entity(limitless_id, poke_map, poke_keys)
                
                attacks = pkmn.get("attacks", [])
                attacks += [None] * (4 - len(attacks))
                
                normalized_records.append({
                    "player": player_name,
                    "placing": placing if placing is not None else -1,
                    "wins": record.get("wins", 0),
                    "losses": record.get("losses", 0),
                    "ties": record.get("ties", 0),
                    "slot": slot + 1,
                    "limitless_species_id": limitless_id,
                    "pokeapi_id": pokeapi_id,
                    "item": pkmn.get("item"),
                    "ability": pkmn.get("ability"),
                    "tera_type": pkmn.get("tera", "None"),
                    "move_1": attacks[0],
                    "move_2": attacks[1],
                    "move_3": attacks[2],
                    "move_4": attacks[3]
                })

    df = pd.DataFrame(normalized_records, columns=COLUMNS)
    df = df.drop_duplicates(subset=['player', 'limitless_species_id', 'slot']).reset_index(drop=True)

    for col in ['move_2', 'move_3', 'move_4']:
        df[col] = df[col].fillna('None')
        
    df['tera_type'] = df['tera_type'].fillna('None')
    df = df[df['pokeapi_id'] != -1]
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    execute_normalization()