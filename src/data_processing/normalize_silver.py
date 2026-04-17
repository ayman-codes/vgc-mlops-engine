import json
import pandas as pd
import os
import glob
from typing import Any
from thefuzz import process

POKEAPI_PATH = "data/raw/pokeapi_base.json"
LIMITLESS_DIR = "data/raw/limitless/"
OUTPUT_PATH = "data/processed/silver_standings.parquet"

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

def extract_players(data_node: Any) -> list[dict[str, Any]]:
    if isinstance(data_node, dict):
        if "decklist" in data_node or "teamlist" in data_node:
            return [data_node]
        elif "data" in data_node:
            return extract_players(data_node["data"])
    elif isinstance(data_node, list):
        out: list[dict[str, Any]] = []
        for item in data_node:
            out.extend(extract_players(item))
        return out
    return []

def resolve_entity(limitless_id: Any, poke_map: dict[str, int], poke_keys: list[str]) -> int:
    if not limitless_id:
        return -1
        
    limitless_id_str = str(limitless_id).lower()
    
    if limitless_id_str in poke_map:
        return poke_map[limitless_id_str]
        
    if limitless_id_str in STATIC_OVERRIDE:
        return STATIC_OVERRIDE[limitless_id_str]
        
    match, score = process.extractOne(limitless_id_str, poke_keys)
    if score >= FUZZY_THRESHOLD:
        return poke_map[match]
        
    return -1

def execute_normalization() -> None:
    with open(POKEAPI_PATH, "r", encoding="utf-8") as f:
        poke_data = json.load(f)

    poke_map = {}
    for p in poke_data.get("results", []):
        poke_id = int(p["url"].rstrip("/").split("/")[-1])
        poke_map[p["name"].lower()] = poke_id
        
    poke_keys = list(poke_map.keys())

    partitioned_files = glob.glob(os.path.join(LIMITLESS_DIR, "*.json"))
    
    raw_standings = []
    for file_path in partitioned_files:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_standings.extend(json.load(f))

    normalized_records = []
    players = extract_players(raw_standings)

    for player_data in players:
        player_name = player_data.get("player") or player_data.get("name")
        placing = player_data.get("placing")
        record = player_data.get("record", {})
        decklist = player_data.get("decklist") or player_data.get("teamlist", [])
        
        invalid_format_flag = False
        for pkmn in decklist:
            item = str(pkmn.get("item", "")).lower()
            if "ite" in item and item not in ["eviolite", "meteorite"]: 
                invalid_format_flag = True
                break
                
        if invalid_format_flag:
            continue
        
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
    
    df = df[df['pokeapi_id'] != -1].reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    execute_normalization()