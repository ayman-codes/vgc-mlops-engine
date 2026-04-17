import json
import pandas as pd
import os

POKEAPI_PATH = "vgc_data/pokeapi_base.json"
LIMITLESS_PATH = "vgc_data/limitless_vgc.json"
OUTPUT_PATH = "vgc_data/silver_standings.parquet"

def extract_players(data_node):
    if isinstance(data_node, dict):
        if "decklist" in data_node or "teamlist" in data_node:
            return [data_node]
        elif "data" in data_node:
            return extract_players(data_node["data"])
    elif isinstance(data_node, list):
        out = []
        for item in data_node:
            out.extend(extract_players(item))
        return out
    return []

def execute_normalization():
    # 1. Build Dimension Map
    with open(POKEAPI_PATH, "r", encoding="utf-8") as f:
        poke_data = json.load(f)

    poke_map = {}
    for p in poke_data.get("results", []):
        poke_id = int(p["url"].rstrip("/").split("/")[-1])
        poke_map[p["name"].lower()] = poke_id

    # 2. Extract and Normalize Standings
    with open(LIMITLESS_PATH, "r", encoding="utf-8") as f:
        raw_standings = json.load(f)

    normalized_records = []
    players = extract_players(raw_standings)

    for player_data in players:
        player_name = player_data.get("player") or player_data.get("name")
        placing = player_data.get("placing")
        record = player_data.get("record", {})
        decklist = player_data.get("decklist") or player_data.get("teamlist", [])
        
        for slot, pkmn in enumerate(decklist):
            limitless_id = pkmn.get("id", "").lower()
            pokeapi_id = poke_map.get(limitless_id, -1) # -1 flags unresolved forms (e.g., specific Megas)
            
            attacks = pkmn.get("attacks", [])
            attacks += [None] * (4 - len(attacks))
            
            normalized_records.append({
                "player": player_name,
                "placing": placing,
                "wins": record.get("wins", 0),
                "losses": record.get("losses", 0),
                "ties": record.get("ties", 0),
                "slot": slot + 1,
                "limitless_species_id": limitless_id,
                "pokeapi_id": pokeapi_id,
                "item": pkmn.get("item"),
                "ability": pkmn.get("ability"),
                "tera_type": pkmn.get("tera"),
                "move_1": attacks[0],
                "move_2": attacks[1],
                "move_3": attacks[2],
                "move_4": attacks[3]
            })

    # 3. Persist Silver Layer
    df = pd.DataFrame(normalized_records)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    
    print("=== SILVER NORMALIZATION COMPLETE ===")
    print(f"Total rows structured: {len(df)}")
    print(f"Unmapped entities (-1 IDs): {len(df[df['pokeapi_id'] == -1])}")

if __name__ == "__main__":
    execute_normalization()