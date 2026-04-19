import json
import os
import pandas as pd
from typing import Dict, List, Any
from pydantic import BaseModel, Field, field_validator, ValidationError
from thefuzz import fuzz

INPUT_PATH = "data/raw/smogon/gen9vgc2024regf-1760.json"
POKEAPI_PATH = "data/raw/pokeapi_base.json"
OUTPUT_PATH = "data/processed/smogon_normalized.parquet"

class CompetitiveBuild(BaseModel):
    pokeapi_id: int
    item: str
    ability: str
    nature: str
    evs: List[int] = Field(min_length=6, max_length=6)
    moves: List[str] = Field(min_length=4, max_length=4)

    @field_validator('evs')
    @classmethod
    def validate_ev_total(cls, evs: List[int]) -> List[int]:
        if sum(evs) > 510:
            raise ValueError("EV allocation exceeds mathematical bound (510).")
        return evs

def build_poke_map() -> tuple[Dict[str, int], List[str]]:
    with open(POKEAPI_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    poke_map = {p["name"]: int(p["url"].rstrip("/").split("/")[-1]) for p in data.get("results", [])}
    return poke_map, list(poke_map.keys())

def resolve_entity(query_id: str, poke_map: Dict[str, int], poke_keys: List[str]) -> int:
    query = query_id.lower().replace("-", " ")
    best_match = None
    best_score = -1
    
    for key in poke_keys:
        target = key.replace("-", " ")
        score = fuzz.token_set_ratio(query, target)
        if score > best_score:
            best_score = score
            best_match = key
            
    if best_score >= 85 and best_match:
        return poke_map[best_match]
    return -1

def parse_ev_string(spread_str: str) -> Dict[str, Any]:
    parts = spread_str.split(':')
    if len(parts) != 2:
        return {"nature": "Hardy", "evs": [0, 0, 0, 0, 0, 0]}
    return {"nature": parts[0], "evs": [int(x) for x in parts[1].split('/')]}

def execute_smogon_etl() -> None:
    poke_map, poke_keys = build_poke_map()
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chaos_data = json.load(f)

    valid_records = []
    for species, data in chaos_data.get("data", {}).items():
        pokeapi_id = resolve_entity(species, poke_map, poke_keys)
        if pokeapi_id == -1:
            continue

        top_items = sorted(data.get("Items", {}).items(), key=lambda i: i[1], reverse=True)
        top_abilities = sorted(data.get("Abilities", {}).items(), key=lambda a: a[1], reverse=True)
        top_moves = sorted(data.get("Moves", {}).items(), key=lambda m: m[1], reverse=True)
        top_spreads = sorted(data.get("Spreads", {}).items(), key=lambda s: s[1], reverse=True)

        if not top_items or not top_abilities or len(top_moves) < 4 or not top_spreads:
            continue

        primary_moves = [m[0] for m in top_moves[:4]]
        spread_data = parse_ev_string(top_spreads[0][0])

        try:
            build = CompetitiveBuild(
                pokeapi_id=pokeapi_id,
                item=top_items[0][0],
                ability=top_abilities[0][0],
                nature=spread_data["nature"],
                evs=spread_data["evs"],
                moves=primary_moves
            )
            valid_records.append(build.model_dump())
        except ValidationError:
            continue

    df = pd.DataFrame(valid_records)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    execute_smogon_etl()