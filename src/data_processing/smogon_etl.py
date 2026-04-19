import json
import os
import pandas as pd
from typing import Dict, List, Any
from pydantic import BaseModel, Field, field_validator, ValidationError

INPUT_PATH = "data/raw/smogon/gen9vgc2024regf-1760.json"
OUTPUT_PATH = "data/processed/smogon_normalized.parquet"

class CompetitiveBuild(BaseModel):
    species: str
    item: str
    ability: str
    nature: str
    evs: List[int] = Field(min_length=6, max_length=6)
    moves: List[str] = Field(min_length=4, max_length=4)

    @field_validator('evs')
    @classmethod
    def validate_ev_total(cls, evs: List[int]) -> List[int]:
        total = sum(evs)
        if total > 510:
            raise ValueError(f"EV allocation exceeds mathematical bound (510): {total}")
        if any(ev < 0 or ev > 252 for ev in evs):
            raise ValueError("Individual EV stat out of bounds (0-252).")
        return evs

    @field_validator('moves')
    @classmethod
    def validate_move_count(cls, moves: List[str]) -> List[str]:
        valid_moves = [m for m in moves if m and m.lower() != 'none']
        if len(valid_moves) != 4:
            raise ValueError(f"Invalid move selection count: {len(valid_moves)}")
        return moves

def parse_ev_string(spread_str: str) -> Dict[str, Any]:
    parts = spread_str.split(':')
    if len(parts) != 2:
        return {"nature": "Hardy", "evs": [0, 0, 0, 0, 0, 0]}
    
    nature = parts[0]
    evs = [int(x) for x in parts[1].split('/')]
    return {"nature": nature, "evs": evs}

def execute_smogon_etl() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Source JSON missing: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chaos_data = json.load(f)

    valid_records = []
    
    for species, data in chaos_data.get("data", {}).items():
        top_items = sorted(data.get("Items", {}).items(), key=lambda item: item[1], reverse=True)
        top_abilities = sorted(data.get("Abilities", {}).items(), key=lambda item: item[1], reverse=True)
        top_moves = sorted(data.get("Moves", {}).items(), key=lambda item: item[1], reverse=True)
        top_spreads = sorted(data.get("Spreads", {}).items(), key=lambda item: item[1], reverse=True)

        if not top_items or not top_abilities or len(top_moves) < 4 or not top_spreads:
            continue

        primary_item = top_items[0][0]
        primary_ability = top_abilities[0][0]
        primary_moves = [m[0] for m in top_moves[:4]]
        spread_data = parse_ev_string(top_spreads[0][0])

        try:
            build = CompetitiveBuild(
                species=species,
                item=primary_item,
                ability=primary_ability,
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