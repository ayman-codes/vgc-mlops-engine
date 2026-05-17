import pandas as pd
import json
from typing import Dict, Any

LIMITLESS_PATH = "data/processed/limitless_discrete.parquet"
SMOGON_PATH = "data/processed/smogon_normalized.parquet"
POKEAPI_PATH = "data/raw/pokeapi_base.json"
OUTPUT_PATH = "data/processed/hydrated_rosters.parquet"
HEURISTIC_LIMIT = 0.40

class DataDeficitError(Exception):
    pass

def map_species_to_id(species_series: pd.Series, pokeapi_data: Dict[str, Any]) -> pd.Series:
    poke_map = {p["name"].replace("-", ""): int(p["url"].rstrip("/").split("/")[-1]) for p in pokeapi_data.get("results", [])}
    return species_series.map(poke_map).fillna(-1).astype(int)

def execute_hydration() -> None:
    df_limitless = pd.read_parquet(LIMITLESS_PATH)
    
    # Load PokeAPI lookup map to resolve Smogon IDs
    with open(POKEAPI_PATH, "r", encoding="utf-8") as f:
        pokeapi_data = json.load(f)
        
    df_limitless['pokeapi_id'] = map_species_to_id(df_limitless['species'], pokeapi_data)
    
    # Load Smogon continuous variables
    try:
        df_smogon = pd.read_parquet(SMOGON_PATH)
    except FileNotFoundError:
        # Fallback empty dataframe structure if Smogon ETL was not executed locally
        df_smogon = pd.DataFrame(columns=['pokeapi_id', 'nature', 'evs'])

    # Aggregate Smogon to top EV spread per species
    if not df_smogon.empty:
        df_smogon_top = df_smogon.groupby('pokeapi_id').first().reset_index()
    else:
        df_smogon_top = pd.DataFrame(columns=['pokeapi_id', 'nature', 'evs'])

    # Relational Cross-Join
    df_hydrated = pd.merge(df_limitless, df_smogon_top[['pokeapi_id', 'nature', 'evs']], on='pokeapi_id', how='left')

    # Heuristic Fallback Application
    total_entities = len(df_hydrated)
    heuristic_count = df_hydrated['evs'].isna().sum()
    
    # Fallback values
    df_hydrated['nature'] = df_hydrated['nature'].fillna('Hardy')
    df_hydrated['evs'] = df_hydrated['evs'].apply(lambda x: x if isinstance(x, list) else [84, 84, 84, 84, 84, 90])

    heuristic_rate = heuristic_count / total_entities

    print(f"Entities processed: {total_entities}")
    print(f"Heuristic fallbacks applied: {heuristic_count} ({heuristic_rate:.2%})")

    if heuristic_rate > HEURISTIC_LIMIT:
        raise DataDeficitError(f"CRITICAL FAILURE: Heuristic fallback rate ({heuristic_rate:.2%}) exceeds 40% threshold. Hydration blocked.")

    df_hydrated.to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    execute_hydration()