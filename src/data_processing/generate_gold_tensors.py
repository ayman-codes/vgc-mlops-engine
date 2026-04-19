import os
import pandas as pd

SMOGON_PATH = "data/processed/smogon_normalized.parquet"
STATS_PATH = "data/processed/dimension_stats.parquet"
MOVES_PATH = "data/processed/dimension_moves.parquet"
OUTPUT_PATH = "data/processed/gold_tensors.parquet"

def execute_tensor_engineering() -> None:
    df_smogon = pd.read_parquet(SMOGON_PATH)
    df_stats = pd.read_parquet(STATS_PATH)
    df_moves = pd.read_parquet(MOVES_PATH)

    ev_cols = ['hp_ev', 'atk_ev', 'def_ev', 'spa_ev', 'spd_ev', 'spe_ev']
    df_smogon[ev_cols] = pd.DataFrame(df_smogon['evs'].tolist(), index=df_smogon.index)
    df_smogon = df_smogon.drop(columns=['evs'])

    move_cols = ['move_1', 'move_2', 'move_3', 'move_4']
    df_smogon[move_cols] = pd.DataFrame(df_smogon['moves'].tolist(), index=df_smogon.index)
    df_smogon = df_smogon.drop(columns=['moves'])

    df_gold = df_smogon.merge(df_stats, on='pokeapi_id', how='left')

    for i in range(1, 5):
        df_gold = df_gold.merge(df_moves, left_on=f'move_{i}', right_on='move_name', how='left')
        df_gold = df_gold.rename(columns={
            'base_power': f'm{i}_bp', 
            'accuracy': f'm{i}_acc',
            'type': f'm{i}_type', 
            'damage_class': f'm{i}_class'
        }).drop(columns=['move_name', f'move_{i}'])

    numeric_fill = {}
    for i in range(1, 5):
        numeric_fill[f'm{i}_bp'] = 0.0
        numeric_fill[f'm{i}_acc'] = 100.0
    df_gold = df_gold.fillna(value=numeric_fill)

    categorical_cols = [
        'item', 'ability', 'nature', 'type_1', 'type_2'
    ] + [f'm{i}_type' for i in range(1, 5)] + [f'm{i}_class' for i in range(1, 5)]
    
    df_gold = pd.get_dummies(df_gold, columns=categorical_cols, dummy_na=True, dtype='float32')
    
    for col in df_gold.select_dtypes(include=['int64', 'float64', 'int32']).columns:
        df_gold[col] = df_gold[col].astype('float32')

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_gold.to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    execute_tensor_engineering()