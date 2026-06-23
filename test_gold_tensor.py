import duckdb
import pandas as pd
import time
import os

# --- PATH CONFIGURATION ---
GOLD_TENSORS_PATH = "data/processed/gold_tensors.parquet"
# Update this path to point to your parsed BC logs or transition Parquet
TRANSITIONS_PATH = "data/processed/bc_aggro.parquet" # Or the master 17M file

def analyze_gold_tensors():
    print("--- GOLD TENSORS ANALYSIS ---")
    if not os.path.exists(GOLD_TENSORS_PATH):
        print(f"Error: {GOLD_TENSORS_PATH} not found.")
        return

    df = pd.read_parquet(GOLD_TENSORS_PATH)
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    print(f"Total Unique Pokémon (Rows): {len(df)}")
    print(f"Feature Dimension per Pokémon (D): {len(df.columns)}")
    print(f"Data Types: {df.dtypes.unique()}")
    print(f"Memory Footprint: {memory_mb:.2f} MB")
    print("Zero-Variance Columns (Potential OHE bloat):", 
          len(df.columns[df.nunique() <= 1]))
    print("-" * 30 + "\n")

def analyze_target_sparsity():
    print("--- TARGET VARIABLE SPARSITY (DUCKDB) ---")
    if not os.path.exists(TRANSITIONS_PATH):
        print(f"Error: {TRANSITIONS_PATH} not found. Skipping sparsity check.")
        return

    start_time = time.time()
    
    # Connect to DuckDB (runs entirely in-memory for speed)
    con = duckdb.connect(database=':memory:')
    
    # Assume the transition log contains columns identifying the team.
    # Adjust 'p1_team_string' or equivalent column name based on your schema.
    try:
        query = f"""
            SELECT 
                COUNT(*) as total_battles,
                COUNT(DISTINCT team_id) as unique_teams
            FROM (
                -- Replace 'team_id' with however teams are identified in your logs
                -- e.g., CONCAT_WS('|', p1_mon1, p1_mon2, p1_mon3...)
                SELECT p1_team_id as team_id FROM '{TRANSITIONS_PATH}'
                UNION ALL
                SELECT p2_team_id as team_id FROM '{TRANSITIONS_PATH}'
            )
        """
        result = con.execute(query).fetchone()
        total_battles = result[0]
        unique_teams = result[1]
        
        print(f"Total Team Appearances: {total_battles:,}")
        print(f"Unique 6-Pokémon Teams: {unique_teams:,}")
        if unique_teams > 0:
            print(f"Average Appearances per Team: {total_battles / unique_teams:.2f}")
        
    except Exception as e:
        print(f"Could not execute sparsity query (Schema mismatch?): {e}")
        print("Please manually verify the column names in your transition dataset.")

    print(f"Query executed in {time.time() - start_time:.2f} seconds.")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    analyze_gold_tensors()
    analyze_target_sparsity()