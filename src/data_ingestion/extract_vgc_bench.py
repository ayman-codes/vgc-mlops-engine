import os
import duckdb
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

OUTPUT_DIR = "./vgc_data/showdown_logs"
TARGET_FILE = "logs_gen9vgc2026regfbo3.json"
REPO_ID = "cameronangliss/vgc-battle-logs"

def execute_duckdb_extraction() -> None:
    load_dotenv()
    hf_token: str | None = os.getenv("HF_TOKEN")
    
    if not hf_token:
        raise ValueError("HF_TOKEN missing from .env file.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Executing direct network transfer...")
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=TARGET_FILE,
        repo_type="dataset",
        token=hf_token,
        local_dir=OUTPUT_DIR
    )
    
    parquet_path = os.path.join(OUTPUT_DIR, f"vgc_bench_{TARGET_FILE.split('.')[0]}.parquet")
    
    print("Executing DuckDB out-of-core Parquet serialization...")
    duckdb.sql("INSTALL json;")
    duckdb.sql("LOAD json;")
    
    # 2GB object allocation override
    query = f"COPY (SELECT * FROM read_json_auto('{file_path}', maximum_object_size=2147483648)) TO '{parquet_path}' (FORMAT PARQUET);"
    duckdb.sql(query)
    
    print(f"Serialization absolute. Output: {parquet_path}")

if __name__ == "__main__":
    execute_duckdb_extraction()