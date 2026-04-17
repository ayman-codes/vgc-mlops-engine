import os
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_info
from dotenv import load_dotenv

OUTPUT_DIR = "./vgc_data/showdown_logs"
TARGET_FILE = "logs_gen9vgc2026regfbo3.json" 

def execute_hf_extraction() -> None:
    load_dotenv()
    hf_token: str | None = os.getenv("HF_TOKEN")
    
    if not hf_token:
        raise ValueError("HF_TOKEN missing from .env file.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Force stdout progress rendering
    set_verbosity_info()
    
    dataset = load_dataset(
        "cameronangliss/vgc-battle-logs", 
        data_files=TARGET_FILE, 
        split="train", 
        token=hf_token
    )
    
    output_path = os.path.join(OUTPUT_DIR, f"vgc_bench_{TARGET_FILE.split('.')[0]}.parquet")
    dataset.to_parquet(output_path)

if __name__ == "__main__":
    execute_hf_extraction()