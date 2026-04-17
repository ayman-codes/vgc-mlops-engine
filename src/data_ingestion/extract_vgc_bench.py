from datasets import load_dataset
import os

OUTPUT_DIR = "/vgc_data/showdown_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def execute_hf_extraction():
    # Load dataset; target specific format split as needed
    dataset = load_dataset("cameronangliss/vgc-battle-logs", split="train")
    output_path = os.path.join(OUTPUT_DIR, "vgc_bench_offline.parquet")
    dataset.to_parquet(output_path)
    print(f"VGC-Bench logs persisted to {output_path}")

if __name__ == "__main__":
    execute_hf_extraction()