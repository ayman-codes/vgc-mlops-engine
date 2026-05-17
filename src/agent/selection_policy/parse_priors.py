import os
import json
import glob
from collections import defaultdict

BATTLE_LOGS_DIR = "C:/Users/Mohammed Ayman PC/vgc-bench/battle_logs"
OUTPUT_PATH = "data/processed/bayesian_priors.json"

def parse_priors() -> None:
    # Structure: priors[species]["items" | "abilities" | "moves"][value] = count
    priors: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: {
        "items": defaultdict(int),
        "abilities": defaultdict(int),
        "moves": defaultdict(int),
        "total": defaultdict(int) # to store total occurrences for probabilities
    })

    log_files = glob.glob(os.path.join(BATTLE_LOGS_DIR, "*.json"))
    
    # Process just a few files for demonstration/speed, or all if needed.
    for log_file in log_files[:2]: 
        print(f"Processing {log_file}...")
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for game_id, log_data in data.items():
                if not isinstance(log_data, list):
                    continue
                # log_data is usually a list where the second element is the string log
                log_text = ""
                for item in log_data:
                    if isinstance(item, str):
                        log_text = item
                        break
                
                for line in log_text.split('\n'):
                    if line.startswith('|showteam|'):
                        parts = line.split('|')
                        if len(parts) >= 7:
                            # |showteam|p1|species|nickname|item|ability|moves|...
                            species = parts[3].strip()
                            item = parts[5].strip()
                            ability = parts[6].strip()
                            moves = parts[7].strip().split(',')
                            
                            if species:
                                priors[species]["total"]["count"] += 1
                                if item:
                                    priors[species]["items"][item] += 1
                                if ability:
                                    priors[species]["abilities"][ability] += 1
                                for move in moves:
                                    if move:
                                        priors[species]["moves"][move] += 1

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Convert defaultdicts to regular dicts for JSON serialization
    serialized_priors = {}
    for species, data in priors.items():
        serialized_priors[species] = {
            "items": dict(data["items"]),
            "abilities": dict(data["abilities"]),
            "moves": dict(data["moves"]),
            "total": data["total"]["count"]
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(serialized_priors, f, indent=2)
    
    print(f"Successfully generated Bayesian priors at {OUTPUT_PATH}")

if __name__ == "__main__":
    parse_priors()
