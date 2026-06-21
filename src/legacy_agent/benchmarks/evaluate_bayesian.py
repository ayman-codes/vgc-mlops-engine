import json
import glob
import mlflow
from src.agent.selection_policy.bayesian_inference import BayesianInferenceEngine

def run_bayesian_benchmark() -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Bayesian_Offline_Accuracy_Benchmarks")
    
    with mlflow.start_run(run_name="Bayesian_Hydration_Accuracy"):
        engine = BayesianInferenceEngine()
        
        log_files = glob.glob("C:/Users/Mohammed Ayman PC/vgc-bench/battle_logs/*.json")
        total_mons = 0
        correct_items = 0
        correct_abilities = 0
        
        print("Starting offline evaluation over Showdown Battle Logs...")
        for log_file in log_files[:5]: # Evaluate over a subset of files to save time
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for game_id, log_data in data.items():
                    if not isinstance(log_data, list):
                        continue
                    log_text = ""
                    for item in log_data:
                        if isinstance(item, str):
                            log_text = item
                            break
                    for line in log_text.split('\n'):
                        if line.startswith('|showteam|'):
                            parts = line.split('|')
                            if len(parts) >= 7:
                                species = parts[3].strip()
                                item = parts[5].strip()
                                ability = parts[6].strip()
                                
                                if species:
                                    # Simulate Team Preview: Only provide species name
                                    predicted = engine.predict_pokemon_build(species)
                                    total_mons += 1
                                    
                                    # Compare against ground truth
                                    if predicted.get("item") == item:
                                        correct_items += 1
                                    if predicted.get("ability") == ability:
                                        correct_abilities += 1

        if total_mons > 0:
            item_accuracy = correct_items / total_mons
            ability_accuracy = correct_abilities / total_mons
            
            mlflow.log_metric("item_prediction_accuracy", item_accuracy)
            mlflow.log_metric("ability_prediction_accuracy", ability_accuracy)
            
            print(f"Bayesian Benchmark Complete. Total Masked Entities Evaluated: {total_mons}")
            print(f"Item Prediction Accuracy: {item_accuracy:.2%}")
            print(f"Ability Prediction Accuracy: {ability_accuracy:.2%}")
        else:
            print("No entities evaluated. Check dataset path.")

if __name__ == "__main__":
    run_bayesian_benchmark()
