import os
import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.agent.selection_policy.math_utils import TYPE_CHART

GOLD_TENSOR_PATH = "data/processed/gold_tensors.parquet"
MODEL_PATH = "models/archetype_gmm.pkl"
RANDOM_STATE = 42

MACRO_FEATURE_NAMES = ["avg_speed", "phys_spec_ratio", "bulk_index", "type_synergy_density"]
ALL_TYPES = list(TYPE_CHART.keys())


def load_gold_tensor() -> np.ndarray:
    df = pd.read_parquet(GOLD_TENSOR_PATH)

    spe = df["spe"].to_numpy(dtype=np.float32)
    atk = df["atk"].to_numpy(dtype=np.float32)
    spa = df["spa"].to_numpy(dtype=np.float32)
    hp = df["hp"].to_numpy(dtype=np.float32)
    defense = df["def"].to_numpy(dtype=np.float32)
    spd = df["spd"].to_numpy(dtype=np.float32)

    phys_spec_ratio = atk / np.maximum(spa, 1.0)
    bulk_index = hp + defense + spd

    type_1_cols = [c for c in df.columns if c.startswith("type_1_")]
    type_2_cols = [c for c in df.columns if c.startswith("type_2_")]

    n = len(df)
    synergy = np.zeros(n, dtype=np.float32)

    for i in range(n):
        t1 = next((c.split("_", 2)[2] for c in type_1_cols if df.iloc[i][c] == 1.0), "normal")
        t2 = next((c.split("_", 2)[2] for c in type_2_cols if df.iloc[i][c] == 1.0), None)
        defender_types = [t for t in [t1, t2] if t and t != "nan" and t != "None"]
        resist_count = 0
        for move_type in ALL_TYPES:
            type_eff = 1.0
            for dt in defender_types:
                type_eff *= TYPE_CHART.get(move_type, {}).get(dt, 1.0)
            if type_eff < 1.0:
                resist_count += 1
        synergy[i] = resist_count / 18.0

    X = np.column_stack([spe, phys_spec_ratio, bulk_index, synergy]).astype(np.float32)
    return X


def main() -> None:
    X = load_gold_tensor()
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Feature names: {MACRO_FEATURE_NAMES}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feat_df = pd.DataFrame(X, columns=MACRO_FEATURE_NAMES)
    print(f"Feature stats:\n{feat_df.describe().to_string()}")
    print()

    best_score = -1.0
    best_model: GaussianMixture | None = None
    best_k = 0
    best_scaler = scaler

    for k in range(2, 16):
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=5)
        labels = gmm.fit_predict(X_scaled)
        if len(np.unique(labels)) < 2:
            print(f"  k={k:2d}  only {len(np.unique(labels))} cluster(s) found, skipping silhouette")
            continue
        score = silhouette_score(X_scaled, labels)
        print(f"  k={k:2d}  silhouette={score:.6f}")

        if score > best_score:
            best_score = score
            best_model = gmm
            best_k = k

    print()
    print(f"Best k={best_k} with silhouette={best_score:.6f}")

    if best_score > 0.0 and best_model is not None:
        print(f"Silhouette positive. Serializing to {MODEL_PATH}")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"gmm": best_model, "scaler": best_scaler}, MODEL_PATH)
        print("Serialization complete (model + scaler).")
    else:
        print("Silhouette non-positive or no valid model. just my shitty luck. Pivot to KNN.")


if __name__ == "__main__":
    main()
