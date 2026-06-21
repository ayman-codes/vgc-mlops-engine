import numpy as np
from numpy.typing import NDArray
from poke_env.data import GenData

from src.agent.selection_policy.math_utils import TYPE_CHART

_gen_data: GenData | None = None


def _get_gen_data() -> GenData:
    global _gen_data
    if _gen_data is None:
        _gen_data = GenData.from_gen(9)
    return _gen_data


ALL_TYPES: list[str] = list(TYPE_CHART.keys())


def aggregate_macro_features(species_list: list[str]) -> dict[str, np.float32]:
    gen_data = _get_gen_data()
    n = len(species_list)

    avg_speed_val = np.float32(0.0)
    phys_spec_ratio_val = np.float32(0.0)
    bulk_index_val = np.float32(0.0)
    type_synergy_density_val = np.float32(0.0)

    if n == 0:
        return {
            "avg_speed": avg_speed_val,
            "phys_spec_ratio": phys_spec_ratio_val,
            "bulk_index": bulk_index_val,
            "type_synergy_density": type_synergy_density_val,
        }

    phys_sum = 0.0
    spec_sum = 0.0
    hp_sum = 0.0
    def_sum = 0.0
    spd_sum = 0.0
    speed_sum = 0.0

    resist_counts: dict[str, float] = {t: 0.0 for t in ALL_TYPES}

    for species in species_list:
        entry = gen_data.pokedex.get(species)
        if entry is None:
            continue
        base_stats = entry.get("baseStats", {})
        speed_sum += base_stats.get("spe", 100)

        atk_val = base_stats.get("atk", 100)
        spa_val = base_stats.get("spa", 100)
        phys_sum += atk_val
        spec_sum += spa_val

        hp_sum += base_stats.get("hp", 100)
        def_sum += base_stats.get("def", 100)
        spd_sum += base_stats.get("spd", 100)

        species_types: list[str] = entry.get("types", [])
        for t in ALL_TYPES:
            type_eff = 1.0
            for dt in species_types:
                type_eff *= TYPE_CHART.get(t, {}).get(dt, 1.0)
            if type_eff < 1.0:
                resist_counts[t] += 1.0

    avg_speed_val = np.float32(speed_sum / n)
    bulk_index_val = np.float32((hp_sum + def_sum + spd_sum) / n)
    phys_spec_ratio_val = np.float32(phys_sum / max(spec_sum, 1.0))
    total_possible_resists = 18.0 * float(n)
    resist_sum = sum(resist_counts.values())
    type_synergy_density_val = np.float32(resist_sum / max(total_possible_resists, 1.0))

    return {
        "avg_speed": avg_speed_val,
        "phys_spec_ratio": phys_spec_ratio_val,
        "bulk_index": bulk_index_val,
        "type_synergy_density": type_synergy_density_val,
    }


def macro_features_array(species_list: list[str]) -> NDArray[np.float32]:
    features = aggregate_macro_features(species_list)
    return np.array(
        [features["avg_speed"], features["phys_spec_ratio"], features["bulk_index"], features["type_synergy_density"]],
        dtype=np.float32,
    )
