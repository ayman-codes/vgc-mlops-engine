from src.config.loader import load_battle_weights

def test_load_battle_weights():
    weights = load_battle_weights()
    assert weights.W_FOCUS_FIRE_BONUS == 0.2700