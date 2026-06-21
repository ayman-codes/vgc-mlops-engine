from src.agent.selection_policy.bayesian_do_selection import BayesianDoubleOraclePolicy

def test_generate_payoff_matrix() -> None:
    policy = BayesianDoubleOraclePolicy()
    my_roster = ["Pikachu", "Charizard", "Blastoise", "Venusaur", "Snorlax", "Alakazam"]
    
    # Mock hydrated opponent (from Bayesian engine)
    opp_hydrated = {
        "Gengar": {"item": "Life Orb"},
        "Dragonite": {"item": "Choice Band"},
        "Gyarados": {"item": "Lum Berry"},
        "Machamp": {"item": "Focus Sash"},
        "Jolteon": {"item": "Magnet"},
        "Lapras": {"item": "Leftovers"}
    }
    
    matrix = policy.generate_payoff_matrix(my_roster, opp_hydrated)
    
    # 6 choose 4 = 15
    assert matrix.shape == (15, 15)

def test_calculate_selection() -> None:
    policy = BayesianDoubleOraclePolicy()
    my_roster = ["Pikachu", "Charizard", "Blastoise", "Venusaur", "Snorlax", "Alakazam"]
    opp_roster = ["Gengar", "Dragonite", "Gyarados", "Machamp", "Jolteon", "Lapras"]
    
    # Needs to return a list of 4 Pokemon
    selection = policy.calculate_selection(my_roster, opp_roster)
    
    assert len(selection) == 4
    for mon in selection:
        assert mon in my_roster
