import asyncio
import os
import matplotlib.pyplot as plt
import mlflow

from poke_env import cross_evaluate
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration
from poke_env.teambuilder import ConstantTeambuilder

from src.agent.battle_policy.main import DongimonHeuristic
from src.agent.battle_policy.baselines.epsilon_greedy import EpsilonGreedyBattlePolicy
from src.agent.battle_policy.baselines.softmax import SoftmaxBattlePolicy

# Valid Gen 9 VGC Team
DUMMY_TEAM = """
Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Flare Blitz
- Knock Off
- Parting Shot
- Fake Out

Rillaboom @ Assault Vest
Ability: Grassy Surge
Level: 50
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Wood Hammer
- Grassy Glide
- U-turn
- Fake Out

Flutter Mane @ Booster Energy
Ability: Protosynthesis
Level: 50
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Moonblast
- Shadow Ball
- Dazzling Gleam
- Protect

Urshifu-Rapid-Strike @ Mystic Water
Ability: Unseen Fist
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Surging Strikes
- Close Combat
- Aqua Jet
- Protect

Ogerpon-Wellspring @ Wellspring Mask
Ability: Water Absorb
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Ivy Cudgel
- Horn Leech
- Spiky Shield
- Follow Me

Chien-Pao @ Focus Sash
Ability: Sword of Ruin
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Icicle Crash
- Sucker Punch
- Sacred Sword
- Protect
"""

async def run_battle_benchmark() -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Battle_Policy_Heuristic_Benchmarks")
    
    with mlflow.start_run(run_name="DongimonHeuristic_Enhanced_v1"):
        print("Starting Battle Policy Cross-Evaluation Tournament...")
        
        battle_format = "gen9vgc2024regf"
        server_config = ServerConfiguration(
            "ws://localhost:8000/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?"
        )
        team_builder = ConstantTeambuilder(DUMMY_TEAM)
        
        # Instantiate the Competitor League
        players = [
            RandomPlayer(battle_format=battle_format, server_configuration=server_config, team=team_builder, max_concurrent_battles=10, accept_open_team_sheet=True),
            MaxBasePowerPlayer(battle_format=battle_format, server_configuration=server_config, team=team_builder, max_concurrent_battles=10, accept_open_team_sheet=True),
            SimpleHeuristicsPlayer(battle_format=battle_format, server_configuration=server_config, team=team_builder, max_concurrent_battles=10, accept_open_team_sheet=True),
            DongimonHeuristic(battle_format=battle_format, server_configuration=server_config, team=team_builder, max_concurrent_battles=10, accept_open_team_sheet=True),
            EpsilonGreedyBattlePolicy(epsilon=0.2, battle_format=battle_format, server_configuration=server_config, team=team_builder, max_concurrent_battles=10, accept_open_team_sheet=True),
            SoftmaxBattlePolicy(tau=1.0, battle_format=battle_format, server_configuration=server_config, team=team_builder, max_concurrent_battles=10, accept_open_team_sheet=True)
        ]
        
        n_challenges = 5
        cross_eval_results = await cross_evaluate(players, n_challenges=n_challenges)
        
        player_names = ["Random", "MaxBasePower", "SimpleHeuristics", "DongimonHeuristic", "EpsilonGreedy", "Softmax"]
        win_rates = []
        
        for i, p1 in enumerate(players):
            total_wins = 0
            total_matches = 0
            for j, p2 in enumerate(players):
                if i != j:
                    wins = cross_eval_results.get(p1.username, {}).get(p2.username, 0)
                    total_wins += wins if wins is not None else 0
                    total_matches += n_challenges
            
            win_rate = total_wins / total_matches if total_matches > 0 else 0
            win_rates.append(win_rate)
            mlflow.log_metric(f"{player_names[i]}_win_rate", win_rate)
            print(f"{player_names[i]} Win Rate: {win_rate:.2%}")

        # Visualization
        plt.figure(figsize=(12, 6))
        colors = ['gray', 'orange', 'blue', 'green', 'purple', 'red']
        plt.bar(player_names, [wr * 100 for wr in win_rates], color=colors)
        plt.title('Battle Policy Cross-Evaluation Win Rates')
        plt.ylabel('Win Rate (%)')
        plt.ylim(0, 100)
        plt.xticks(rotation=15)
        
        for i, wr in enumerate(win_rates):
            plt.text(i, wr * 100 + 2, f"{wr * 100:.1f}%", ha='center')
            
        os.makedirs("images", exist_ok=True)
        img_path = "images/battle_baselines_benchmark.png"
        plt.savefig(img_path, bbox_inches="tight")
        mlflow.log_artifact(img_path)
        print(f"Chart saved to {img_path}")

if __name__ == "__main__":
    asyncio.run(run_battle_benchmark())
