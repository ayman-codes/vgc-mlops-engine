import concurrent.futures
import multiprocessing
from typing import Tuple

from vgc2.battle_engine import BattleEngine, State
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.competition.match import label_teams
from vgc2.util.generator import gen_team
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.battle_engine.pokemon import Pokemon
from vgc2.battle_engine.team import Team

from src.agent.battle_policy.main import MyBattlePolicy # type: ignore
from temp.my_battle_policy import LabBattlePolicy
from src.agent.battle_policy.baselines.epsilon_greedy import EpsilonGreedyBattlePolicy
from src.agent.battle_policy.baselines.softmax import SoftmaxBattlePolicy

# --- Execution Parameters ---
ITERATIONS = 5000
TEAM_SIZE = 2
N_MOVES = 4
N_ACTIVE = 2
TURN_LIMIT = 50

def generate_mirror_teams() -> Tuple[Team, Team]:
    base_team = gen_team(TEAM_SIZE, N_MOVES)
    members_p0 = []
    members_p1 = []
    
    for p in base_team.members:
        move_indices = [p.species.moves.index(m) for m in p.moves]
        members_p0.append(Pokemon(p.species, move_indices, p.level, p.evs, p.ivs, p.nature))
        members_p1.append(Pokemon(p.species, move_indices, p.level, p.evs, p.ivs, p.nature))
        
    return Team(members_p0), Team(members_p1)

def format_commands(cmd, state_view, side: int) -> list:
    num_active = sum(1 for p_idx, p in enumerate(state_view.sides[side].team.active) 
                     if p and p.hp > 0 and p_idx < N_ACTIVE)
    
    if isinstance(cmd, tuple) and len(cmd) == 2:
        cmd = [cmd]
        
    if not isinstance(cmd, list):
        cmd = [(0, 0)] * num_active
        
    while len(cmd) < num_active: 
        cmd.append((0, 0))
        
    return cmd[:num_active]

def execute_simulation_instance(match_id: int, p0_name: str, p1_name: str) -> int:
    team_p0, team_p1 = generate_mirror_teams()
    label_teams((team_p0, team_p1))

    try:
        team_views_full = (TeamView(team_p0, False), TeamView(team_p1, False))
    except TypeError:
        team_views_full = (TeamView(team_p0), TeamView(team_p1))
        
    battling_teams = get_battle_teams((team_p0, team_p1), N_ACTIVE)
    engine = BattleEngine(State(battling_teams))
    
    def get_policy_instance(name: str):
        if name == "LabBattlePolicy":
            return LabBattlePolicy(detailed_logging=False)
        elif name == "MyBattlePolicy":
            return MyBattlePolicy()
        elif name == "GreedyBattlePolicy":
            return GreedyBattlePolicy()
        elif name == "EpsilonGreedyBattlePolicy":
            return EpsilonGreedyBattlePolicy(epsilon=0.2)
        elif name == "SoftmaxBattlePolicy":
            return SoftmaxBattlePolicy(tau=1.0)
            
    policy_p0 = get_policy_instance(p0_name)
    policy_p1 = get_policy_instance(p1_name)

    turn_count = 0
    while not engine.finished() and turn_count < TURN_LIMIT:
        turn_count += 1
        
        state_view_p0 = StateView(engine.state, 0, team_views_full)
        state_view_p1 = StateView(engine.state, 1, team_views_full)

        if p0_name in ["LabBattlePolicy", "MyBattlePolicy", "SoftmaxBattlePolicy"]:
            cmd_p0_raw = policy_p0.decision(state_view_p0, turn_count)
        else:
            cmd_p0_raw = policy_p0.decision(state_view_p0)
            
        if p1_name in ["LabBattlePolicy", "MyBattlePolicy", "SoftmaxBattlePolicy"]:
            cmd_p1_raw = policy_p1.decision(state_view_p1, turn_count)
        else:
            cmd_p1_raw = policy_p1.decision(state_view_p1)

        cmd_p0 = format_commands(cmd_p0_raw, state_view_p0, 0)
        cmd_p1 = format_commands(cmd_p1_raw, state_view_p1, 1)

        engine.run_turn((cmd_p0, cmd_p1))

    return engine.winning_side

def execute_validation_pipeline():
    cpu_cores = max(1, multiprocessing.cpu_count() - 1)
    
    matchups = [
        ("MyBattlePolicy", "GreedyBattlePolicy"),
        ("MyBattlePolicy", "EpsilonGreedyBattlePolicy"),
        ("MyBattlePolicy", "SoftmaxBattlePolicy"),
        ("SoftmaxBattlePolicy", "GreedyBattlePolicy"),
        ("EpsilonGreedyBattlePolicy", "GreedyBattlePolicy"),
        ("LabBattlePolicy", "MyBattlePolicy")
    ]
    
    for p0_name, p1_name in matchups:
        print(f"\nINITIALIZING MATRIX: {ITERATIONS} Iterations. Mirror Match: {p0_name} vs {p1_name}")
        
        wins_p0 = 0
        wins_p1 = 0
        draws = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            futures = [executor.submit(execute_simulation_instance, i, p0_name, p1_name) for i in range(ITERATIONS)]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result == 0:
                    wins_p0 += 1
                elif result == 1:
                    wins_p1 += 1
                else:
                    draws += 1

        win_rate = (wins_p0 / ITERATIONS) * 100
        
        print("--- EXECUTION TERMINATED ---")
        print(f"{p0_name} Wins: {wins_p0}")
        print(f"{p1_name} Wins: {wins_p1}")
        print(f"Draws / Timeouts: {draws}")
        print(f"{p0_name} WIN RATE: {win_rate:.2f}%")

if __name__ == '__main__':
    execute_validation_pipeline()