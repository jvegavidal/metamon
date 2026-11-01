from metamon.baselines import get_baseline
from metamon.env import get_metamon_teams, BattleAgainstBaseline
from metamon.interface import (
    DefaultObservationSpace,
    DefaultShapedReward,
    DefaultActionSpace,
)

team_set = get_metamon_teams("gen9ou", "competitive")
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
action_space = DefaultActionSpace()


env = BattleAgainstBaseline(
    battle_format="gen9ou",
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_fn,
    team_set=team_set,
    opponent_type=get_baseline("PokeEnvHeuristic"),
)

# standard `gymnasium` environment
#obs, info = env.reset()
#next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

obs, info = env.reset()
terminated = False

while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward, terminated, truncated)