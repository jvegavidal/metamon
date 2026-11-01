from metamon.env import get_metamon_teams, QueueOnLocalLadder
from metamon.interface import (
    DefaultObservationSpace,
    DefaultShapedReward,
    DefaultActionSpace,
)

team_set = get_metamon_teams("gen9ou", "competitive")
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
action_space = DefaultActionSpace()


env = QueueOnLocalLadder(
    battle_format="gen9ou",
    player_username="my_oponent",
    num_battles=10,
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_fn,
    player_team_set=team_set,
)

# standard `gymnasium` environment
"""obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
""" 

obs, info = env.reset()
terminated = False
turn = 0

while not terminated:
    turn += 1
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Turn {turn}: reward={reward}, terminated={terminated}, truncated={truncated}")

print("Battle finished")