from metamon.env import get_metamon_teams
from metamon.interface import (
    DefaultObservationSpace,
    DefaultShapedReward,
    DefaultActionSpace,
)

team_set = get_metamon_teams("gen9ou", "competitive")
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
action_space = DefaultActionSpace()
