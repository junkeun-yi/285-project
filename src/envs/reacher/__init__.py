from gym.envs.registration import register

register(
    id='reacher-src-v0',
    entry_point='src.envs.reacher:Reacher7DOFEnv',
    max_episode_steps=500,
)
from src.envs.reacher.reacher_env import Reacher7DOFEnv
