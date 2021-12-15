from gym.envs.registration import register

register(
    id='cheetah-src-v0',
    entry_point='src.envs.cheetah:HalfCheetahEnv',
    max_episode_steps=1000,
)
from src.envs.cheetah.cheetah import HalfCheetahEnv
