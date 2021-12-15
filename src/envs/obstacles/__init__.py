from gym.envs.registration import register

register(
    id='obstacles-src-v0',
    entry_point='src.envs.obstacles:Obstacles',
    max_episode_steps=500,
)
from src.envs.obstacles.obstacles_env import Obstacles
