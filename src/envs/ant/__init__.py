from gym.envs.registration import register

register(
    id='ant-src-v0',
    entry_point='src.envs.ant:AntEnv',
    max_episode_steps=1000,
)
from src.envs.ant.ant import AntEnv
