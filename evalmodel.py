from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

env = make_atari_env("FreewayNoFrameskip-v0")

def evaluate_model(args):
    model = PPO.load(args['teacher_chkpt'])

    mean, std = evaluate_policy(model, env, n_eval_episodes=args['n_eval_episodes'])
    
    print(f"mean: {mean}, std: {std}")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--teacher_chkpt',
        default='cs285/teachers/FreewayNoFrameskip-v0-n_steps128-batch_size256-timesteps10000000.0-2021:12:03_07:07:59'
    )

    parser.add_argument(
        '--n_eval_episodes',
        default=10
    )

    args = vars(parser.parse_args())
    evaluate_model(args)