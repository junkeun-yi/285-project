from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import os

env = make_atari_env("FreewayNoFrameskip-v0")

def evaluate_model(args):
    model = PPO.load(args['teacher_chkpt'])

    mean, std = evaluate_policy(model, env, n_eval_episodes=args['n_eval_episodes'])
    
    print(f"mean: {mean}, std: {std}")

    print("Type Ctrl+C to exit the simulation.")
    obs = env.reset()
    while True:
        action, states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    teachers = os.listdir('cs285/teachers/')
    latest_teacher = teachers[-1]

    parser.add_argument(
        '--teacher_chkpt',
        default=latest_teacher
    )

    parser.add_argument(
        '--n_eval_episodes',
        default=10
    )

    args = vars(parser.parse_args())
    evaluate_model(args)