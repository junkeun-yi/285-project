from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import os

def evaluate_model(args):
    env_name = args['env']
    env = make_atari_env(env_name)
    render = args['render']
    model = PPO.load(args['teacher_chkpt'])

    mean, std = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=args['n_eval_episodes'], 
        render=render
    )
    
    print(f"mean: {mean}, std: {std}")

    # print("Type Ctrl+C to exit the simulation.")
    # obs = env.reset()
    # while True:
    #     action, states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env',
        choices=[
            "FreewayNoFrameskip-v0", # good returns 30
            "BeamRiderNoFrameskip-v4", # good returns 1000
            "BowlingNoFrameskip-v4", # good returns 100
            "PongNoFrameskip-v4", # good returns 7.5
            "MsPacmanNoFrameskip-v4", # good returns 450
            "QbertNoFrameskip-v4", # good returns 2300
            "UpNDownNoFrameskip-v4", # good returns 12500-15000
        ],
        default="FreewayNoFrameskip-v0"
    )

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

    parser.add_argument(
        '--render',
        action='store_true'
    )

    args = vars(parser.parse_args())
    evaluate_model(args)