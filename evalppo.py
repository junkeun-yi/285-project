from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import os
import csv

env_choices = [
            "FreewayNoFrameskip-v0", # standard ppo returns 32.5 (40 million iters)
            "FreewayNoFrameskip-v4", # standard ppo returns 32.5 (40 million iters)
            "BeamRiderNoFrameskip-v4", # standard ppo returns 1590
            "BowlingNoFrameskip-v4", # standard ppo returns 40.1
            "PongNoFrameskip-v4", # standard ppo returns 20.7
            "MsPacmanNoFrameskip-v4", # standard ppo returns 2096
            "QbertNoFrameskip-v4", # standard ppo returns 14293.3
            "UpNDownNoFrameskip-v4", # standard ppo returns 95445
            "DO-ALL"
        ]

def evaluate_model(env_name, teacher_chkpt_name, n_eval_episodes, render):
    env = make_atari_env(env_name)
    model = PPO.load(teacher_chkpt_name)

    mean, std = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_eval_episodes, 
        render=render
    )

    info_dict = {
        'teacher_name': teacher_chkpt_name,
        'mean': mean,
        'std': std,
        'n_eval_episodes': n_eval_episodes
    }

    return info_dict

def main(args):
    env_name = args['env']
    teacher_chkpt_loc = args['teacher_chkpt']
    output_csv_loc = args['output_csv_loc']
    n_eval_episodes = args['n_eval_episodes']
    render = args['render']

    if env_name == "DO-ALL":
        os.makedirs(os.path.dirname(output_csv_loc), exist_ok=True)
        with open(output_csv_loc, 'w') as csvfile:
            fieldnames = ['teacher_name', 'env', 'n_iters', 'mean', 'std', 'n_eval_episodes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for teacher_chkpt in os.listdir(teacher_chkpt_loc):
                teacher_chkpt_name = os.path.join(teacher_chkpt_loc, teacher_chkpt)
                try:
                    env_name=None
                    n_iters='unknown'
                    teacher_name_values = teacher_chkpt.replace('.zip', '').split('_')
                    for teacher_name_value in teacher_name_values:
                        if 'env' in teacher_name_value:
                            env_name = teacher_name_value.replace('env', '')
                        if 'iters' in teacher_name_value:
                            n_iters = teacher_name_value.replace('iters', '')
                    info_dict = evaluate_model(env_name, teacher_chkpt_name, n_eval_episodes, render)
                    info_dict['env'] = env_name
                    info_dict['n_iters'] = n_iters
                    writer.writerow(info_dict)
                    print(info_dict)
                except Exception as e:
                    print(f"Failed to eval teacher, {teacher_chkpt_name}. {e}")
    else:
        teacher_chkpt_name = teacher_chkpt_loc
        info_dict = evaluate_model(env_name, teacher_chkpt_name, n_eval_episodes, render)
        print(info_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env',
        choices=env_choices,
        required=True,
        help="which environment to eval in, DO-ALL for do all in a folder"
    )

    parser.add_argument(
        '--teacher_chkpt',
        required=True,
        help="location for teacher checkpoint model. If using DO-ALL as the env, give the folder location for the teacher"
    )

    parser.add_argument(
        '--n_eval_episodes',
        default=10
    )

    parser.add_argument(
        '--render',
        action='store_true'
    )

    parser.add_argument(
        '--output_csv_loc',
        type=str,
        default='evalppo_logs/evalppo.csv',
        help="location to output csv of teachername, evaluation pairs. Only used when using DO-ALL"
    )

    args = vars(parser.parse_args())
    main(args)