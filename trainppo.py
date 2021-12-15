import sys
import gym
import os
from datetime import datetime
import argparse
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env

class SaveCheckpointSmartCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps),
    given a certain base check frequency, save a checkpoint at intervals multiplying by 
    1, 2, 5, 10, 20, 50, 100, etc.

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, base_freq: int, save_dir: str, save_name: str, verbose=1):
        super(SaveCheckpointSmartCallback, self).__init__(verbose)
        self.base_freq = base_freq
        self.save_dir = save_dir
        self.save_name = save_name

        self.tens_multiplier = 1
        self.within_tens_indexer = 0
        self.within_tens_list = [1, 2, 5]

        self.env_step_counter = 0

        print(self.base_freq, self.save_dir, self.save_name)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_name is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:

        iter_to_save_at = self.base_freq * self.tens_multiplier * \
            self.within_tens_list[self.within_tens_indexer]

        # if self.n_calls % 10000 == 0:
        # print(self.num_timesteps, iter_to_save_at)

        if self.num_timesteps >= iter_to_save_at:
            print(f"Saving checkpoint, n_timesteps {self.num_timesteps}.")

            # Retrieve training reward
            self.model.save(self.save_dir + self.save_name + f"n_iters{iter_to_save_at}")

            if self.within_tens_indexer + 1 == len(self.within_tens_list):
                self.tens_multiplier *= 10
                self.within_tens_indexer = 0
            else:
                self.within_tens_indexer += 1

        return True

def train_model(arguments):
    env_name = arguments['env']
    n_iters = arguments['n_iters']
    device = arguments['device']
    tensorboard_log = 'logs/'
    verbose = arguments['verbose']
    seed = arguments['seed']
    base_chkpt_freq = arguments['base_chkpt_freq']
    env = make_atari_env(env_name, n_envs=8)  # make 8 parallel environments
    datestring = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = "src/teachers/"
    save_name = (
        f"env{env_name}_"
        f"{datestring}_"
    )

    # callback to save checkpoints
    callback = SaveCheckpointSmartCallback(
        base_freq=base_chkpt_freq, 
        save_dir=save_dir, 
        save_name=save_name
    )

    model = PPO("CnnPolicy", 
        env, 
        verbose=verbose,
        learning_rate=lambda x: 2.5*1e-4*x, 
        n_steps=128, 
        batch_size=32*8, 
        n_epochs=3,
        clip_range=lambda x: 0.1*x, 
        ent_coef=0.001, 
        vf_coef=1, 
        device=device,
        tensorboard_log='logs/',
        seed=seed
    )

    model.learn(total_timesteps=n_iters, callback=callback, tb_log_name=save_name)
    model.save(save_dir + save_name + f"n_iters{n_iters}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env',
        choices=[
            "FreewayNoFrameskip-v0", # standard ppo returns 32.5 (40 million iters)
            "FreewayNoFrameskip-v4", # standard ppo returns 32.5 (40 million iters)
            "BeamRiderNoFrameskip-v4", # standard ppo returns 1590
            "BowlingNoFrameskip-v4", # standard ppo returns 40.1
            "PongNoFrameskip-v4", # standard ppo returns 20.7
            "MsPacmanNoFrameskip-v4", # standard ppo returns 2096
            "QbertNoFrameskip-v4", # standard ppo returns 14293.3
            "UpNDownNoFrameskip-v4", # standard ppo returns 95445
        ],
        required=True,
        help="The atari environment name."
    )

    parser.add_argument(
        '--n_iters',
        type=int,
        default=int(4e7),  # PPO was benched on 40 million iterations
        help="Number of iterations to train the PPO model."
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default="auto",
        help='pytorch device to run training on.'
    )

    parser.add_argument(
        '--verbose',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="The verbosity level for PPO training. 0 no output 1 info 2 debug"
    )

    parser.add_argument(
        '--base_chkpt_freq',
        type=int,
        default=100000,
        help="The base frequency at which to save checkpoints. The frequency is slowly "
        "decreased (i.e. saves less and less frequently)."
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Seed for training PPO, default None (i.e. seed is not called)."
    )

    arguments = vars(parser.parse_args())

    print(arguments)
    train_model(arguments)
