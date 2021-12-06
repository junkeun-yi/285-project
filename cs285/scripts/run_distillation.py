import os
import time

from cs285.infrastructure.rl_trainer_distillation import RL_Trainer
from cs285.agents.distillation_agent import DistillationAgent
from cs285.infrastructure.dqn_utils import get_env_kwargs, PiecewiseSchedule, ConstantSchedule


class Distill_Trainer(object):

    def __init__(self, params):
        self.params = params

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'double_q': params['double_q'],
            'use_boltzmann': params['use_boltzmann'],
        }

        env_args = get_env_kwargs(params['env_name'])

        self.agent_params = {**train_args, **env_args, **params}

        self.params['agent_class'] = DistillationAgent
        self.params['agent_params'] = self.agent_params
        self.params['train_batch_size'] = params['batch_size']
        self.params['env_wrappers'] = self.agent_params['env_wrappers']

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.agent_params['num_timesteps'],
            collect_policy = self.rl_trainer.agent.actor,  # TODO check these
            eval_policy = self.rl_trainer.agent.actor,
            )

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name',
        default='FreewayNoFrameskip-v0',
        choices=[
            "FreewayNoFrameskip-v0", # standard ppo returns 32.5 (40 million iters)
            "FreewayNoFrameskip-v4", # standard ppo returns 32.5 (40 million iters)
            "BeamRiderNoFrameskip-v4", # standard ppo returns 1590
            "BowlingNoFrameskip-v4", # standard ppo returns 40.1
            "PongNoFrameskip-v4", # standard ppo returns 20.7
            "MsPacmanNoFrameskip-v4", # standard ppo returns 2096
            "QbertNoFrameskip-v4", # standard ppo returns 14293.3
            "UpNDownNoFrameskip-v4", # standard ppo returns 95445
        ]
    )

    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--use_rnd', action='store_true')
    parser.add_argument('--num_exploration_steps', type=int, default=10000)
    parser.add_argument('--unsupervised_exploration', action='store_true')

    parser.add_argument('--offline_exploitation', action='store_true')
    parser.add_argument('--cql_alpha', type=float, default=0.0)

    parser.add_argument('--exploit_rew_shift', type=float, default=0.0)
    parser.add_argument('--exploit_rew_scale', type=float, default=1.0)

    parser.add_argument('--rnd_output_size', type=int, default=5)
    parser.add_argument('--rnd_n_layers', type=int, default=2)
    parser.add_argument('--rnd_size', type=int, default=400)

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    # parser.add_argument('--scalar_log_freq', type=int, default=int(1e3))
    parser.add_argument('--scalar_log_freq', type=int, default=256) # make same as batch size
    parser.add_argument('--save_params', action='store_true')

    parser.add_argument('--use_boltzmann', action='store_true')

    # for DQNAgent
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--learning_rate', type=int, default=1e-4)
    parser.add_argument('--n_actions', type=int, default=3)

    # Distillation parameters
    # Teacher
    parser.add_argument('--distill_policy', type=str, default='CnnPolicy')
    parser.add_argument('--teacher_chkpt', type=str, default='cs285/teachers/2021-12-05_04:26:45_envFreewayNoFrameskip-v0_n_iters10000000.zip')

    # Student
    parser.add_argument('--temperature', type=int, default=0.01)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    print(params["teacher_chkpt"])
    params['double_q'] = True
    params['num_agent_train_steps_per_iter'] = 1
    params['num_critic_updates_per_agent_update'] = 1
    params['exploit_weight_schedule'] = ConstantSchedule(1.0)
    params['video_log_freq'] = -1 # This param is not used for DQN
    params['eps'] = 0.05
    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    # if params['env_name']=='PointmassEasy-v0':
    #     params['ep_len']=50
    # if params['env_name']=='PointmassMedium-v0':
    #     params['ep_len']=150
    # if params['env_name']=='PointmassHard-v0':
    #     params['ep_len']=100
    # if params['env_name']=='PointmassVeryHard-v0':
    #     params['ep_len']=200
    
    # NOTE: had to change this for each environment
    if params['env_name']=='FreewayNoFrameskip-v0':
        params['ep_len']=128
    
    if params['use_rnd']:
        params['explore_weight_schedule'] = PiecewiseSchedule([(0,1), (params['num_exploration_steps'], 0)], outside_value=0.0)
    else:
        params['explore_weight_schedule'] = ConstantSchedule(0.0)

    if params['unsupervised_exploration']:
        params['explore_weight_schedule'] = ConstantSchedule(1.0)
        params['exploit_weight_schedule'] = ConstantSchedule(0.0)
        
        if not params['use_rnd']:
            params['learning_starts'] = params['num_exploration_steps']

    logdir_prefix = 'distill_'
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    trainer = Distill_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
