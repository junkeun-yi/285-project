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
        required=True,
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

    parser.add_argument('--exp_name', type=str, default='')

    parser.add_argument('--eval_batch_size', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)

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
    parser.add_argument('--teacher_chkpt', type=str, required=True)

    # Student
    parser.add_argument('--temperature', type=int, default=0.01)
    parser.add_argument("--use_curiosity", action="store_true")
    parser.add_argument("--use_icm", action="store_true")
    parser.add_argument("--curiosity_weight", type=float, default=0.1)
    parser.add_argument("--icm_beta", type=float, default = 0.1)
    parser.add_argument("--use_uncertainity", action="store_true", help="Use our Uncertainity based method (unless o.w. uses Random Feat)")

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

    # If using our method, must use curiosity (default if user only types --use_uncertainity is Random Feat curiosity)
    if params["use_uncertainity"] and not params["use_curiosity"]:
        params["use_curiosity"] = True
    
    
    # NOTE: had to change this for each environment
    if params['env_name']=='FreewayNoFrameskip-v0':
        params['ep_len']=128
    
    if params['use_curiosity']:
        params['explore_weight_schedule'] = ConstantSchedule(params['curiosity_weight'])
    else:
        params['explore_weight_schedule'] = ConstantSchedule(0.0)

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = 'distill'
    if args.exp_name:
        logdir = f"{logdir}_{args.exp_name}_"
    cleaned_teacher_name = os.path.basename(args.teacher_chkpt).replace('.zip', '')
    if args.env_name in args.teacher_chkpt:
        logdir = f"{logdir}_teacher{cleaned_teacher_name}"
    else:
        logdir = f"{logdir}_teacher{cleaned_teacher_name}_env{args.env_name}"
    current_time = time.strftime("%Y%m%d-%H%M%S")
    logdir = f"{logdir}_{current_time}"

    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    trainer = Distill_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
