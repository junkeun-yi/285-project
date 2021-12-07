from collections import OrderedDict
import pickle
import os
import sys
import time
import pdb

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from cs285.agents.explore_or_exploit_agent import ExplorationOrExploitationAgent
from cs285.agents.distillation_agent import DistillationAgent
from cs285.infrastructure.dqn_utils import (
        get_wrapper_by_name,
        register_custom_envs,
)

# do NOT register mujoco envs, not using mujoco
# import cs285.envs

from stable_baselines3.common.env_util import make_atari_env

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        # register_custom_envs()
        # originally make atari env returns a wrapper with list of envs.
        # we only want one env, so pop to get the environment.
        self.env = make_atari_env(self.params['env_name']).envs.pop()
        self.eval_env = make_atari_env(self.params['env_name']).envs.pop()

        # initial status
        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')

        self.env.seed(seed)
        self.eval_env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.envs[0].metadata.keys():
            self.fps = self.env.envs[0].metadata['video.frames_per_second']
        else:
            self.fps = 10


        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          buffer_name=None,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        # print_period = 1000 if isinstance(self.agent, DistillationAgent) else 1
        print_period = self.params['batch_size']

        # teacher eval
        self.evaluate_run_policy(self.agent.teacher)

        eval_env_monitor = get_wrapper_by_name(self.eval_env, "Monitor")
        eval_episode_rewards = eval_env_monitor.get_episode_rewards()
        eval_returns = eval_episode_rewards[-self.params['eval_batch_size']:]

        self.teacher_avg_return = np.mean(eval_returns)

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            # Adapted from hw3 DQN for atari environment stepping.
            # FIXME [Path Issue]: Resolve paths issue (pls walk through execution of homework code)
            self.agent.step_env()  # this adds to the replay buffer in the agent
            envsteps_this_batch = 1
            train_video_paths = None
            paths = None
            self.total_envsteps += envsteps_this_batch

            # train agent (using sampled data from replay buffer)
            all_logs = []
            if itr % print_period == 0:
                print("\nTraining agent...")
            # train the agent
            all_logs = self.train_agent()

            # log
            if (self.logvideo or self.logmetrics):
                print('\nBeginning logging procedure...')
                self.perform_dqn_logging(np.array(all_logs))

            # Log densities and output trajectories
            # TODO: make sure we can use later
            # if isinstance(self.agent, ExplorationOrExploitationAgent) and (itr % print_period == 0):
            #     self.dump_density_graphs(itr)

        if self.params['save_params']:
            self.agent.actor.save(f"{self.params['logdir']}/student_chkpt_n_iter{n_iter}.pt")

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample, save_expert_data_to_disk=False):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # This if statement does nothing, it is old
        if itr == 0:
            if initial_expertdata is not None:
                paths = pickle.load(open(self.params['expert_data'], 'rb'))
                return paths, 0, None
            if save_expert_data_to_disk:
                num_transitions_to_sample = self.params['batch_size_initial']

        # collect data to be used for training
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, num_transitions_to_sample, self.params['ep_len'])

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        train_video_paths = None
        if self.logvideo:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        if save_expert_data_to_disk and itr == 0:
            with open('expert_data_{}.pkl'.format(self.params['env_name']), 'wb') as file:
                pickle.dump(paths, file)

        return paths, envsteps_this_batch, train_video_paths

    def collect_eval_trajectories(self, collect_policy, num_transitions_to_sample):
        """
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # collect data to be used for eval
        print("\nCollecting data to be used for eval...")
        # print(self.params['ep_len'])
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.eval_env, 
            collect_policy, 
            num_transitions_to_sample, 
            self.params['ep_len']
        )

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        train_video_paths = None
        if self.logvideo:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.eval_env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################

    def evaluate_run_policy(self, policy):
        """
            Step the environment for [eval_batch_size] episodes
        """
        env = self.eval_env

        for _ in range(self.params['eval_batch_size']):
            ob = env.reset()
            done = False
            while not done:
                ac = policy.get_action(ob)
                ob, rew, done, _ = env.step(ac)
                # env.render()

            
    
    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        # episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)
        
        # evaluate the policy
        self.evaluate_run_policy(self.agent.actor)

        eval_env_monitor = get_wrapper_by_name(self.eval_env, "Monitor")
        eval_episode_rewards = eval_env_monitor.get_episode_rewards()
        eval_returns = eval_episode_rewards[-self.params['eval_batch_size']:]
        eval_episode_times = eval_env_monitor.get_episode_times()
        eval_ep_lens = eval_episode_times[-self.params['eval_batch_size']:]

        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"] = np.std(eval_returns)
        if len(eval_returns) > 0:
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
        logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens) # TODO: fix
        logs["Teacher_AverageReturn"] = self.teacher_avg_return

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()

    def dump_density_graphs(self, itr):
        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        filepath = lambda name: self.params['logdir']+'/curr_{}.png'.format(name)

        num_states = self.agent.replay_buffer.num_in_buffer - 2
        states = self.agent.replay_buffer.obs[:num_states]
        if num_states <= 0: return
        
        H, xedges, yedges = np.histogram2d(states[:,0], states[:,1], range=[[0., 1.], [0., 1.]], density=True)
        plt.imshow(np.rot90(H), interpolation='bicubic')
        plt.colorbar()
        plt.title('State Density')
        self.fig.savefig(filepath('state_density'), bbox_inches='tight')

        plt.clf()
        ii, jj = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
        obs = np.stack([ii.flatten(), jj.flatten()], axis=1)
        density = self.agent.exploration_model.forward_np(obs)
        density = density.reshape(ii.shape)
        plt.imshow(density[::-1])
        plt.colorbar()
        plt.title('RND Value')
        self.fig.savefig(filepath('rnd_value'), bbox_inches='tight')

        plt.clf()
        exploitation_values = self.agent.exploitation_critic.qa_values(obs).mean(-1)
        exploitation_values = exploitation_values.reshape(ii.shape)
        plt.imshow(exploitation_values[::-1])
        plt.colorbar()
        plt.title('Predicted Exploitation Value')
        self.fig.savefig(filepath('exploitation_value'), bbox_inches='tight')

        plt.clf()
        exploration_values = self.agent.exploration_critic.qa_values(obs).mean(-1)
        exploration_values = exploration_values.reshape(ii.shape)
        plt.imshow(exploration_values[::-1])
        plt.colorbar()
        plt.title('Predicted Exploration Value')
        self.fig.savefig(filepath('exploration_value'), bbox_inches='tight')
