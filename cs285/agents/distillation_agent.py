from collections import OrderedDict

from cs285.critics.dqn_critic import DQNCritic
from cs285.critics.cql_critic import CQLCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.policies.teacher_policy import DistillationTeacherPolicy
from cs285.policies.MLP_policy import MLPPolicyDistillationStudent
from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs285.exploration.rnd_model import RNDModel
from cs285.agents.base_agent import BaseAgent
from .dqn_agent import DQNAgent
import numpy as np
import cs285.infrastructure.pytorch_util as ptu
from cs285.exploration.icm_model import ICMModel


class DistillationAgent(DQNAgent):
    def __init__(self, env, agent_params):
        super(DistillationAgent, self).__init__(env, agent_params)
        self.env = env
        self.agent_params = agent_params
        
        # Retrieve teacher policy
        self.teacher = DistillationTeacherPolicy(
            self.agent_params['teacher_chkpt']
        )

        # setup
        self.critic = None

        print(self.agent_params['ob_dim'])

        self.actor = MLPPolicyDistillationStudent(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            temperature=self.agent_params['temperature'],
            flatten_input=True   # build mlp input as width * height
        )

        # create exploration model for additional data gathering.
        # using curiosity.
        self.exploration_model = ICMModel(agent_params, self.optimizer_spec, flatten_input=True)

        self.eval_policy = self.actor
        # self.eval_policy = self.teacher

        # TODO: utilize these parameters for exploration.
        # changed to ReplayBuffer because using add_rollouts and sample_recent_data
        # self.num_exploration_steps = agent_params['num_exploration_steps']
        # self.offline_exploitation = agent_params['offline_exploitation']

        # self.exploitation_critic = CQLCritic(agent_params, self.optimizer_spec)
        # self.exploration_critic = DQNCritic(agent_params, self.optimizer_spec)

        # self.exploration_model = RNDModel(agent_params, self.optimizer_spec)
        # self.explore_weight_schedule = agent_params['explore_weight_schedule']
        # self.exploit_weight_schedule = agent_params['exploit_weight_schedule']

        # self.actor = ArgMaxPolicy(self.exploration_critic)
        # self.eval_policy = ArgMaxPolicy(self.exploitation_critic)
        # self.exploit_rew_shift = agent_params['exploit_rew_shift']
        # self.exploit_rew_scale = agent_params['exploit_rew_scale']
        # self.eps = agent_params['eps']

    # Training function for naive distillation
    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            # retrieve teacher's action logits on observations
            ac_logits_teacher = self.teacher.get_act_logits(ob_no)
            # print(ac_logits_teacher)

            # update the student
            kl_loss = self.actor.update(ob_no, ac_na, ac_logits_teacher)

            # add curiosity update here.
            # TODO: what to do with the curiosity model ? need to like integrate it to the original policy too.
            # reward = a*distillation reward + b*exploration reward right?
            # so distillation reward up if kl div high
            # and exploration reward up if intrinsic reward high
            # how to combine both ?

            log['kl_div_loss'] = kl_loss

            self.num_param_updates += 1

        self.t += 1
        return log

    # override dqn_agent step_env to avoid epsilon greedy
    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        if self.t == 0:
            # first action random to appease MemoryBuffer warnings
            # TODO: is there a better way to do this ?
            action = np.random.randint(self.num_actions)
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(processed)

        self.last_obs, reward, done, info = self.env.step(action)

        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()

############################################################
############################################################

    # # TODO: add curiosity (ICM) and run on this function.
    # def train_explore(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
    #     log = {}

    #     if self.t > self.num_exploration_steps:
    #         # TODO: After exploration is over, set the student to optimize the extrinsic critic
    #         #HINT: Look at method ArgMaxPolicy.set_critic
    #         self.actor.set_critic(self.exploitation_critic)

    #     if (self.t > self.learning_starts
    #             and self.t % self.learning_freq == 0
    #             and self.replay_buffer.can_sample(self.batch_size)
    #     ):

    #         # Get Reward Weights
    #         # TODO: Get the current explore reward weight and exploit reward weight
    #         #       using the schedule's passed in (see __init__)
    #         # COMMENT: Until part 3, explore_weight = 1, and exploit_weight = 0
    #         # explore_weight = 1 # self.explore_weight_schedule.value(self.t)
    #         # exploit_weight = 0 # self.exploit_weight_schedule.value(self.t)
    #         explore_weight = self.explore_weight_schedule.value(self.t)
    #         exploit_weight = self.exploit_weight_schedule.value(self.t)

    #         # Run Exploration Model #
    #         # TODO: Evaluate the exploration model on s' to get the exploration bonus
    #         # HINT: Normalize the exploration bonus, as RND values vary highly in magnitude
    #         expl_bonus = self.exploration_model.forward_np(next_ob_no)
    #         expl_bonus = normalize(expl_bonus, np.mean(expl_bonus), np.std(expl_bonus))

    #         # Reward Calculations #
    #         # TODO: Calculate mixed rewards, which will be passed into the exploration critic
    #         # HINT: See doc for definition of mixed_reward
    #         mixed_reward = explore_weight * expl_bonus + exploit_weight * re_n

    #         # TODO: Calculate the environment reward
    #         # HINT: For part 1, env_reward is just 're_n'
    #         #       After this, env_reward is 're_n' shifted by self.exploit_rew_shift,
    #         #       and scaled by self.exploit_rew_scale
    #         env_reward = re_n # (re_n + self.exploit_rew_shift) * self.exploit_rew_scale
    #         # env_reward = (re_n + self.exploit_rew_shift) * self.exploit_rew_scale

    #         # Update Critics And Exploration Model #

    #         # TODO 1): Update the exploration model (based off s')
    #         # TODO 2): Update the exploration critic (based off mixed_reward)
    #         # TODO 3): Update the exploitation critic (based off env_reward)
    #         expl_model_loss = self.exploration_model.update(next_ob_no)
    #         exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)
    #         exploitation_critic_loss = self.exploitation_critic.update(ob_no, ac_na, next_ob_no, env_reward, terminal_n)

    #         # Target Networks #
    #         if self.num_param_updates % self.target_update_freq == 0:
    #             # TODO: Update the exploitation and exploration target networks
    #             self.exploitation_critic.update_target_network()
    #             self.exploration_critic.update_target_network()

    #         # Logging #
    #         log['Exploration Critic Loss'] = exploitation_critic_loss['Training Loss']
    #         log['Exploitation Critic Loss'] = exploration_critic_loss['Training Loss']
    #         log['Exploration Model Loss'] = expl_model_loss

    #         if self.exploitation_critic.cql_alpha >= 0:
    #             log['Exploitation Data q-values'] = exploitation_critic_loss['Data q-values']
    #             log['Exploitation OOD q-values'] = exploitation_critic_loss['OOD q-values']
    #             log['Exploitation CQL Loss'] = exploitation_critic_loss['CQL Loss']

    #         self.num_param_updates += 1

    #     self.t += 1
    #     return log

############################################################
############################################################