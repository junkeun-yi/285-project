from math import log
from cs285.agents.dqn_agent import DQNAgent
from cs285.policies.argmax_policy import ArgMaxPolicy
from .explore_or_exploit_agent import ExplorationOrExploitationAgent
from cs285.exploration.icm_model import ICMModel
from cs285.policies.teacher_policy import DistillationTeacherPolicy
from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs285.critics.dqn_critic import DQNCritic
from cs285.infrastructure.utils import *
import cs285.infrastructure.pytorch_util as ptu
import torch
from torch.nn import functional as F
from torch.nn import KLDivLoss
from torchvision import transforms
import numpy as np

class DistillationAgent(DQNAgent):
    def __init__(self, env, agent_params):
        super(DistillationAgent, self).__init__(env, agent_params)

        self.replay_buffer = MemoryOptimizedReplayBuffer(100000, 1, float_obs=True)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        self.offline_exploitation = agent_params['offline_exploitation']

        self.exploration_critic = DQNCritic(agent_params, self.optimizer_spec)
        
        self.explore_weight_schedule = agent_params['explore_weight_schedule']
        self.exploit_weight_schedule = agent_params['exploit_weight_schedule']
        
        self.actor = ArgMaxPolicy(self.exploration_critic)
        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = agent_params['eps']
        
        self.curiosity = self.agent_params['use_curiosity']
        if self.curiosity:
            print("USING ICM")
            self.exploration_model = ICMModel(agent_params, self.optimizer_spec)
        
        # self.exploitation_critic = None
        # Retrieve teacher policy
        self.teacher = DistillationTeacherPolicy(
            self.agent_params['teacher_chkpt']
        )
        
        # Get Temperature
        self.T = self.agent_params['temperature']

        # Flag if using uncertainty weighted distillation
        self.uncertainity_aware = self.agent_params["use_uncertainty"]
        if self.uncertainity_aware:
            print("Using Uncertainty!")
            self.to_img = transforms.ToPILImage()
            self.data_aug = [
                                transforms.ColorJitter(brightness=0.05),
                                transforms.Compose([
                                                        transforms.RandomAffine(10),
                                                        transforms.Resize(self.agent_params['ob_dim'][:-1]) #ob_dim is (H,W,C) need (H,W)            
                                ]),
                                transforms.Compose([
                                                        transforms.Pad(2),
                                                        transforms.Resize(self.agent_params['ob_dim'][:-1])
                                ])
            ]

        self.eval_policy = self.actor

    def train(self, ob_no: np.ndarray, ac_na: np.ndarray, re_n: np.ndarray, next_ob_no: np.ndarray, terminal_n: np.ndarray):
        
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            # Calculate Distill Reward
            ac_logits_teacher = self.teacher.get_act_logits(ob_no)
            act_logits_student = ptu.from_numpy(self.exploration_critic.qa_values(ob_no))

            student_val = F.log_softmax(act_logits_student, dim=1)
            teacher_val = F.softmax(ac_logits_teacher / self.T, dim=1)

            # print(f"Shapes: {student_val.shape}, {teacher_val.shape}")

            kl_div = F.kl_div(student_val, teacher_val, reduction='none').sum(axis=1)

            log['kl_div'] = kl_div.sum()
            
            # print(kl_div.shape)
            
            distill_reward = ptu.to_numpy(-kl_div)

            # Get Reward Weights
            #       using the schedule's passed in (see __init__)
            explore_weight = self.explore_weight_schedule.value(self.t)
            if self.uncertainity_aware:
                explore_weight = self.get_teacher_uncertainty(ob_no, ac_logits_teacher)
            log['Explore Weight'] = explore_weight
            exploit_weight = self.exploit_weight_schedule.value(self.t)

            # Run Exploration Model #
            expl_bonus = 0
            if self.curiosity:
                expl_bonus = ptu.to_numpy(self.exploration_model.get_intrinsic_reward(ob_no, next_ob_no, ac_na))
                expl_bonus = normalize(expl_bonus, np.mean(expl_bonus), np.std(expl_bonus))
                expl_model_loss = self.exploration_model.update(ob_no, next_ob_no, ac_na)
                log['Exploration Model Loss'] = expl_model_loss

            # Reward Calculations #
            mixed_reward = explore_weight * expl_bonus + exploit_weight * distill_reward

            # print(ob_no.shape, ac_na.shape, next_ob_no.shape, mixed_reward.shape, terminal_n.shape)

            # Update Critic Model #
            exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)

            # Target Networks #
            if self.num_param_updates % self.target_update_freq == 0:
                self.exploration_critic.update_target_network()

            # Logging #
            log['DQN Critic Loss'] = exploration_critic_loss['Training Loss']

            self.num_param_updates += 1

        self.t += 1
        return log
    
    # Returns avg cross entropy between teacher original action logit for ob_no and teacher actions under augmented ob_no
    def get_teacher_uncertainty(self, ob_no: np.ndarray, teacher_ac_logits: torch.Tensor) -> torch.Tensor():
        
        teacher_ac_probs = F.softmax(teacher_ac_logits, dim=1)
        obs = ptu.from_numpy(ob_no).permute((0,3,1,2))
        uncertainty = 0
        for data_aug in self.data_aug:
            obs_prime = data_aug(obs).permute((0,2,3,1))
            np_obs_prime = ptu.to_numpy(obs_prime)
            teacher_aug_logit = self.teacher.get_act_logits(np_obs_prime)
            log_prob_aug = F.log_softmax(teacher_aug_logit)
            if uncertainty is None:
                uncertainty = (-teacher_ac_probs * log_prob_aug).sum()
            else:
                uncertainty += (-teacher_ac_probs * log_prob_aug).sum()

        return uncertainty.cpu().item() / (ob_no.shape[0] * len(self.data_aug))
        
    def get_action(self, ob):
        return self.actor.get_action(ob)

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        perform_random_action = np.random.random() < self.eps or self.t < self.learning_starts

        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(processed)

        next_obs, reward, done, info = self.env.step(action)
        self.last_obs = next_obs.copy()

        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()