import abc
import itertools
from gym.core import ObservationWrapper
from torch import nn
from torch.nn import functional as F
from torch import optim
import os
import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy

from stable_baselines3 import PPO
class DistillationTeacherPolicy(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, teacher_ckpt, **kwargs):
        super().__init__(**kwargs)
        self.load(teacher_ckpt)

    ##################################

    def load(self, filepath):
        # load stable_baselines3 model
        self.model = PPO.load(os.path.join(os.getcwd(), filepath), device=ptu.device)

    ##################################

    def save(self, filepath):
        self.model.save(filepath)

    ##################################

    def get_act_logits(self, obs: np.ndarray):
        action_dist = self.forward(obs)
        return action_dist.distribution.logits.detach()

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        action_distribution = self(observation)
        action = action_distribution.sample()  # don't bother with rsample
        return ptu.to_numpy(action)

    ####################################
    ####################################

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # Return the action distribution of the teacher policy on the given observation
    def forward(self, observation: np.ndarray):
        obs = observation

        # action_dist = self.model.policy.get_distribution(ptu.from_numpy(obs))
        action_dist = self.model.policy.get_distribution(
            self.model.policy.obs_to_tensor(obs)[0])
        return action_dist

    ####################################
    ####################################


#####################################################
#####################################################