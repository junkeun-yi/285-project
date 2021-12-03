import abc
import itertools
from gym.core import ObservationWrapper
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy

from stable_baselines3 import PPO
class DistillationTeacherPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
            policy, 
            env, 
            learning_rate=lambda x: 2.5*1e-4*x, 
            n_steps=128, 
            batch_size=32*8, 
            n_epochs=3, 
            gamma=0.99, 
            gae_lambda=0.95, 
            clip_range=lambda x: 0.1*x, 
            clip_range_vf=None, 
            ent_coef=0.001, 
            vf_coef=1, 
            max_grad_norm=0.5, 
            use_sde=False, 
            sde_sample_freq=- 1, 
            target_kl=None, 
            tensorboard_log=None,
            create_eval_env=False, 
            policy_kwargs=None, 
            verbose=0, 
            seed=None, 
            device='auto', 
            _init_setup_model=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # create model based on initial parameters.
        self.model = PPO(
            policy, 
            env, 
            learning_rate=learning_rate, 
            n_steps=n_steps, 
            batch_size=batch_size, 
            n_epochs=n_epochs, 
            gamma=gamma, 
            gae_lambda=gae_lambda, 
            clip_range=clip_range, 
            clip_range_vf=clip_range_vf, 
            ent_coef=ent_coef, 
            vf_coef=vf_coef, 
            max_grad_norm=max_grad_norm, 
            use_sde=use_sde, 
            sde_sample_freq=sde_sample_freq, 
            target_kl=target_kl, 
            tensorboard_log=tensorboard_log, 
            create_eval_env=create_eval_env, 
            policy_kwargs=policy_kwargs, 
            verbose=verbose, 
            seed=seed, 
            device=device, 
            _init_setup_model=_init_setup_model
        )

    ##################################

    def load(self, filepath):
        # load stable_baselines3 model
        self.model = PPO.load(filepath)

    ##################################

    def save(self, filepath):
        self.model.save(filepath)

    ##################################

    def get_act_logits(self, obs: torch.FloatTensor):
        action_dist = self.forward(obs)
        return action_dist.distribution.logits.detach()

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = ptu.from_numpy(observation)
        action_distribution = self(observation)
        action = action_distribution.sample()  # don't bother with rsample
        return ptu.to_numpy(action)

    ####################################
    ####################################

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # Return the action distribution of the teacher policy on the given observation
    def forward(self, observations):

        action_dist = self.model.policy.get_distribution(
            self.model.policy.obs_to_tensor(observations)[0])
        return action_dist

    ####################################
    ####################################


#####################################################
#####################################################