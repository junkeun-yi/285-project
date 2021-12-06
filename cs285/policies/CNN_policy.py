import abc
import itertools
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from torch.nn import KLDivLoss

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy



class CNNPolicyDistillationStudent(BasePolicy, nn.Module):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 temperature=0.01,
                 **kwargs,
                 ):
        # drop some kwargs that don't apply to the supers
        kwargs_copy = deepcopy(kwargs)
        if 'flatten_input' in kwargs_copy:
            del kwargs_copy['flatten_input']

        super().__init__(**kwargs_copy)

        # init vars
        self.ac_dim = ac_dim
        self.ob_channels = ob_dim[-1] #Assuming ob_dim(H,W,C)
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training

        if self.discrete:
            self.logits_na = ptu.build_policy_CNN(
                input_channels = self.ob_channels,
                output_size = self.ac_dim
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            raise NotImplementedError

        self.T = temperature
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def forward(self, observation: torch.FloatTensor):
        logits = self.logits_na(observation)
        action_distribution = distributions.Categorical(logits=logits)
        return action_distribution
    
    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        observation = ptu.from_numpy(obs)
        action_distribution = self(observation)
        action = action_distribution.sample()  # don't bother with rsample
        return ptu.to_numpy(action)

    def update(self, observations, actions, act_logits_teacher, adv_n=None):
        if adv_n is not None:
            assert False
        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations)
        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions)
        if isinstance(adv_n, np.ndarray):
            adv_n = ptu.from_numpy(adv_n)
        
        # action_dist = self.forward(observations.view(observations.shape[0],-1))
        action_dist = self.forward(observations)
        act_logits_student = action_dist.logits
        
        kl_loss = KLDivLoss(reduction='batchmean')
        loss = kl_loss(
            F.log_softmax(act_logits_student, dim=1), 
            F.softmax(act_logits_teacher / self.T, dim=1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

