import abc
import itertools
from copy import deepcopy
from typing import Optional
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from torch.nn import KLDivLoss
from src.exploration.rnd_model import RNDModel

from src.infrastructure import pytorch_util as ptu
from src.policies.base_policy import BasePolicy

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
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_channels = ob_dim[-1] #Assuming ob_dim is (H,W,C)
        self.learning_rate = learning_rate
        self.training = training
        self.discrete = discrete
        
        if self.discrete:
            self.logits_na = ptu.build_policy_CNN(
                input_channels=self.ob_channels, 
                output_size=self.ac_dim,
            )
            self.logits_na.to(ptu.device)
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            raise NotImplementedError

    
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

    def update(self, loss):
        # Update policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


