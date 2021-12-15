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

from src.infrastructure import pytorch_util as ptu
from src.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        # drop some kwargs that don't apply to the supers
        kwargs_copy = deepcopy(kwargs)
        if 'flatten_input' in kwargs_copy:
            del kwargs_copy['flatten_input']

        super().__init__(**kwargs_copy)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.flatten_input = kwargs.get('flatten_input', False)  # build mlp input as width * height

        print(f"flatten_input: {self.flatten_input}")

        if self.flatten_input:  # build mlp input as width * height
            temp_ob_dim = 1
            for v in self.ob_dim:
                temp_ob_dim *= v
            self.ob_dim = temp_ob_dim

        if self.discrete:
            print(f"input size {self.ob_dim}, output size {self.ac_dim}")
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim, 
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

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

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # observation = observation[None]  # TODO [flatten_input] is this correct?
        observation = observation.reshape((-1, self.ob_dim))  # TODO [flatten_input] is this correct?
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution

    ####################################
    ####################################


#####################################################
#####################################################

class MLPPolicyDistillationStudent(MLPPolicy):
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
        super().__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, nn_baseline, **kwargs)
        self.T = temperature

        # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        observation = ptu.from_numpy(obs)
        # action_distribution = self(observation.view(observation.shape[0], -1))
        observation = observation.view(-1)
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
        action_dist = self.forward(observations)  # NOTE after changing to flatten_input, switched this back
        act_logits_student = action_dist.logits
        
        kl_loss = KLDivLoss(reduction='batchmean')
        loss = kl_loss(
            F.log_softmax(act_logits_student, dim=1), 
            F.softmax(act_logits_teacher / self.T, dim=1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class MLPPolicyAC(MLPPolicy):
    # MJ: cut acs_labels_na and qvals from the signature if they are not used
    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################

class MLPPolicyAWAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 lambda_awac=10,
                 **kwargs,
                 ):
        self.lambda_awac = lambda_awac
        super().__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, nn_baseline, **kwargs)
    
    def update(self, observations, actions, adv_n=None):
        if adv_n is None:
            assert False
        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations)
        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions)
        if isinstance(adv_n, np.ndarray):
            adv_n = ptu.from_numpy(adv_n)

        # TODO update the policy network utilizing AWAC update
        action_dist = self(observations)
        log_probs = action_dist.log_prob(actions)
        awac_weights = torch.exp((1/self.lambda_awac)*adv_n)
        actor_loss = -1 * torch.mean(log_probs * awac_weights)

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        return actor_loss.item()
