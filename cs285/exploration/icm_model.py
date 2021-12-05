from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class ICMModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):

        # drop some kwargs that don't apply to the supers
        kwargs_copy = deepcopy(kwargs)
        if 'flatten_input' in kwargs_copy:
            del kwargs_copy['flatten_input']
        
        super().__init__(**kwargs_copy)
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.n_layers = hparams['n_layers']
        self.size = hparams['size']
        self.optimizer_spec = optimizer_spec

        self.flatten_input = kwargs.get('flatten_input', False)  # flatten the input, (84 * 84 * 1)

        # ICM: intrinsic curiosity module
        # forward network: given a_t, phi(s_t), predict phi(s_t+1)
        # inverse network: given phi(s_t), phi(s_t+1), predict a_t
        # TODO: model specifications should follow original Curiosity paper.

        self.feat_size = 512 # TODO: make this a parameter.
        self.eta = 0.01 # TODO: make ICM ETA parameter

        # TODO: feature encoder should be a CNN
        # f(s_t) = phi(s_t)

        # TODO check, the input (84, 84, 1) is multiplied to become 7056 size input
        if self.flatten_input:  # build mlp input as width * height
            temp_ob_dim = 1
            for v in self.ob_dim:
                temp_ob_dim *= v
            self.ob_dim = temp_ob_dim
        self.to_feature = ptu.build_mlp(
            self.ob_dim, self.feat_size, self.n_layers, self.size
        )
        
        # TODO: fix
        # f(a_t, phi(s_t)) = phi(s_t+1)
        self.forward_net = nn.Linear(self.feat_size+self.ac_dim, self.feat_size)

        # TODO: fix
        # f(phi(s_t), phi(s_t+1)) = a_t
        self.inverse_net = ptu.build_mlp(self.feat_size*2, self.ac_dim, self.n_layers, self.size)

        # optimizer
        self.optimizer = self.optimizer_spec.constructor(
            self.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        # send to gpu if possible
        self.to(ptu.device)

    # return s_{t+1}, s^{hat}_{t+1}, a^{hat}_t
    def forward(self, ob_no, next_ob_no, ac_na):
        obs = ob_no
        next_obs = next_ob_no
        acs = ac_na

        if self.flatten_input:
            obs = obs.reshape((-1, self.ob_dim))
            next_obs = next_obs.reshape((-1, self.ob_dim))

        enc_obs = self.to_feature(obs)
        enc_next_obs = self.to_feature(next_obs)

        # inverse model
        # f(phi(s_t), phi(s_t+1)) = a_t
        pred_acs = torch.cat((enc_obs, enc_next_obs), 1) # TODO: check dimensions
        pred_acs = self.inverse_net(pred_acs)

        # forward model
        # f(a_t, phi(s_t)) = phi(s_t+1)
        pred_enc_next_obs = torch.cat((enc_obs, acs), 1) # TODO: check dimensions
        pred_enc_next_obs = self.forward_net(pred_enc_next_obs)

        return enc_next_obs, pred_enc_next_obs, pred_acs


    def forward_np(self, ob_no, next_ob_no, ac_na):
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        ac_na = ptu.from_numpy(ac_na)
        eno, peno, pa = self(ob_no)
        return ptu.to_numpy(eno), ptu.to_numpy(peno), ptu.to_numpy(pa)

    # get the intrinsic reward
    # intrinsic reward = difference between (phi(s_t+1), phi(pred_s_t+1))
    def get_intrinsic_reward(self, ob_no, next_ob_no, ac_na):
        if isinstance(ob_no, np.ndarray):
            ob_no = ptu.from_numpy(ob_no)
        if isinstance(ac_na, np.ndarray):
            next_ob_no = ptu.from_numpy(ac_na)
        if isinstance(ac_na, np.ndarray):
            ac_na = ptu.from_numpy(ac_na)
        
        enc_next_obs, pred_enc_next_obs, _ = self.forward(ob_no, next_ob_no, ac_na)

        # TODO: verify how the original ICM paper calculates intrinsic reward
        intrinsic_reward = self.eta * F.mse_loss(pred_enc_next_obs, enc_next_obs, reduction='none').mean(-1)

        return ptu.to_numpy(intrinsic_reward)


    # update the ICM model
    # TODO: check curiosity code to see what losses were used
    # forward: MSE between predicted next state and actual next state
    # inverse: Cross Entropy between predicted actions and actual actions
    def update(self, ob_no, next_ob_no, ac_na):

        enc_next_obs, pred_enc_next_obs, pred_acs = self.forward(ob_no, next_ob_no, ac_na)

        # forward dynamics
        forward_loss = nn.MSELoss()
        forward_loss = forward_loss(pred_enc_next_obs, enc_next_obs)

        # inverse dynamics
        inverse_loss = nn.CrossEntropyLoss()
        inverse_loss = inverse_loss(pred_acs, ac_na)

        loss = forward_loss + inverse_loss

        # update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
