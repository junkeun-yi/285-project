from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
from .icm_model import ICMModel
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


class RandomFeatModel(ICMModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):

        # drop some kwargs that don't apply to the supers
        kwargs_copy = deepcopy(kwargs)
        if 'flatten_input' in kwargs_copy:
            del kwargs_copy['flatten_input']
        
        super().__init__(hparams, optimizer_spec, **kwargs)


        self.feat_encoder = ptu.build_feat_encoder(self.ob_dim[-1]) #Using same feat encoder but freezing!
        # Freeze encoder for random feat
        for param in self.feat_encoder.parameters():
            param.requires_grad = False
        
        self.check_frozen_feat()
    
    def check_frozen_feat(self):
        # Helper test to ensure we are not updating feats
        for param in self.feat_encoder.parameters():
            assert not param.requires_grad


    def to_feature(self, obs):
        return self.feat_encoder(ptu.from_numpy(obs))

    # return s_{t+1}, s^{hat}_{t+1}, a^{hat}_t
    def forward(self, ob_no, next_ob_no, ac_na):
        obs = ob_no
        next_obs = next_ob_no
        acs = ac_na

        enc_obs = self.to_feature(obs)
        enc_next_obs = self.to_feature(next_obs)

        # forward model
        # f(a_t, phi(s_t)) = phi(s_t+1)
        pred_enc_next_obs = torch.cat((enc_obs, acs)) # TODO: check dimensions
        pred_enc_next_obs = self.forward_net(pred_enc_next_obs)

        return enc_next_obs, pred_enc_next_obs


    def forward_np(self, ob_no, next_ob_no, ac_na):
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        ac_na = ptu.from_numpy(ac_na)
        enc_next_obs, pred_next_obs= self(ob_no)
        return ptu.to_numpy(enc_next_obs), ptu.to_numpy(pred_next_obs)

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
        intrinsic_reward = self.eta * self.forward_loss(pred_enc_next_obs, enc_next_obs)

        return ptu.to_numpy(intrinsic_reward)


    # update the ICM model
    # TODO: check curiosity code to see what losses were used
    # forward: MSE between predicted next state and actual next state
    # inverse: Cross Entropy between predicted actions and actual actions
    def update(self, ob_no, next_ob_no, ac_na):

        enc_next_obs, pred_enc_next_obs, pred_acs = self.forward(ob_no, next_ob_no, ac_na)

        # forward dynamics
        forward_loss = self.forward_loss(pred_enc_next_obs, enc_next_obs)

        # inverse dynamics
        inverse_loss = self.inverse_loss(pred_acs, ac_na)

        loss = self.beta * forward_loss + (1 - self.beta)*inverse_loss

        # update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
