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
        
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.n_layers = hparams['n_layers']
        self.size = hparams['size']
        self.optimizer_spec = optimizer_spec

        # ICM: intrinsic curiosity module
        # forward network: given a_t, phi(s_t), predict phi(s_t+1)
        # inverse network: given phi(s_t), phi(s_t+1), predict a_t
        self.feat_size = 1152
        self.beta = hparams['curiosity_beta']

        self.feat_encoder = ptu.build_feat_encoder(self.ob_dim[-1])

        # f(a_t, phi(s_t)) = phi(s_t+1)
        # Forward is 2 hidden layers of 288, 256. Our feat sizes are 1152
        self.forward_net = ptu.build_mlp(self.feat_size+self.ac_dim, self.feat_size, 2, self.feat_size)
        self.forward_loss = nn.MSELoss(reduction='mean')

        # f(phi(s_t), phi(s_t+1)) = a_t
        # Inverse is 1 hidden layer of 256 units
        if "no_inverse_net" not in hparams:
            self.inverse_net = ptu.build_mlp(self.feat_size*2, self.ac_dim, 1, 256)
            self.inverse_loss = nn.CrossEntropyLoss(reduction='mean')

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

    def to_feature(self, obs: torch.Tensor):
        return self.feat_encoder(obs)

    # return s_{t+1}, s^{hat}_{t+1}, a^{hat}_t
    def forward(self, ob_no, next_ob_no, ac_na):
        obs, acs, next_obs = ptu.from_numpy_obs_ac_next(ob_no, ac_na, next_ob_no)

        # Get phi(s_t), phi(s_t+1)
        enc_obs = self.to_feature(obs)
        enc_next_obs = self.to_feature(next_obs)

        # forward model
        # f(a_t, phi(s_t)) = phi(s_t+1)
        pred_enc_next_obs = self.forward_net(
                                            torch.cat(
                                                        (enc_obs, F.one_hot(acs.long(),self.ac_dim)), 
                                                        dim=1
                                                    )
                                            )

        # inverse model
        # f(phi(s_t), phi(s_t+1)) = a_t
        pred_acs = self.inverse_net(
                                        torch.cat(
                                                    (enc_obs, enc_next_obs),
                                                    dim =1
                                        )
                                    )


        return enc_next_obs, pred_enc_next_obs, pred_acs

    def forward_np(self, ob_no: np.ndarray, next_ob_no:np.ndarray, ac_na:np.ndarray):
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        ac_na = ptu.from_numpy(ac_na)

        eno, peno, pa = self(ob_no)
        return ptu.to_numpy(eno), ptu.to_numpy(peno), ptu.to_numpy(pa)

    # get the intrinsic reward
    # intrinsic reward = difference between (phi(s_t+1), phi(pred_s_t+1))
    def get_intrinsic_reward(self, ob_no, next_ob_no, ac_na):   

        # Run through forward, inverse model 
        enc_next_obs, pred_enc_next_obs, _ = self.forward(ob_no, next_ob_no, ac_na)

        intrinsic_reward = self.forward_loss(pred_enc_next_obs, enc_next_obs)

        return intrinsic_reward.detach()


    # forward: MSE between predicted next state and actual next state
    # inverse: Cross Entropy between predicted actions and actual actions
    def update(self, ob_no, next_ob_no, ac_na):
        obs, acs, next_obs = ptu.from_numpy_obs_ac_next(ob_no, ac_na, next_ob_no)
        acs = acs.long()

        enc_next_obs, pred_enc_next_obs, pred_acs = self.forward(obs, next_obs, acs)

        # forward dynamics (predict next state given s_t and a_t)
        forward_loss = self.forward_loss(pred_enc_next_obs, enc_next_obs)

        # Inverse dynamics (predict action which took us from s_t to s_t+1)
        inverse_loss = self.inverse_loss(pred_acs, acs)

        # Get Loss = Beta * L_forward + (1-Beta) * L_inverse
        loss = self.beta * forward_loss + (1-self.beta) * inverse_loss
        
        # update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return loss.item()
