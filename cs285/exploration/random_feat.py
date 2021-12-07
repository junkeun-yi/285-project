from cs285.infrastructure import pytorch_util as ptu
from .icm_model import ICMModel
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
        
        hparams["no_inverse_net"] = True #To save mem (hack!)
        super().__init__(hparams, optimizer_spec, **kwargs)

        # Freeze encoder for random feat
        for param in self.feat_encoder.parameters():
            param.requires_grad = False


    def to_feature(self, obs: torch.Tensor):
        return self.feat_encoder(obs)

    # return s_{t+1}, s^{hat}_{t+1}, a^{hat}_t
    def forward(self, ob_no, next_ob_no, ac_na):
        obs, acs, next_obs = ptu.from_numpy_obs_ac_next(ob_no, ac_na, next_ob_no)

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

        return enc_next_obs, pred_enc_next_obs


    def forward_np(self, ob_no: np.ndarray, next_ob_no:np.ndarray, ac_na:np.ndarray):
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        ac_na = ptu.from_numpy(ac_na)
        enc_next_obs, pred_next_obs = self(ob_no)
        return ptu.to_numpy(enc_next_obs), ptu.to_numpy(pred_next_obs)

    # get the intrinsic reward
    # intrinsic reward = difference between (phi(s_t+1), phi(pred_s_t+1))
    def get_intrinsic_reward(self, ob_no, next_ob_no, ac_na):
        obs, acs, next_obs = ptu.from_numpy_obs_ac_next(ob_no, ac_na, next_ob_no)
        
        enc_next_obs, pred_enc_next_obs = self.forward(obs, next_obs, acs)

        intrinsic_reward = self.forward_loss(pred_enc_next_obs, enc_next_obs)

        return ptu.to_numpy(intrinsic_reward)


    # forward: MSE between predicted next state and actual next state
    def update(self, ob_no, next_ob_no, ac_na):

        enc_next_obs, pred_enc_next_obs = self.forward(ob_no, next_ob_no, ac_na)

        # forward dynamics
        forward_loss = self.forward_loss(pred_enc_next_obs, enc_next_obs)

        loss = forward_loss

        # update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
