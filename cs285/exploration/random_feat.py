from typing import List, Tuple
from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class RandomFeatCuriosity(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.n_layers = hparams['n_layers']
        self.size = hparams['size']
        self.optimizer_spec = optimizer_spec

        self.feat_size = 1152
        
        # Get CNN Encoder for state s_t
        self.feat_encoder = ptu.build_feat_encoder(self.ob_dim[-1])

        # Forward network: given a_t, phi(s_t), predict phi(s_{t+1})
        # Forward is originally (from Pathak et al.) 2 hidden layers of 288, 256. 
        # Our feat sizes are 1152, so we did input->1152->1152->output
        self.forward_net = ptu.build_mlp(self.feat_size + self.ac_dim, self.feat_size, 2, self.feat_size)
        self.forward_loss = nn.MSELoss(reduction='mean')

        # Freeze encoder for random feat (not icm)
        if not hparams['use_icm']:
            print("Freezing Encoder!")
            for param in self.feat_encoder.parameters():
                param.requires_grad = False
        
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
    
    # Returns phi(s_t), i.e. encodes the state via CNN
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.feat_encoder(obs)

    # return s_t, s_{t+1}, s^{hat}_{t+1}
    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor, acs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Encode s_t, s_t+1 to get phi(s_t), phi(s_t+1)
        enc_obs = self.encode(obs)
        enc_next_obs = self.encode(next_obs)

        # forward model
        # f(a_t, phi(s_t)) = phi(s_t+1)
        pred_enc_next_obs = self.forward_net(
                                    torch.cat(
                                                (enc_obs, F.one_hot(acs.long(),self.ac_dim)), 
                                                dim=1
                                            )
                                    )

        return enc_obs, enc_next_obs, pred_enc_next_obs

    # Returns LIST (to handle varying number of forward outputs)
    def forward_np(self, ob_no: np.ndarray, next_ob_no:np.ndarray, ac_na:np.ndarray) -> List[np.ndarray]:
        # Convert to tensor
        obs = ptu.from_numpy(ob_no)
        next_obs = ptu.from_numpy(next_ob_no)
        acs = ptu.from_numpy(ac_na)

        # Run through forward and conver to numpy
        return [ptu.to_numpy(y) for y in self(obs, next_obs, acs)]

    # get the intrinsic reward
    # intrinsic reward = difference between (phi(s_t+1), phi(pred_s_t+1)). We don't use inverse model here.
    def get_intrinsic_reward(self, ob_no: np.ndarray, next_ob_no: np.ndarray, ac_na: np.ndarray):
        # Convert to tensor
        obs, acs, next_obs = ptu.from_numpy_obs_ac_next(ob_no, ac_na, next_ob_no)
        
        # Get next state, predicted next state
        enc_next_obs = self.encode(next_obs)
        pred_enc_next_obs = self.forward_net(
                                            torch.cat((self.encode(obs), F.one_hot(acs.long(),self.ac_dim)), dim=1)
                                        )

        # intrinsic reward = next state prediction error
        intrinsic_reward = self.forward_loss(pred_enc_next_obs, enc_next_obs)

        return intrinsic_reward.detach()


    # forward: MSE between predicted next state and actual next state
    def update(self, ob_no: np.ndarray, next_ob_no: np.ndarray, ac_na: np.ndarray):
        # Convert to tensor
        obs, acs, next_obs = ptu.from_numpy_obs_ac_next(ob_no, ac_na, next_ob_no)

        # Get next_observation, next_observation prediction (from forward model)
        _, enc_next_obs, pred_enc_next_obs = self(obs, next_obs, acs)

        # Get forward prediction loss
        loss = self.forward_loss(pred_enc_next_obs, enc_next_obs)

        # update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
