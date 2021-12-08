from typing import Tuple
from cs285.infrastructure import pytorch_util as ptu
from .random_feat import RandomFeatCuriosity
import torch
from torch import nn
import numpy as np

class ICMModel(RandomFeatCuriosity):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        
        super().__init__(hparams, optimizer_spec, **kwargs)
        self.beta = hparams['icm_beta']

        # Inverse network: given phi(s_t) and phi(s_{t+1}), predict a_t
        # Inverse is 1 hidden layer of 256 units
        self.inverse_net = ptu.build_mlp(self.feat_size * 2, self.ac_dim, 1, 256)
        self.inverse_loss = nn.CrossEntropyLoss(reduction='mean')
        self.inverse_net.to(ptu.device) #Move inverse net to device

    # return s_{t}, s_{t+1}, s^{hat}_{t+1}, a^{hat}_t
    def forward(self, obs: torch.Tensor, 
                next_obs: torch.Tensor, acs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Get forward prediction of next state (returns encoding s_t, encoding s_{t+1}, predicted s_{t+1})
        enc_obs, enc_next_obs, pred_enc_next_obs = super().forward(obs, next_obs, acs)

        # Run Inverse model
        # f(phi(s_t), phi(s_t+1)) = a_t
        pred_acs = self.inverse_net(
                                        torch.cat(
                                                    (enc_obs, enc_next_obs),
                                                    dim =1
                                                )
                                    )

        return enc_obs, enc_next_obs, pred_enc_next_obs, pred_acs

    # forward: MSE between predicted next state and actual next state
    # inverse: Cross Entropy between predicted actions and actual actions
    def update(self, ob_no: np.ndarray, next_ob_no: np.ndarray, ac_na: np.ndarray):
        # Convert to tensor
        obs, acs, next_obs = ptu.from_numpy_obs_ac_next(ob_no, ac_na, next_ob_no)
        acs = acs.long()

        # Get next_obs, predicted next_obs, predicted action (from inverse model)
        _, enc_next_obs, pred_enc_next_obs, pred_acs = self(obs, next_obs, acs)

        # forward dynamics (predict next state given s_t and a_t)
        forward_loss = self.forward_loss(pred_enc_next_obs, enc_next_obs)

        # Inverse dynamics (predict action which took us from s_t to s_{t+1})
        inverse_loss = self.inverse_loss(pred_acs, acs)

        # Get Loss = Beta * L_forward + (1-Beta) * L_inverse
        loss = self.beta * forward_loss + (1 - self.beta) * inverse_loss
        
        # update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
