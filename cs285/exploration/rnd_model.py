import numpy as np
from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.output_size = 5 #hparams['rnd_output_size']
        self.n_layers = 2 #hparams['rnd_n_layers']
        self.size = 400 #hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        self.encoder = ptu.build_feat_encoder(self.ob_dim[-1])
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.feat_size = 1152
        self.f = ptu.build_mlp(1152 + self.ac_dim, self.output_size, self.n_layers, self.size, init_method=init_method_1)
        self.f_hat = ptu.build_mlp(1152 + self.ac_dim, self.output_size, self.n_layers, self.size, init_method=init_method_2)
        
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.encoder.to(ptu.device)
        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def encode(self, obs: torch.Tensor):
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor, acs: torch.Tensor):
        # Encode input (encoded_obs concat with acs)
        enc_obs = self.encode(obs)
        enc_obs_and_acs = torch.cat((enc_obs, acs), dim=1)

        # Get pred error
        targets = self.f(enc_obs_and_acs).detach()
        predictions = self.f_hat(enc_obs_and_acs)
        return torch.norm(predictions - targets, dim=1)

    def forward_np(self, ob_no, ac_na):
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        error = self(ob_no, ac_na)
        return ptu.to_numpy(error)

    def update(self, ob_no: np.ndarray, acs: torch.Tensor): #acs is tensor bc of gumbel softmax stuff
        prediction_errors = self(ptu.from_numpy(ob_no), acs)
        loss = torch.mean(prediction_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
