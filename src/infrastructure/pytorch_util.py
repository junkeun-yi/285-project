from typing import Union
import numpy as np
import torch
from torch import nn
from src.infrastructure.dqn_utils import Flatten, PreprocessAtari
Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
        init_method=None,
        # flatten_input=False
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
            flatten_input: if the input_size is a tuple, multiply the values together to get the input_size
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []

    in_size = input_size

    # if flatten_input and type(input_size) == tuple:
    #     in_size = 1
    #     for v in input_size:
    #         in_size *= v
    # elif not flatten_input and type(input_size) == tuple:
    #     print(
    #         "WARNING: received tuple input_size to build_mlp, without argument flatten_input=True. "\
    #         "Try calling build_mlp with flatten_input=True.")
    # else:
    #     in_size = input_size

    # if flatten_input:
    #     layers.append(nn.Flatten(1, -1))  # TODO [flatten_input] is this correct?

    for _ in range(n_layers):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size

    last_layer = nn.Linear(in_size, output_size)
    if init_method is not None:
        last_layer.apply(init_method)

    layers.append(last_layer)
    layers.append(output_activation)
        
    return nn.Sequential(*layers)


def build_policy_CNN(input_channels: int,
                    output_size: int,
                    init_method=None,):
    # Same structure as dqn conv (0.27 : 1, ours #  params : stable baselines cnn)

    cnn = nn.Sequential(
        PreprocessAtari(),
        nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(7*7*32, 256),
        nn.ReLU(),
        nn.Linear(256, output_size),
    )
    if init_method:
        cnn.apply(init_method)

    return cnn

def build_feat_encoder(input_channels: int,
                    init_method=None,):
    # 4 layers (32 channel, 3x3 kernel, stride 2, padding 1) from ICM paper. Returns flattened feat 
    cnn = nn.Sequential(
        PreprocessAtari(),
        nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        Flatten()
    )
    if init_method:
        cnn.apply(init_method)

    return cnn

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs).to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def from_numpy_obs_ac_next(ob_no, ac_na, next_ob_no):
        if isinstance(ob_no, np.ndarray):
            ob_no = from_numpy(ob_no)
        if isinstance(next_ob_no, np.ndarray):
            next_ob_no = from_numpy(next_ob_no)
        if isinstance(ac_na, np.ndarray):
            ac_na = from_numpy(ac_na)
        return ob_no, ac_na, next_ob_no

