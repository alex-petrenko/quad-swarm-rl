import numpy as np
import torch
import json
import os
import torch.nn as nn
import torch.nn.functional as F

from sample_factory.algorithms.appo.model import create_actor_critic
from swarm_rl.train import register_custom_components

from gym.spaces import Box
from attrdict import AttrDict


class GaussianMLP(nn.Module):
    def __init__(self):
        super(GaussianMLP, self).__init__()

        self.layer_0 = nn.Linear(18, 64, bias=True)
        self.layer_1 = nn.Linear(64, 64, bias=True)
        self.layer_2 = nn.Linear(64, 4, bias=True)

    def forward(self, x):
        out = F.tanh(self.layer_0(x))
        out = F.tanh(self.layer_1(out))
        out = self.layer_2(out)
        return out


class SFPolicy(nn.Module):
    def __init__(self, actor_encoder, action_param_layer):
        '''
        Sample factory policy
        :param actor_encoder: Encoder network
        :param action_param_layer: Final layer that outputs the actions
        '''
        super(SFPolicy, self).__init__()
        self.encoder = actor_encoder
        self.output_layer = action_param_layer

    def forward(self, x):
        return self.output_layer(self.encoder(x))


def generate_weights(model, output_path='model_weights.txt', transpose=False):
    """
    Convert a pytorch model into the list of weights that sim2real can digest (see network_evaluate.c in sim2real)
    Based off of https://github.com/amolchanov86/quad_sim2multireal/blob/8bb11facfacb00ba19fcae7168c793de44491ed4/quad_gen/gaussian_mlp.py
    :param model: pytorch model
    :param output_path: weights file
    :param transpose: transpose each layer if true
    :return:
    """
    weights, biases = [], []
    for name, param in model.named_parameters():
        if transpose:
            param = param.T
        name = name.replace('.', '_')
        if 'weight' in name:
            weight = 'static const float ' + name + '[' + str(param.shape[0]) + '][' + str(param.shape[1]) + '] = {'
            for row in param:
                weight += '{'
                for num in row:
                    weight += str(num.item()) + ','
                # get rid of comma after the last number
                weight = weight[:-1]
                weight += '},'
            # get rid of comma after the last curly bracket
            weight = weight[:-1]
            weight += '};\n'
            weights.append(weight)

        if 'bias' in name:
            bias = 'static const float ' + name + '[' + str(param.shape[0]) + '] = {'
            for num in param:
                bias += str(num.item()) + ','
            # get rid of comma after last number
            bias = bias[:-1]
            bias += '};\n'
            biases.append(bias)

        # combine all the code
        source = ''
        for weight in weights:
            source += weight
        for bias in biases:
            source += bias

        with open(output_path, 'w') as f:
            f.write(source)


def load_sf_model(model_path, cfg_path):
    '''
    Load a sample factory model
    :param model_path: Model path
    :param cfg_path: cfg path
    :return: PyTorch Model
    '''

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    with open(cfg_path, 'r') as cfg_file:
        cfg = AttrDict(json.load(cfg_file))

    obs_space = Box(-100, 100, (156,))
    action_space = Box(-1, 1, (4,))
    register_custom_components()

    policy = create_actor_critic(cfg, obs_space, action_space)
    policy.load_state_dict(checkpoint['model'])

    return policy


if __name__ == '__main__':
    ####################################################################################################################
    # # LOAD SIM2REAL BASELINE POLICY WEIGHTS INTO PYTORCH MODEL
    sim2real_policy = GaussianMLP()
    with open('weights.txt', 'r') as f:
        lines = f.readlines()
        l0, l1, l2 = lines
        layer_0_weights = np.fromstring(l0, dtype=float, sep=',')
        layer_1_weights = np.fromstring(l1, dtype=float, sep=',')
        layer_2_weights = np.fromstring(l2, dtype=float, sep=',')

    with open('biases.txt', 'r') as f:
        lines = f.readlines()
        b0, b1, b2 = lines
        bias_0_weights = np.fromstring(b0, dtype=float, sep=',')
        bias_1_weights = np.fromstring(b1, dtype=float, sep=',')
        bias_2_weights = np.fromstring(b2, dtype=float, sep=',')

    with torch.no_grad():
        # PyTorch stores weights as (out_features, in_features) for whatever reason
        sim2real_policy.layer_0.weight = torch.nn.Parameter(torch.from_numpy(layer_0_weights).float().view(64, 18))
        sim2real_policy.layer_1.weight = torch.nn.Parameter(torch.from_numpy(layer_1_weights).float().view(64, 64))
        sim2real_policy.layer_2.weight = torch.nn.Parameter(torch.from_numpy(layer_2_weights).float().view(4, 64))

        sim2real_policy.layer_0.bias = torch.nn.Parameter(torch.from_numpy(bias_0_weights).float().view(64,))
        sim2real_policy.layer_1.bias = torch.nn.Parameter(torch.from_numpy(bias_1_weights).float().view(64,))
        sim2real_policy.layer_2.bias = torch.nn.Parameter(torch.from_numpy(bias_2_weights).float().view(4,))
    ####################################################################################################################

    ####################################################################################################################
    # # SAVE SF POLICY WEIGHTS INTO A TEXT FILE
    model_path = '../../train_dir/quads_multi_one_drone_v112/one_drone_/03_one_drone_see_3333_non_tanh/checkpoint_p0/checkpoint_000976564_1000001536.pth'
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    cfg_path = '../../train_dir/quads_multi_one_drone_v112/one_drone_/03_one_drone_see_3333_non_tanh/cfg.json'
    cfg_path = os.path.join(os.path.dirname(__file__), cfg_path)
    sf_policy = load_sf_model(model_path, cfg_path)
    # save the weights of the sf policy for sim2real
    generate_weights(sf_policy, transpose=True)
    ####################################################################################################################

    # test outputs of PyTorch model given a random observation that can be compared to the c* version of the model
    obs = torch.rand(1, 18)
    obs_dict = {'obs': obs}

    # compare outputs
    print("Sample Factory policy output: ", sf_policy.action_parameterization(sf_policy.actor_encoder(obs_dict))[1].means)
    print("Sim2real policy output: ", sim2real_policy(obs))

