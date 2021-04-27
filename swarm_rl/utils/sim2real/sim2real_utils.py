import numpy as np
import torch
import json
import os
import subprocess
import torch.nn as nn
import torch.nn.functional as F

from sample_factory.algorithms.appo.model import create_actor_critic
from swarm_rl.train import register_custom_components

from gym.spaces import Box
from attrdict import AttrDict

from code_blocks import (
	headers_network_evaluate,
	linear_activation,
	sigmoid_activation,
	relu_activation,
)

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


def generate_c_model(model, output_path='network_evaluate.cpp'):

    layer_names, bias_names, weights, biases, outputs = generate_weights(model, output_path, transpose=True)
    num_layers = len(layer_names)

    structure = 'static const int structure [' + str(int(num_layers)) + '][2] = {'
    for name, param in model.named_parameters():
        param = param.T
        if 'weight' in name and 'critic' not in name:
            structure += '{' + str(param.shape[0]) + ', ' + str(param.shape[1]) + '},'

    # complete the structure array
    # get rid of the comma after the last curly bracket
    structure = structure[:-1]
    structure += '};\n'

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
        for (int i = 0; i < structure[0][1]; i++) {{
            output_0[i] = 0;
            for (int j = 0; j < structure[0][0]; j++) {{
                output_0[i] += state_array[j] * {layer_names[0]}[j][i];
            }}
            output_0[i] += {bias_names[0]}[i];
            output_0[i] = tanhf(output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, num_layers - 1):
        for_loop = f'''
        for (int i = 0; i < structure[{str(n)}][1]; i++) {{
            output_{str(n)}[i] = 0;
            for (int j = 0; j < structure[{str(n)}][0]; j++) {{
                output_{str(n)}[i] += output_{str(n - 1)}[j] * {layer_names[n]}[j][i];
            }}
            output_{str(n)}[i] += {bias_names[n]}[i];
            output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
        }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
                for (int i = 0; i < structure[{str(n)}][1]; i++) {{
                    output_{str(n)}[i] = 0;
                    for (int j = 0; j < structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {layer_names[n]}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n]}[i];
                }}
    '''
    for_loops.append(output_for_loop)

    # assign network outputs to control
    assignment = """
            control_n->thrust_0 = output_"""+str(n)+"""[0];
            control_n->thrust_1 = output_"""+str(n)+"""[1];
            control_n->thrust_2 = output_"""+str(n)+"""[2];
            control_n->thrust_3 = output_"""+str(n)+"""[3];	
    """

    # construct the network evaluate function
    controller_eval = """void networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    for code in for_loops:
        controller_eval += code
    # assignment to control_n
    controller_eval += assignment

    # closing bracket
    controller_eval += """}"""

    # combine all the codes
    source = ""
    # headers
    source += headers_network_evaluate
    # helper funcs
    source += linear_activation
    source += sigmoid_activation
    source += relu_activation
    # network eval func
    source += structure
    for output in outputs:
        source += output
    for weight in weights:
        source += weight
    for bias in biases:
        source += bias
    source += controller_eval

    if output_path:
        with open(output_path, 'w') as f:
            f.write(source)
        f.close()

    return source


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
    layer_names, bias_names, outputs = [], [], []
    n_bias = 0
    for name, param in model.named_parameters():
        if transpose:
            param = param.T
        name = name.replace('.', '_')
        if 'weight' in name and 'critic' not in name:
            layer_names.append(name)
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

        if 'bias' in name and 'critic' not in name:
            bias_names.append(name)
            bias = 'static const float ' + name + '[' + str(param.shape[0]) + '] = {'
            for num in param:
                bias += str(num.item()) + ','
            # get rid of comma after last number
            bias = bias[:-1]
            bias += '};\n'
            biases.append(bias)
            output = 'static float output_' + str(n_bias) + '[' + str(param.shape[0]) + '];\n'
            outputs.append(output)
            n_bias += 1


    return layer_names, bias_names, weights, biases, outputs


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
    with open('original_weights.txt', 'r') as f:
        lines = f.readlines()
        l0, l1, l2 = lines
        layer_0_weights = np.fromstring(l0, dtype=float, sep=',')
        layer_1_weights = np.fromstring(l1, dtype=float, sep=',')
        layer_2_weights = np.fromstring(l2, dtype=float, sep=',')

    with open('original_biases.txt', 'r') as f:
        lines = f.readlines()
        b0, b1, b2 = lines
        bias_0_weights = np.fromstring(b0, dtype=float, sep=',')
        bias_1_weights = np.fromstring(b1, dtype=float, sep=',')
        bias_2_weights = np.fromstring(b2, dtype=float, sep=',')

    with torch.no_grad():
        # PyTorch stores weights as (out_features, in_features) for whatever reason
        sim2real_policy.layer_0.weight = torch.nn.Parameter(torch.from_numpy(layer_0_weights).float().view(18, 64).T)
        sim2real_policy.layer_1.weight = torch.nn.Parameter(torch.from_numpy(layer_1_weights).float().view(64, 64).T)
        sim2real_policy.layer_2.weight = torch.nn.Parameter(torch.from_numpy(layer_2_weights).float().view(64, 4).T)

        sim2real_policy.layer_0.bias = torch.nn.Parameter(torch.from_numpy(bias_0_weights).float().view(64,))
        sim2real_policy.layer_1.bias = torch.nn.Parameter(torch.from_numpy(bias_1_weights).float().view(64,))
        sim2real_policy.layer_2.bias = torch.nn.Parameter(torch.from_numpy(bias_2_weights).float().view(4,))
    ####################################################################################################################

    ####################################################################################################################
    # # SAVE SF AND SIM2REAL POLICY WEIGHTS INTO A TEXT FILE (for testing correctness)
    model_path = '../../train_dir/quads_multi_one_drone_v112/one_drone_/03_one_drone_see_3333_non_tanh/checkpoint_p0/checkpoint_000976564_1000001536.pth'
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    cfg_path = '../../train_dir/quads_multi_one_drone_v112/one_drone_/03_one_drone_see_3333_non_tanh/cfg.json'
    cfg_path = os.path.join(os.path.dirname(__file__), cfg_path)
    sf_policy = load_sf_model(model_path, cfg_path)
    print(sf_policy)
    # save the weights of the sf policy for sim2real
    generate_weights(sf_policy, output_path="sf_model_weights.txt", transpose=True)
    # save weights of sim2real policy
    generate_weights(sim2real_policy, output_path="sim2real_model_weights.txt", transpose=True)
    generate_c_model(sf_policy, output_path="SF_network_evaluate_autogen.cpp")
    ####################################################################################################################

    # test outputs of PyTorch model given a random observation that can be compared to the c* version of the model
    # obs = torch.rand(1, 18)
    # obs = torch.FloatTensor([0.165928, 0.942265, 0.683338, 0.476424, 0.175134, 0.393026, 0.37596, 0.173734, 0.937535, 0.15913, 0.967515, 0.620271, 0.406, 0.990523, 0.404797, 0.968619, 0.853021, 0.956329]).view(1, 18)
    # obs = torch.FloatTensor([0, 0, -10, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]).view(1, 18)
    # obs_dict = {'obs': obs}

    # compare outputs
    # print("Sample Factory policy output: ", sf_policy.action_parameterization(sf_policy.actor_encoder(obs_dict))[1].means)
    # print("Sim2real policy output: ", sim2real_policy(obs))

    rel_pos = [
        [0, 0, 1], [0, 0, 2], [0, 0, -1], [0, 0, -2],  # strafe up/down
        [1, 0, 0], [2, 0, 0], [-1, 0, 0], [-2, 0, 0],  # strafe forward/back
        [0, 1, 0], [0, 2, 0], [0, -1, 0], [0, -2, 0],  # strafe left/right
        [0, 0, 0]  # no movement
    ]

    vxyz_R_omega = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]

    for pos in rel_pos:
        # compare outputs of different models (c++ vs python, sample-factory vs sim2real, etc) for different inputs
        obs = torch.FloatTensor(pos + vxyz_R_omega).view(1, 18)
        obs_dict = {'obs': obs}
        SF_out = sf_policy.action_parameterization(sf_policy.actor_encoder(obs_dict))[1].means.detach().numpy()
        sim2real_out = sim2real_policy(obs).detach().numpy()
        # print(f'Relative Pos: {pos}, Sample Factory Output: {SF_out}, Sim2Real Output: {sim2real_out} \n')
        print(f'Relative Pos: {pos}, Sample Factory Output: {SF_out} \n')

    # Compare PyTorch SF model output to C++ SF Model output
    output = subprocess.run(['/usr/bin/g++', 'sf_eval.cpp', './a.out'], stdout=subprocess.PIPE)
    res = subprocess.call('./a.out')
    print(res)