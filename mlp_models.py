import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch import nn

from embedder import Embedder

from torchmeta.modules import (MetaModule, MetaSequential)
from helpers import (ImageDownsampling, FCBlock)

from collections import OrderedDict


class MLP(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_neurons,
        use_tanh=True,
        over_param=False,
        use_bias=True,
    ):
        super().__init__()
        multires = 1
        self.over_param = over_param
        if not over_param:
            self.embedder = Embedder(
                include_input=True,
                input_dims=2,
                max_freq_log2=multires - 1,
                num_freqs=multires,
                log_sampling=True,
                periodic_fns=[torch.sin, torch.cos],
            )
        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))
        self.use_tanh = use_tanh

    def forward(self, x):
        if not self.over_param:
            x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x, None


class MLP3D(nn.Module):
    def __init__(
        self,
        out_size,
        hidden_neurons,
        use_leaky_relu=False,
        use_bias=True,
        multires=10,
        output_type=None,
        move=False,
        **kwargs,
    ):
        super().__init__()
        self.embedder = Embedder(
            include_input=True,
            input_dims=3 if not move else 4,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.layers = nn.ModuleList([])
        self.output_type = output_type
        self.use_leaky_relu = use_leaky_relu
        in_size = self.embedder.out_dim
        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        x = coords_org
        x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)
        x = self.layers[-1](x)

        if self.output_type == "occ":
            # x = torch.sigmoid(x)
            pass
        elif self.output_type == "sdf":
            x = torch.tanh(x)
        elif self.output_type == "logits":
            x = x
        else:
            raise f"This self.output_type ({self.output_type}) not implemented"
        x = dist.Bernoulli(logits=x).logits

        return {"model_in": coords_org, "model_out": x}

class SingleBVPNet(MetaModule): ## SIREN 2D
    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        # if self.mode == 'rbf':
        #     self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
        #     in_features = kwargs.get('rbf_centers', 1024)
        # elif self.mode == 'nerf':
        #     self.positional_encoding = PosEncodingNeRF(in_features=in_features,
        #                                                sidelength=kwargs.get('sidelength', None),
        #                                                fn_samples=kwargs.get('fn_samples', None),
        #                                                use_nyquist=kwargs.get('use_nyquist', True))
        #     in_features = self.positional_encoding.out_dim

        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        # various input processing methods for different applications
        if self.image_downsampling.downsample:
            coords = self.image_downsampling(coords)
        if self.mode == 'rbf':
            coords = self.rbf_layer(coords)
        elif self.mode == 'nerf':
            coords = self.positional_encoding(coords)

        output = self.net(coords, self.get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}