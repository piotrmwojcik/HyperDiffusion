import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch import nn
import math
import torch.nn.functional as F

from embedder import Embedder

from torchmeta.modules import (MetaModule, MetaSequential)
from math import pi
from einops import rearrange
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



class FMMLinear(nn.Module):
    """
    FMM layer via BMM instead of F.conv
    """
    def __init__(self, in_channel: int, out_channel: int, factorization_rank: int):

        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.rank = factorization_rank

        self.left_matrix = nn.Parameter(torch.randn(out_channel, factorization_rank))
        self.right_matrix = nn.Parameter(torch.randn(factorization_rank, in_channel))
        self.bias = nn.Parameter(torch.zeros(out_channel).fill_(0))

        self.reset_parameters()

    def reset_parameters(self):
        # Standard initialization (usually Xavier or Kaiming)
        nn.init.kaiming_uniform_(self.left_matrix, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.right_matrix, a=math.sqrt(5))

        W = self.left_matrix @ self.right_matrix

        if self.bias is not None:
            # Initialize bias uniformly
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(W)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        W = self.left_matrix @ self.right_matrix # [batch_size, out_channel, in_channel]
        out = F.linear(input, W, self.bias)

        return out

class GaussianFourierFeatureTransform(nn.Module):
    """
    From https://github.com/ndahlquist/pytorch-fourier-feature-networks/blob/master/fourier_feature_transform.py
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_dim*2, width, height].
    """

    def __init__(self, B, num_input_channels=2, mapping_dim=256):
        super().__init__()

        self._num_input_channels = num_input_channels
        self.mapping_dim = mapping_dim
        self._B = B

    def forward(self, x, phase=None):
        batches, channels, width, height = x.shape
        assert channels == self._num_input_channels, "Expected input to have {} channels (got {} channels)".format(
            self._num_input_channels, channels
        )

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self.mapping_dim)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        if phase is not None:
            x = 2 * pi * x + phase
        else:
            x = 2 * pi * x

        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class ImplicitMLP(nn.Module):
    def __init__(self, B_path):
        super(ImplicitMLP, self).__init__()
        loaded_B = torch.load(B_path)
        self.gff = GaussianFourierFeatureTransform(B=loaded_B, mapping_dim=128)
        self.linear1 = FMMLinear(128 * 2, 256, 70)
        self.linear2 = FMMLinear(256, 128, 10)
        self.linear3 = nn.Linear(128, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 3)

    def forward(self, model_input):
        h = 64
        w = 64

        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        x = self.gff(coords)
        x = rearrange(x, "b c h w -> (b h w) c")  # Flatten the images
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        output = self.linear5(x).unsqueeze(0)

        return {'model_in': coords_org, 'model_out': output}


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
        #print(self)

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