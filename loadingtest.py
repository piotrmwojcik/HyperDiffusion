# from mlp_models import FMMLinear
from implicit_kan.implicit_kan.modules import GaussianFourierFeatureTransform
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torch
import time
import numpy as np
import math

h = 128
w = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
class ImplicitMLP(nn.Module):
    def __init__(self):
        super(ImplicitMLP, self).__init__()
        # loaded_B = torch.load(B_path)
        self.gff = GaussianFourierFeatureTransform(mapping_dim=128)
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


if __name__ == '__main__':
    d = {}
    start = time.time()
    for i in range(200_000):
        if i % 10000 == 0:
            end = time.time()
            print(i, np.round(end - start,2))
            start = time.time()
        d[i] = ImplicitMLP().to(device)
    print("ok")