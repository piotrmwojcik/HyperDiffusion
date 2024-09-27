from implicit_kan.KANLayer import FastKANLayer, KANLinear
from implicit_kan.utils import get_grid, set_random_seed
from implicit_kan.modules import GaussianFourierFeatureTransform
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torch
import time
import numpy as np

h = 128
w = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class ImplicitMLP(nn.Module):
    def __init__(self):
        super(ImplicitMLP, self).__init__()
        self.gff = GaussianFourierFeatureTransform(mapping_dim=128)
        self.linear1 = nn.Linear(2 * 128, 256)
        self.ln1 = nn.LayerNorm(256)  # To avoid gradient vanishing caused by tanh
        self.linear2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.linear3 = nn.Linear(128, 32)
        self.ln3 = nn.LayerNorm(32)
        self.linear4 = nn.Linear(32, 16)
        self.ln4 = nn.LayerNorm(16)
        self.linear5 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.gff(x)
        x = rearrange(x, "b c h w -> (b h w) c")  # Flatten the images
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.ln1(x)
        x = self.linear2(x)
        x = F.tanh(x)
        x = self.ln2(x)
        x = self.linear3(x)
        x = F.tanh(x)
        x = self.ln3(x)
        x = self.linear4(x)
        x = F.tanh(x)
        x = self.ln4(x)
        x = self.linear5(x)
        return x.reshape(1, 3, h, w)

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