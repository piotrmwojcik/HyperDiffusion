import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from PIL import Image
from torchvision.transforms import ToTensor, Normalize
from time import time
import cv2
from tqdm.auto import tqdm

from implicit_kan.ChebyKANLayer import ChebyKANLayer
from implicit_kan.KANLayer import FastKANLayer, KANLinear
from implicit_kan.utils import get_grid, set_random_seed
from implicit_kan.modules import GaussianFourierFeatureTransform

h = 256
w = 256

num_steps = 3000

set_random_seed(1)

device = 'cuda'

img = ToTensor()(Image.open('./inputs/eye.jpeg').resize((h, w)))[None].to(device) * 2 - 1.

grid = get_grid(img.shape[2], img.shape[3], b=1).to(device)


class ImplicitKAN(nn.Module):
    def __init__(self, pos_enc='gff'):
        super(ImplicitKAN, self).__init__()
        self.pos_enc = pos_enc
        assert pos_enc in ['gff', 'kan']
        if pos_enc == 'gff':
            self.pe = GaussianFourierFeatureTransform(mapping_dim=128)
        elif pos_enc == 'kan':
            self.pe =  FastKANLayer(2, 128)
        self.kan1 = FastKANLayer(256, 32)
        self.kan2 = FastKANLayer(32, 16)
        self.kan3 = FastKANLayer(16, 3)

    def forward(self, x):
        if self.pos_enc == 'kan':
            x = rearrange(x, "b c h w -> (b h w) c")  # Flatten the images
            x = self.pe(x)
        elif self.pos_enc == 'gff':
            x = self.pe(x)
            x = rearrange(x, "b c h w -> (b h w) c")  # Flatten the images
        x = self.kan1(x)
        x = self.kan2(x)
        x = self.kan3(x)
        return x.reshape(1, 3, h, w)


class ImplicitEKAN(nn.Module):
    def __init__(self, pos_enc='gff', grid_size=5):
        super(ImplicitEKAN, self).__init__()
        self.pos_enc = pos_enc
        assert pos_enc in ['gff', 'kan']
        if pos_enc == 'gff':
            self.pe = GaussianFourierFeatureTransform(mapping_dim=128)
        elif pos_enc == 'kan':
            self.pe = KANLinear(2, 128, grid_size=grid_size)
        self.kan1 = KANLinear(256, 32, grid_size=grid_size)
        self.kan2 = KANLinear(32, 16, grid_size=grid_size)
        self.kan3 = KANLinear(16, 3, grid_size=grid_size)

    def forward(self, x):
        if self.pos_enc == 'kan':
            x = rearrange(x, "b c h w -> (b h w) c")  # Flatten the images
            x = self.pe(x)
        elif self.pos_enc == 'gff':
            x = self.pe(x)
            x = rearrange(x, "b c h w -> (b h w) c")  # Flatten the images
        x = self.kan1(x)
        x = self.kan2(x)
        x = self.kan3(x)
        return x.reshape(1, 3, h, w)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in [self.kan1, self.kan2, self.kan3]
        )


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

kan_model = ImplicitEKAN(pos_enc='gff').to(device)
total_params = sum(p.numel() for p in kan_model.parameters() if p.requires_grad)
print(f"Total trainable parameters in efficient kan with gff: {total_params}")

kan_model_6 = ImplicitEKAN(pos_enc='gff', grid_size=6).to(device)
total_params = sum(p.numel() for p in kan_model_6.parameters() if p.requires_grad)
print(f"Total trainable parameters in efficient kan with gff grid 6: {total_params}")

kan_model_7 = ImplicitEKAN(pos_enc='gff', grid_size=7).to(device)
total_params = sum(p.numel() for p in kan_model_7.parameters() if p.requires_grad)
print(f"Total trainable parameters in efficient kan with gff grid 7: {total_params}")

kan_model_7_reg = ImplicitEKAN(pos_enc='gff', grid_size=7).to(device)
total_params = sum(p.numel() for p in kan_model_7_reg.parameters() if p.requires_grad)
print(f"Total trainable parameters in efficient kan with gff grid 7 with reg: {total_params}")

mlp = ImplicitMLP().to(device)
total_params_mlp = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
print(f"Total trainable parameters in MLP with gff: {total_params_mlp}")

optim_kan = optim.AdamW(kan_model.parameters(), lr=1e-3, weight_decay=1e-4)
optim_kan_6 = optim.AdamW(kan_model_6.parameters(), lr=1e-3, weight_decay=1e-4)
optim_kan_7 = optim.AdamW(kan_model_7.parameters(), lr=1e-3, weight_decay=1e-4)
optim_kan_7_reg = optim.AdamW(kan_model_7_reg.parameters(), lr=1e-3, weight_decay=1e-4)
optim_mlp = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)

kan_loss = []
ekan_gff_imgs = []
start = time()
for step_idx in tqdm(range(num_steps)):
    optim_kan.zero_grad()
    out = kan_model(grid)
    loss = ((F.tanh(out) - img) ** 2).mean()
    loss.backward()
    optim_kan.step()
    kan_loss.append(((F.tanh(out) - img) ** 2).mean())
    ekan_gff_imgs.append(out)
kan_loss = [l.item() for l in kan_loss]
ekan_gff_imgs = [im.cpu().data.numpy() for im in ekan_gff_imgs]
print(f"trained in {time() - start} to {kan_loss[-1]}")

kan_loss_reg = []
ekan_gff_imgs_reg = []
start = time()
for step_idx in tqdm(range(num_steps)):
    optim_kan_6.zero_grad()
    out = kan_model_6(grid)
    loss = ((F.tanh(out) - img) ** 2).mean()
    loss.backward()
    optim_kan_6.step()
    kan_loss_reg.append(((F.tanh(out) - img) ** 2).mean())
    ekan_gff_imgs_reg.append(out)
kan_loss_reg = [l.item() for l in kan_loss_reg]
ekan_gff_imgs_reg = [im.cpu().data.numpy() for im in ekan_gff_imgs_reg]
print(f"trained in {time() - start} to {kan_loss_reg[-1]}")

kan_gff_imgs = []
kan_gff_loss = []
start = time()
for step_idx in tqdm(range(num_steps)):
    optim_kan_7.zero_grad()
    out = kan_model_7(grid)
    loss = ((F.tanh(out) - img) ** 2).mean()
    loss.backward()
    optim_kan_7.step()
    kan_gff_loss.append(loss)
    kan_gff_imgs.append(out)

kan_gff_loss = [l.item() for l in kan_gff_loss]
kan_gff_imgs = [im.cpu().data.numpy() for im in kan_gff_imgs]

print(f"trained in {time() - start} to {loss.item()}")

kan_7_reg_imgs = []
kan_7_reg_loss = []
start = time()
for step_idx in tqdm(range(num_steps)):
    optim_kan_7_reg.zero_grad()
    out = kan_model_7_reg(grid)
    loss = ((F.tanh(out) - img) ** 2).mean() + kan_model_7_reg.regularization_loss()
    loss.backward()
    optim_kan_7_reg.step()
    kan_7_reg_loss.append(((F.tanh(out) - img) ** 2).mean())
    kan_7_reg_imgs.append(out)

kan_7_reg_loss = [l.item() for l in kan_7_reg_loss]
kan_7_reg_imgs = [im.cpu().data.numpy() for im in kan_7_reg_imgs]

print(f"trained in {time() - start} to {loss.item()}")


mlp_imgs = []
mlp_loss = []
start = time()
for step_idx in tqdm(range(num_steps)):
    mlp.zero_grad()
    out = mlp(grid)
    loss = ((F.tanh(out) - img) ** 2).mean()
    loss.backward()
    optim_mlp.step()
    mlp_loss.append(loss)
    mlp_imgs.append(out)

mlp_loss = [l.item() for l in mlp_loss]
mlp_imgs = [im.cpu().data.numpy() for im in mlp_imgs]

print(f"trained in {time() - start} to {loss.item()}")

