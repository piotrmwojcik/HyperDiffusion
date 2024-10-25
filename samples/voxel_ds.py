import os
import random

import h5py
import glob
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


class ShapeNetVoxel(Dataset):
    def __init__(self, split='train', sampling=None, dataset_root='datasets', simple_output=False, random_scale=False):
        self.dataset_root = dataset_root
        self.sampling = sampling
        self.init_model_bool = False
        self.split = split
        self.simple_output = simple_output
        self.random_scale = random_scale
        self.init_model()
        self.data_type = 'voxel'

    def __len__(self):
        if self.split == "train":
            return 35019
        else:
            return 8762

    def init_model(self):
        split = self.split
        points_path = os.path.join(self.dataset_root, 'shapenet', 'all_vox256_img', 'data_points_int_' + split + '.pth')
        values_path = os.path.join(self.dataset_root, 'shapenet', 'all_vox256_img', 'data_values_' + split + '.pth')

        self.data_points_int = torch.load(points_path).byte()
        self.data_values = torch.load(values_path).byte()

    def __getitem__(self, idx):
        #points = (self.data_points_int[idx].float() + 1) / 128 - 1
        points = self.data_points_int[idx].float()
        occs = self.data_values[idx].float() * 2 -1

        if self.sampling is not None:
            idcs = np.random.randint(0, len(points), size=self.sampling)
            points = points[idcs]
            occs = occs[idcs]

        if self.random_scale:
            points = random.uniform(0.75, 1.25) * points

        if self.simple_output:
            return occs

        else:
            in_dict = {'idx': idx, 'coords': points}
            gt_dict = {'img': occs}

            return in_dict, gt_dict

root_path = '/Users/piotrwojcik/Downloads/'
shapenet = ShapeNetVoxel(dataset_root=root_path)

for sample_idx, sample in enumerate(shapenet):
    in_dict, gt_dict = sample
    img = gt_dict['img']
    #print(img)
    print(in_dict['coords'].shape)
    #print(in_dict['coords'].view(128, 128, 3))
    #print(in_dict['coords'].view(64, 64, 3))
    #tensor_reshaped = img.view(64, 64, 64)

    # Create a figure for 3D plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Generate the voxel plot
    ax.voxels(tensor_reshaped, facecolors='blue', edgecolor='k')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.show()

    if sample_idx >= 20:
        break

