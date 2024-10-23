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
    def __init__(self, split='train', sampling=None, dataset_root='datasets'):
        """
        Initialize the ShapeNetVoxel dataset.

        Args:
            split (str): Specify the dataset split ('train' or 'test').
            sampling (int or None): Number of points to sample per voxel grid.
            dataset_root (str): Path to the dataset directory.
        """
        self.dataset_root = dataset_root
        self.data_path = os.path.join(dataset_root, 'shapenet', 'all_vox256_img', 'all_vox256_img_' + split + '.pth')
        self.data_voxels = torch.load(self.data_path).byte()  # Load the voxel data
        self.sampling = sampling  # Number of points to sample
        self.split = split  # Dataset split (train/test)
        self.grid = self.get_mgrid_voxel(64)  # Create a grid of voxel coordinates
        self.affine = (None, None)  # Can be used for future transformations

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        if self.split == "train":
            return 35019
        else:
            return 8762

    def dec2bin(self, x, bits):
        """
        Convert decimal to binary.

        Args:
            x (Tensor): Input tensor.
            bits (int): Number of bits for conversion.

        Returns:
            Tensor: Binary representation of the input tensor.
        """
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().flip(-1)

    def get_mgrid_voxel(self, dim):
        """
        Generate a voxel grid of shape (dim x dim x dim).

        Args:
            dim (int): Dimension of the grid.

        Returns:
            Tensor: Generated grid coordinates.
        """
        ranges = [torch.linspace(-1, 1, steps=dim) for _ in range(3)]
        grid = torch.stack(torch.meshgrid(*ranges), dim=-1).reshape(-1, 3)  # Flatten the grid into (N, 3)
        return grid

    def __getitem__(self, idx):
        """
        Returns the input and target data for a single voxel grid.

        Args:
            idx (int): Index of the voxel grid.

        Returns:
            dict: A dictionary containing the input coordinates and the ground truth voxel values.
        """
        points = self.grid.float()  # Coordinates in the grid

        encoded_voxels = self.data_voxels[idx]  # Get the voxel grid at index `idx`
        occs = self.dec2bin(encoded_voxels, 8).view(-1, 1).float()  # Convert voxel values to binary

        if self.sampling is not None:
            # Randomly sample points and corresponding occupancy values
            idcs = np.random.randint(0, len(points), size=self.sampling)
            points = points[idcs]
            occs = occs[idcs]

        # Return a dictionary of input coordinates and ground truth occupancy values
        in_dict = {'idx': idx, 'coords': points}
        gt_dict = {'img': occs}

        return in_dict, gt_dict

root_path = '/Users/piotrwojcik/Downloads/'
shapenet = ShapeNetVoxel(dataset_root=root_path)

for sample_idx, sample in enumerate(shapenet):
    in_dict, gt_dict = sample
    img = gt_dict['img']
    tensor_reshaped = img.view(64, 64, 64)

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

