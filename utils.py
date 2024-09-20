from torchvision import transforms
from PIL import Image

import torch
import matplotlib.pyplot as plt


# Function to prepare a single image
def prepare_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = transform(image)

    h, w = image.shape[1], image.shape[2]
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    x_grid, y_grid = torch.meshgrid(x, y)
    coords = torch.stack([x_grid, y_grid], dim=-1).view(-1, 2)

    pixels = image.view(3, -1).t()

    return coords, pixels


# Visualization function
def visualize_results(model, image_size, device, save_path=None):
    with torch.no_grad():
        x = torch.linspace(-1, 1, image_size)
        y = torch.linspace(-1, 1, image_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        coords = torch.stack([x_grid, y_grid], dim=-1).view(-1, 2).to(device)

        pixels = model(coords).cpu().view(image_size, image_size, 3).numpy()

        dpi = 100
        fig = plt.figure(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.imshow(pixels)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            print(f"Saved visualization to {save_path}")

        plt.close(fig)
