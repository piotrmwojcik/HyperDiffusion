import sys, os
import torch
import numpy as np

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from mlp_models import ImplicitMLP
import matplotlib.pyplot as plt

def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x

def get_grid(h, w, b=0, norm=True, device="cpu"):
    if norm:
        xgrid = np.linspace(0, w, num=w) / w
        ygrid = np.linspace(0, h, num=h) / h
    else:
        xgrid = np.linspace(0, w, num=w)
        ygrid = np.linspace(0, h, num=h)
    xv, yv = np.meshgrid(xgrid, ygrid, indexing="xy")
    grid = np.stack([xv, yv], axis=-1)[None]

    grid = torch.from_numpy(grid).float().to(device)
    if b > 0:
        grid = grid.expand(b, -1, -1, -1)  # [Batch, H, W, UV]
        return grid.permute(0, 3, 1, 2)  # [Batch, UV, H, W]
    else:
        return grid[0].permute(2, 0, 1)  # [UV, H, W]

def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)

def generate_mlp_from_weights(weights, B_path = None):
    if not os.path.exists(B_path):
        B = torch.randn((2, 128)) * 10
        save_path = 'data/inrs/B.pth'
        torch.save(B, save_path)

    mlp = ImplicitMLP(B_path = B_path)

    state_dict = mlp.state_dict()
    weight_names = list(state_dict.keys())
    for layer in weight_names:
        val = state_dict[layer]
        num_params = np.product(list(val.shape))
        w = weights[:num_params]
        w = w.view(*val.shape)
        state_dict[layer] = w
        weights = weights[num_params:]
    assert len(weights) == 0, f"len(weights) = {len(weights)}"
    mlp.load_state_dict(state_dict)
    return mlp

def generate_input(grid_size_x, grid_size_y):
    grid = get_grid(grid_size_x, grid_size_y, 0).unsqueeze(0)
    idx_tensor = torch.tensor([0])

    input = {
        'idx': idx_tensor,
        'coords': grid
    }

    return input


if __name__ == '__main__':
    image_resolution = (64,64)

    path_to_inr = 'data/inrs/code_2179.pth'
    weights = torch.load(path_to_inr )
    model = generate_mlp_from_weights(weights['param']['code_'], B_path='data/inrs/B.pth')

    input = generate_input(64,64)

    output = model(input)
    pred_img = lin2img(output['model_out'], image_resolution)
    pred_img = rescale_img((pred_img + 1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()

    plt.imshow(pred_img)
    plt.show()




