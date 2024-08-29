from mlp_models import SingleBVPNet
from hd_utils import generate_mlp_from_weights
import torch
import yaml
from hd_utils import Config
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


if __name__ == '__main__':
    weights_path = '/Users/kacpermarzol/PycharmProjects/hyperdiffusionproject/HyperDiffusion/mlp_weights/srn_cars_inr_wgth100/11613_e5b03e7b088d43c09367a6ecbb0d496e_000043.pth'
    config_path = '/Users/kacpermarzol/PycharmProjects/hyperdiffusionproject/HyperDiffusion/configs/' \
                  'diffusion_configs/train_car_2d_img.yaml'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    cfg = DictConfig(config)
    Config.config = cfg
    mlp_kwargs = Config.config["mlp_config"]["params"]

    weights_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    weights = []
    for weight in weights_dict:
        weights.append(weights_dict[weight].flatten().cpu())
    weights = torch.hstack(weights)
    print(weights.shape)
    siren = generate_mlp_from_weights(weights, mlp_kwargs)


    model_input = get_mgrid(128, 2).unsqueeze(0)
    model_input = {'coords': model_input}
    result = siren(model_input)
    img = result['model_out']
    fig, axes = plt.subplots(1, 1, figsize=(9, 9))
    axes.imshow(img.cpu().view(128, 128, 3).detach().numpy())
    plt.savefig("testtt")