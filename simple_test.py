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
    weights_path = '/Users/kacpermarzol/PycharmProjects/hyperdiffusionproject/HyperDiffusion/data/02691156/11441_383ddba4ee9845b6b34dcbe295284658_000031/checkpoints/model_final.pth'
    config_path = '/Users/kacpermarzol/PycharmProjects/hyperdiffusionproject/HyperDiffusion/configs/' \
                  'diffusion_configs/train_car.yaml'

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

    siren = generate_mlp_from_weights(weights, mlp_kwargs)

    state_dict = siren.state_dict()
    layers = []
    layer_names = []
    for l in state_dict:
        shape = state_dict[l].shape
        layers.append(np.prod(shape))
        layer_names.append(l)

    print('layers: ', layers, '\nlayer_names: ', layer_names)


    # model_input = get_mgrid(128, 2).unsqueeze(0)
    # model_input = {'coords': model_input}
    # result = siren(model_input)
    # img = result['model_out']
    # fig, axes = plt.subplots(1, 1, figsize=(9, 9))
    # axes.imshow(img.cpu().view(128, 128, 3).detach().numpy())
    # plt.savefig("testtt")