import os

from hd_utils import generate_mlp_from_weights
import torch
import yaml
from hd_utils import Config, render_meshes
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
from siren.experiment_scripts.test_sdf import SDFDecoder
from siren.sdf_meshing import create_mesh
import copy
import trimesh


def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


if __name__ == '__main__':
    # weights_path = '/Users/kacpermarzol/PycharmProjects/hyperdiffusionproject/HyperDiffusion/mlp_weights/' \
    #                '3d_128_plane_multires_4_manifoldplus_slower_no_clipgrad/occ_1a9b552befd6306cc8f2d5fe7449af61_jitter_0_model_final.pth'
    config_path = 'configs/diffusion_configs/train_plane.yaml'

    checkpoint = torch.load('pca_data_full.pth')
    pca_result = checkpoint['pca_result']
    all_weights = checkpoint['all_weights']
    pca_mean = checkpoint['pca_mean']
    basis = checkpoint['basis']

    mean = torch.mean(pca_result, dim=0)
    std = torch.std(pca_result, dim=0)
    print(mean.shape)
    print(std.shape)
    coefficients = torch.normal(mean, 0.2*std)  # Shape: (400,)

    #coefficients = pca_result[18]
    #print(coefficients.shape)
    weights = torch.zeros(36737)
    #weights = basis[2]
    for i in range(3000):
        weights = weights + basis[i] * coefficients[i]
    #print(weights.shape)
    weights = weights + pca_mean
    print(weights)
    print(all_weights[8])

    #weights = checkpoint['all_weights'][102]

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    cfg = DictConfig(config)
    Config.config = cfg
    mlp_kwargs = Config.config["mlp_config"]["params"]

    # weights_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    # weights = []
    # for weight in weights_dict:
    #     weights.append(weights_dict[weight].flatten().cpu())
    # weights = torch.hstack(weights)
    # print(weights.shape)

    model = generate_mlp_from_weights(weights, mlp_kwargs)
    model_input = get_mgrid(128, 3).unsqueeze(0)

    model_input = {'coords': model_input}

    result = model(model_input)
    x_0 = result['model_out']
    x_0 = x_0.view(len(x_0), -1)

    sdf_decoder = SDFDecoder(
        mlp_kwargs.model_type,
        None,
        "nerf" if mlp_kwargs.model_type == "nerf" else "mlp",
        mlp_kwargs,
    )
    sdf_decoder.model = model.eval()

    os.makedirs("meshes", exist_ok=True)
    folder_name = "meshes"
    res = 128
    sdfs = []
    meshes = []
    level = 0 # ????

    with torch.no_grad():
        effective_file_name = (
            f"{folder_name}/plane_test"
            if folder_name is not None
            else None
        )
        if mlp_kwargs.move:
            for i in range(16):
                v, f, sdf = create_mesh(
                    sdf_decoder,
                    effective_file_name,
                    N=t,
                    level=0
                    if mlp_kwargs.output_type in ["occ", "logits"]
                    else 0,
                    time_val=i,
                )  # 0.9
                if (
                        "occ" in mlp_kwargs.output_type
                        or "logits" in mlp_kwargs.output_type
                ):
                    tmp = copy.deepcopy(f[:, 1])
                    f[:, 1] = f[:, 2]
                    f[:, 2] = tmp
                sdfs.append(sdf)
                mesh = trimesh.Trimesh(v, f)
                meshes.append(mesh)
        else:
            v, f, sdf = create_mesh(
                sdf_decoder,
                effective_file_name,
                N=res,
                level=level
                if mlp_kwargs.output_type in ["occ", "logits"]
                else 0,
            )
            if (
                    "occ" in mlp_kwargs.output_type
                    or "logits" in mlp_kwargs.output_type
            ):
                tmp = copy.deepcopy(f[:, 1])
                f[:, 1] = f[:, 2]
                f[:, 2] = tmp
            sdfs.append(sdf)

            mesh = trimesh.Trimesh(v, f)
            meshes.append(mesh)
        sdfs = torch.stack(sdfs)

    for mesh in meshes:
        mesh.vertices *= 2
    print(
        "sdfs.stats",
        sdfs.min().item(),
        sdfs.max().item(),
        sdfs.mean().item(),
        sdfs.std().item(),
    )
    out_imgs = render_meshes(meshes)
    #print(out_imgs.shape)
    plt.imsave('output_mesh.png', out_imgs[0])