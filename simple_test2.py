from mlp_models import SingleBVPNet
from hd_utils import generate_mlp_from_weights, render_meshes
import torch
import yaml
from hd_utils import Config, get_mlp

from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
from dataset import WeightDataset
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import copy
import trimesh

from siren.experiment_scripts.test_sdf import SDFDecoder
from siren.sdf_meshing import create_mesh
from transformer import Transformer
from torch.utils.data import DataLoader, random_split


def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = ':1'
    config_path = 'configs/diffusion_configs/train_plane.yaml'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    cfg = DictConfig(config)
    Config.config = cfg

    method = Config.get("method")
    mlp_kwargs = None

    if "hyper" in method:
        mlp_kwargs = Config.config["mlp_config"]["params"]

    wandb.init(
        project="hyperdiffusion",
        dir=config["tensorboard_log_dir"],
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        tags=[Config.get("mode")],
        mode="disabled" if Config.get("disable_wandb") else "online",
        config=dict(config),
    )

    wandb_logger = WandbLogger()
    wandb_logger.log_text("config", ["config"], [[str(config)]])
    print("wandb", wandb.run.name, wandb.run.id)

    mlps_folder_train = Config.get("mlps_folder_train")

    print('!!!!')

    if "hyper" in method:
        mlp = get_mlp(mlp_kwargs)
        state_dict = mlp.state_dict()
        layers = []
        layer_names = []
        for l in state_dict:
            shape = state_dict[l].shape
            layers.append(np.prod(shape))
            layer_names.append(l)
        model = Transformer(
            layers, layer_names, **Config.config["transformer_config"]["params"]
        )
    #
    dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])

    # )
    #
    # if not cfg.mlp_config.params.move:
    #     train_object_names = set([str.split(".")[0] for str in train_object_names])
    # Check if dataset folder already has train,test,val split; create otherwise.
    if method == "hyper_3d":
        mlps_folder_all = mlps_folder_train
        # all_object_names = np.array(
        #     [obj for obj in os.listdir(mlps_folder_all) if ".lst" not in obj]
        # )
        # total_size = len(all_object_names)
        # val_size = int(total_size * 0.05)
        # test_size = int(total_size * 0.15)
        # train_size = total_size - val_size - test_size
        # if not os.path.exists(os.path.join(dataset_path, "train_split.lst")):
        #     train_idx = np.random.choice(
        #         total_size, train_size + val_size, replace=False
        #     )
        #     test_idx = set(range(total_size)).difference(train_idx)
        #     val_idx = set(np.random.choice(train_idx, val_size, replace=False))
        #     train_idx = set(train_idx).difference(val_idx)
        #     print(
        #         "Generating new partition",
        #         len(train_idx),
        #         train_size,
        #         len(val_idx),
        #         val_size,
        #         len(test_idx),
        #         test_size,
        #     )
        #
        #     np.savetxt(
        #         os.path.join(dataset_path, "train_split.lst"),
        #         all_object_names[list(train_idx)],
        #         delimiter=" ",
        #         fmt="%s",
        #     )

        #train_object_names = np.genfromtxt(
        #    os.path.join(dataset_path, "train_split.lst"), dtype="str")

        train_dt = WeightDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            # train_object_names
        )
        train_dl = DataLoader(
            train_dt,
            # batch_size=Config.get("batch_size"),
            batch_size=1,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        all_weights = []
        for i, weights in enumerate(train_dl):
            weights = weights[0].view(-1).unsqueeze(0)
            all_weights.append(weights)
        all_weights = torch.cat(all_weights, dim=0)
        print(all_weights.shape)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1000)  # You can choose the number of components
        pca_result = torch.tensor(pca.fit_transform(all_weights))
        basis = torch.tensor(pca.components_)
        print(pca_result.shape)

        mean = torch.mean(pca_result, dim=0)  # Shape: (400,)
        std = torch.std(pca_result, dim=0)  # Shape: (400,)

        mean = torch.mean(pca_result, dim=0)
        std = torch.std(pca_result, dim=0)
        coefficients = torch.normal(mean, std)  # Shape: (400,)

        coefficients = pca_result[1]
        weights = torch.zeros(36737)
        for i in range(coefficients.shape[0]):
            weights = weights + basis[i] * coefficients[i]
        print(weights.shape)

        model = generate_mlp_from_weights(weights, mlp_kwargs).cuda()
        model_input = get_mgrid(128, 3).unsqueeze(0).cuda()

        model_input = {'coords': model_input}

        result = model(model_input)
        x_0 = result['model_out']
        x_0 = x_0.view(len(x_0), -1)

        sdf_decoder = SDFDecoder(
            mlp_kwargs.model_type,
            None,
            "nerf" if mlp_kwargs.model_type == "nerf" else "mlp",
            mlp_kwargs,
        ).cuda()
        sdf_decoder.model = model.eval()

        os.makedirs("meshes", exist_ok=True)
        folder_name = "meshes"
        res = 128
        sdfs = []
        meshes = []
        level = 0  # ????

        effective_file_name = (
            f"{folder_name}/plane_test"
            if folder_name is not None
            else None
        )

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
    plt.imsave('test_siren/output_mesh.png', out_imgs[0])

        # for j in range(50):
        #     # Draw new coefficients from a normal distribution with the computed mean and std
        #     coefficients = torch.normal(mean, std)  # Shape: (400,)
        #
        #     # coefficients will have the same shape as one PCA component vector (400,)
        #     print("Mean:", mean)
        #     print("Standard Deviation:", std)
        #     print("Random Coefficients:", coefficients)
        #
        #     sampled = torch.zeros(50307)
        #
        #     print(pca_result.shape)
        #     print(basis.shape)
        #     for i in range(coefficients.shape[0]):
        #         sampled = sampled + basis[i] * coefficients[i]
        #
        #    # print(weights)
        #     #print(weights.shape)
        #     siren = generate_mlp_from_weights(sampled, mlp_kwargs)
        #
        #     model_input = get_mgrid(128, 2).unsqueeze(0)
        #     model_input = {'coords': model_input}
        #     result = siren(model_input)
        #     img = result['model_out']
        #     fig, axes = plt.subplots(1, 1, figsize=(9, 9))
        #     axes.imshow(img.cpu().view(128, 128, 3).detach().numpy())
        #     #os.makedirs("test", exist_ok=True)
        #     plt.savefig(f"test_siren/dupa_{j}.png")
        #
        #
        # siren = generate_mlp_from_weights(all_weights[5], mlp_kwargs)

        # model_input = get_mgrid(128, 2).unsqueeze(0)
        # model_input = {'coords': model_input}
        # result = siren(model_input)
        # img = result['model_out']
        # fig, axes = plt.subplots(1, 1, figsize=(9, 9))
        # axes.imshow(img.cpu().view(128, 128, 3).detach().numpy())
        # os.makedirs("test", exist_ok=True)
        # plt.savefig(f"test_siren/dupa_ref.png")



