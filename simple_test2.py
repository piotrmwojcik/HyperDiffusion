from mlp_models import SingleBVPNet
from hd_utils import generate_mlp_from_weights
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
from transformer import Transformer
from torch.utils.data import DataLoader, random_split


def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


if __name__ == '__main__':
    config_path = 'configs/diffusion_configs/train_car_2d_relu.yaml'

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
        from sklearn.decomposition import PCA
        pca = PCA(n_components=100)  # You can choose the number of components
        pca_result = pca.fit_transform(all_weights)
        print(pca_result.shape)
            # noise = torch.randn_like(weights) * 0.01
            # weights = weights + noise
            # print(weights)
            # print(weights.shape)
            # siren = generate_mlp_from_weights(weights, mlp_kwargs)
            #
            # model_input = get_mgrid(128, 2).unsqueeze(0)
            # model_input = {'coords': model_input}
            # result = siren(model_input)
            # img = result['model_out']
            # fig, axes = plt.subplots(1, 1, figsize=(9, 9))
            # axes.imshow(img.cpu().view(128, 128, 3).detach().numpy())
            # os.makedirs("test", exist_ok=True)
            # plt.savefig(f"test_siren/img_{i}.png")



