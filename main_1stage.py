import os

from dataset import VoxelDataset, WeightDataset, CelebAHQ, CelebAHQ_collate_fn
from hd_utils import Config, get_mlp
from hyperdiffusion import HyperDiffusion

# Using it to make pyrender work on clusters
from hyperdiffusion_2d_img import HyperDiffusion_2d_img
from mlp_models import ImplicitMLP
from scheduler import PrefixStepLRScheduler

os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
from datetime import datetime
from os.path import join

import hydra
import multiprocessing as mp
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm.autonotebook import tqdm
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import ldm.ldm.modules.diffusionmodules.openaimodel
import wandb
from transformer import Transformer

sys.path.append("siren")


@hydra.main(
    version_base=None,
    config_path="configs/diffusion_configs",
    config_name="train_plane",
)
def main(cfg: DictConfig):
    Config.config = config = cfg
    method = Config.get("method")
    mlp_kwargs = None

    # In HyperDiffusion, we need to know the specifications of MLPs that are used for overfitting
    if "hyper" in method:
        mlp_kwargs = Config.config["mlp_config"]["params"]

    run = wandb.init(
        project="hyperdiffusion",
        dir=config["tensorboard_log_dir"],
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        tags=[Config.get("mode")],
        mode="disabled" if Config.get("disable_wandb") else "online",
        config=dict(config),
    )
    wandb.define_metric("*", step_metric="global_step")
    wandb.define_metric("epoch_loss", step_metric="epoch")
    wandb_logger = WandbLogger()
    wandb_logger.log_text("config", ["config"], [[str(config)]])
    print("wandb", wandb.run.name, wandb.run.id)

    train_dt = val_dt = test_dt = None

    # Although it says train, it includes all the shapes but we only extract training ones in WeightDataset
    data_folder_train = Config.get("data_folder_train")
    experiment_name = Config.get("experiment_name")
    current_path = os.getcwd()
    cache_path = os.path.join(current_path, "work_dirs", experiment_name)
    Config.set('cache_dir', cache_path)


    # Initialize Transformer for HyperDiffusion
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
        ).cuda()
    # Initialize UNet for Voxel baseline
    else:
        model = ldm.ldm.modules.diffusionmodules.openaimodel.UNetModel(
            **Config.config["unet_config"]["params"]
        ).float()

    dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])
    data_res = Config.get("image_resolution")

    if method == "hyper_3d":
        mlps_folder_all = data_folder_train
        # all_object_names = np.array(
        #     [obj for obj in os.listdir(dataset_path) if ".lst" not in obj]
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
        #     # Sanity checking the train, val and test splits
        #     assert len(train_idx.intersection(val_idx.intersection(test_idx))) == 0
        #     assert len(train_idx.union(val_idx.union(test_idx))) == total_size
        #     assert (
        #         len(train_idx) == train_size
        #         and len(val_idx) == val_size
        #         and len(test_idx) == test_size
        #     )
        #
        #     np.savetxt(
        #         os.path.join(dataset_path, "train_split.lst"),
        #         all_object_names[list(train_idx)],
        #         delimiter=" ",
        #         fmt="%s",
        #     )
        #     np.savetxt(
        #         os.path.join(dataset_path, "val_split.lst"),
        #         all_object_names[list(val_idx)],
        #         delimiter=" ",
        #         fmt="%s",
        #     )
        #     np.savetxt(
        #         os.path.join(dataset_path, "test_split.lst"),
        #         all_object_names[list(test_idx)],
        #         delimiter=" ",
        #         fmt="%s",
        #     )

        # val_object_names = np.genfromtxt(
        #     os.path.join(dataset_path, "val_split.lst"), dtype="str"
        # )
        # val_object_names = set([str.split(".")[0] for str in val_object_names])
        # test_object_names = np.genfromtxt(
        #     os.path.join(dataset_path, "test_split.lst"), dtype="str"
        # )
        # test_object_names = set([str.split(".")[0] for str in test_object_names])
        # assert len(train_object_names) == train_size, f"{len(train_object_names)} {train_size}"

        train_dt = CelebAHQ(downsampled=True,
                            resolution=data_res,
                            dataset_root=data_folder_train)
        train_dl = DataLoader(
            train_dt,
            batch_size=Config.get("batch_size"),
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=CelebAHQ_collate_fn
        )

    print(
        "Train dataset length: {}".format(len(train_dt))
    )
    inr_model = ImplicitMLP(B_path=mlp_kwargs['B_path'])
    state_dict = inr_model.state_dict()
    weights = []
    shapes = []
    for weight in state_dict:
        shapes.append(np.prod(state_dict[weight].shape))
        weights.append(state_dict[weight].flatten().cpu())
    input_data = torch.hstack(weights).unsqueeze(0)

    print(
        "Input data shape, min, max:",
        input_data.shape,
        input_data.min(),
        input_data.max(),
    )

    best_model_save_path = Config.get("best_model_save_path")
    model_resume_path = Config.get("model_resume_path")

    # Initialize HyperDiffusion
    diffuser = HyperDiffusion_2d_img(
        model, train_dt, val_dt, test_dt, mlp_kwargs, input_data.shape, method, Config.get("cache_size"), cfg
    )

    diffuser.logger = run

    # Specify where to save checkpoints
    checkpoint_path = join(
        config["tensorboard_log_dir"],
        "lightning_checkpoints",
        f"{str(datetime.now()).replace(':', '-') + '-' + wandb.run.name + '-' + wandb.run.id}",
    )
    # best_acc_checkpoint = ModelCheckpoint(
    #     save_top_k=1,
    #     monitor="val/1-NN-CD-acc",
    #     mode="min",
    #     dirpath=checkpoint_path,
    #     filename="best-val-nn-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    # )
    #
    # best_mmd_checkpoint = ModelCheckpoint(
    #     save_top_k=1,
    #     monitor="val/lgan_mmd-CD",
    #     mode="min",
    #     dirpath=checkpoint_path,
    #     filename="best-val-mmd-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    # )
    #
    # last_model_saver = ModelCheckpoint(
    #     dirpath=checkpoint_path,
    #     filename="last-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    #     save_on_train_epoch_end=True,
    # )

    #lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    optimizer = torch.optim.AdamW(diffuser.parameters(), lr=Config.get("lr"))

    if config["scheduler"]:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config["scheduler_step"], gamma=0.9
        )

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffuser.to(device)

    global_step = 0
    num_epochs = Config.get("epochs")

    if Config.get("mode") == "train":
        with tqdm(total=len(train_dl) * num_epochs) as pbar:
            for epoch in range(num_epochs):
                print(f"epoch: {epoch}")

                # Training phase
                outputs = []
                diffuser.train()  # Set model to training mode
                #total_train_loss = 0.0
                for batch_idx, data in enumerate(train_dl):
                    #data = data.to(device)
                    print(batch_idx, len(train_dl), global_step)

                    optimizer.zero_grad()  # Zero gradients

                    if epoch >= 10:
                        diffuser.cfg['inverse_steps'] = 3
                    if epoch >= 650:
                        diffuser.cfg['inverse_steps'] = 1

                    loss = diffuser.training_step(data, optimizer, global_step)  # Forward pass
                    outputs.append(loss)
                    global_step += 1
                scheduler.step()
                    # Accumulate gradient batches if specified
                    #if (batch_idx + 1) % cfg.accumulate_grad_batches == 0:
                    #    optimizer.step()
                    #   optimizer.zero_grad()

                epoch_loss = sum(output for output in outputs) / len(outputs)
                run.log({"epoch": epoch, "epoch_loss": epoch_loss})
                log_interval = int(Config.get("log_interval"))
                if global_step % log_interval == 0:
                    run.log({"global_step": global_step / log_interval, "epoch": epoch})
                    run.log({"global_step": global_step / log_interval, "lr-AdamW": optimizer.param_groups[0]['lr']})
                # Learning rate step (if using a scheduler)
                # Validation phase
                if epoch % Config.get("val_fid_calculation_period") == 0:
                    diffuser.eval()  # Set model to evaluation mode
                    with torch.no_grad():
                        diffuser.validation_step(epoch)

                # Optionally save the model after certain epochs
                # Saving phase
                #if (epoch + 1) % Config.get("model_save_period") == 0:
                #    torch.save(diffuser.state_dict(), f'{Config.get("model_save_path")}/model_epoch_{epoch}.pt')

    wandb_logger.finalize("Success")


if __name__ == "__main__":
    main()
