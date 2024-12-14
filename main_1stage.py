import os

from dataset import VoxelDataset, WeightDataset, CelebAHQ, CelebAHQ_collate_fn
from hd_utils import Config, get_mlp

# Using it to make pyrender work on clusters
from hyperdiffusion_2d_img import HyperDiffusion_2d_img
from mlp_models import ImplicitMLP, ParallelImplicitMLP, ParallelImplicitShortMLP, ImplicitMLPShort
from scheduler import PrefixStepLRScheduler

os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
from datetime import datetime
from os.path import join

import hydra
import multiprocessing as mp
import numpy as np
#import pytorch_lightning as pl
import torch
import copy
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm.autonotebook import tqdm
from omegaconf import DictConfig
#from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.loggers import WandbLogger
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
    #wandb_logger = WandbLogger()
    #wandb_logger.log_text("config", ["config"], [[str(config)]])
    print("wandb", wandb.run.name, wandb.run.id)

    train_dt = val_dt = test_dt = None

    # Although it says train, it includes all the shapes but we only extract training ones in WeightDataset
    data_folder_train = Config.get("data_folder_train")
    experiment_name = Config.get("experiment_name")
    current_path = os.getcwd()
    cache_path = os.path.join(current_path, "work_dirs",  experiment_name, "cache")
    model_save_path = os.path.join(current_path, "work_dirs", experiment_name, "checkpoint")
    Config.set('cache_dir', cache_path)
    Config.set('model_save_path', model_save_path)

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

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
        )
        #if torch.cuda.device_count() >= 2:
        #   print("Using", torch.cuda.device_count(), "GPUs!")
        #   model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))  # Specify GPUs to use
        #else:
        #    print("Not enough GPUs available.")
        model = model.cuda()
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
    inr_model = ImplicitMLPShort()
    mlps = [ImplicitMLPShort() for _ in range(Config.get("batch_size"))]
    MLP = ParallelImplicitShortMLP(mlps).cuda()
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
    #MLP = ParallelImplicitMLP(B=torch.load(Config.get("B_path")), batch_size=Config.get("batch_size"))

    diffuser = HyperDiffusion_2d_img(
        model, train_dt, val_dt, test_dt, mlp_kwargs, input_data.shape, method, Config.get("cache_size"), cfg
    )
    coords = train_dt[0]['coords'].unsqueeze(0).cuda()
    coords = diffuser.gff(coords)
    coords = rearrange(coords, "b c h w -> (b h w) c").cuda().detach()

    diffuser.logger = run
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
    code_optimizer = diffuser.build_optimizer(MLP, cfg)
    warmup_epochs = 15
    warmup_ratio = 0.001

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            lr = warmup_ratio + (1.0 - warmup_ratio) * (epoch / warmup_epochs)
            return lr
        else:
            # After warm-up, switch to a StepLR-like decay by returning 1 (base_lr)
            return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    step_scheduler = StepLR(optimizer, step_size=config["scheduler_step"], gamma=0.9)
    scheduler = (warmup_scheduler, step_scheduler) if config["scheduler"] else None

    global_step = 0
    epoch_start = 0

    if Config.get('model_resume_path') is not None:
        checkpoint = torch.load(Config.get('model_resume_path'))
        #new_model_dict = {}
        #for key, value in checkpoint['diffuser'].items():
        #    if key.startswith("model"):  # Check if the key starts with 'model'
        #        new_key = key.replace("model", "model.module",
        #                              1)  # Replace 'model' with 'model.module' only at the start
        #    else:
        #        new_key = key  # Keep the key unchanged if it doesn't start with 'model'
        #    new_model_dict[new_key] = value
        #checkpoint['diffuser'] = new_model_dict

        global_step = checkpoint.get('global_step', 0)
        epoch_start = checkpoint.get('epoch', 0)

        model_msg = diffuser.load_state_dict(checkpoint['diffuser'])
        print('Loaded model ',  model_msg)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded optimizer')

        #del checkpoint['code_optimizer']['param_groups']
        code_optimizer_state_dict = code_optimizer.state_dict()
        code_optimizer_state_dict['state'] = copy.deepcopy(checkpoint['code_optimizer']['state'])
        code_optimizer.load_state_dict(code_optimizer_state_dict)
        #print(checkpoint['code_optimizer']['state'])
        #print(code_optimizer.state_dict()['state'])
        print('Loaded code optimizer')

        # Only load the scheduler if you're using one in your config
        if config["scheduler"]:
            step_scheduler.load_state_dict(checkpoint['scheduler'])
            print('Loaded scheduler')

    code_optimizer_state = diffuser.optimizer_state_to(code_optimizer.state_dict(), device='cpu', dtype=torch.float32)

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffuser.to(device)

    num_epochs = Config.get("epochs")

    if Config.get("mode") == "train":
        for epoch in range(epoch_start, num_epochs):
            print(f"epoch: {epoch + 1}")
            # Training phase
            outputs = []
            diffuser.train()  # Set model to training mode
            #start_time = time.time()
            save_to_disk = ((epoch + 1) % Config.get("model_save_period") == 0) or (epoch == num_epochs - 1)
            with tqdm(train_dl, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for batch_idx, data in enumerate(pbar):
                    #print((batch_idx, global_step))
                    #data = data.to(device)
                    optimizer.zero_grad()  # Zero gradients

                    inverse_steps_schedule = [(k, v) for d in diffuser.cfg['inverse_steps_schedule'] for k, v in d.items()]
                    for p in inverse_steps_schedule:
                        if epoch >= p[0]:
                            diffuser.cfg['inverse_steps'] = p[1]
                    code_lr_schedule = [(k, v) for d in diffuser.cfg['code_lr_schedule'] for k, v in d.items()]
                    for p in code_lr_schedule:
                        if epoch >= p[0]:
                            diffuser.cfg['code_lr'] = p[1]
                    code_loss_weight_schedule = [(k, v) for d in diffuser.cfg['code_loss_weight_schedule'] for k, v in d.items()]
                    for p in code_loss_weight_schedule:
                        if epoch >= p[0]:
                            diffuser.cfg['code_loss_weight'] = p[1]
                    loss, code_optimizer_state_ = diffuser.training_step(data, coords, optimizer, code_optimizer_state, global_step, save_to_disk, MLP, epoch)  # Forward pass
                    outputs.append(loss)
                    global_step += 1
                    pbar.set_postfix({"diff_loss": loss.item()})
                if scheduler is not None:

                    # Once warm-up phase is over, switch to StepLR
                    if epoch >= warmup_epochs:
                        step_scheduler.step()
                    else:
                        warmup_scheduler.step()

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

                diffuser.optimizer_state_copy(code_optimizer_state_, code_optimizer_state, device='cpu', dtype=torch.float32)
                if save_to_disk:
                    checkpoint = {
                        'diffuser': diffuser.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'code_optimizer': code_optimizer_state,
                        'scheduler': step_scheduler.state_dict() if scheduler is not None else None,  # Save the scheduler's state
                        'epoch': epoch,
                        'global_step': global_step
                    }

                    torch.save(checkpoint, f'{Config.get("model_save_path")}/model_epoch_{epoch}.pth')
    elif Config.get("mode") == "test":
        diffuser.test_step()
        # Optionally save the model after certain epochs
        # Saving phase
        #if (epoch + 1) % Config.get("model_save_period") == 0:
        #    torch.save(diffuser.state_dict(), f'{Config.get("model_save_path")}/model_epoch_{epoch}.pt')

    #wandb_logger.finalize("Success")


if __name__ == "__main__":
    main()
