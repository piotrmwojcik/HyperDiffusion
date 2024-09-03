import copy
import os

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import wandb
from diffusion.gaussian_diffusion import (GaussianDiffusion, LossType,
                                          ModelMeanType, ModelVarType)
from hd_utils import (Config, calculate_fid_3d, generate_mlp_from_weights,
                      render_mesh, render_meshes)
from siren import sdf_meshing
from siren.dataio import anime_read, get_mgrid
from siren.experiment_scripts.test_sdf import SDFDecoder


class HyperDiffusion_2d_img(pl.LightningModule):
    def __init__(
        self, model, train_dt, val_dt, test_dt, mlp_kwargs, image_shape, method, cfg, pca
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.method = method
        self.mlp_kwargs = mlp_kwargs
        self.train_dt = train_dt
        self.pca = pca
        self.test_dt = test_dt
        self.ae_model = None
        self.sample_count = min(
            8, Config.get("batch_size")
        )  # it shouldn't be more than 36 limited by batch_size
        fake_data = torch.randn(*image_shape)

        encoded_outs = fake_data
        print("encoded_outs.shape", encoded_outs.shape)
        timesteps = Config.config["timesteps"]
        betas = torch.tensor(np.linspace(1e-4, 2e-2, timesteps))
        self.image_size = encoded_outs[:1].shape

        # Initialize diffusion utiities
        self.diff = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType[cfg.diff_config.params.model_mean_type],
            model_var_type=ModelVarType[cfg.diff_config.params.model_var_type],
            loss_type=LossType[cfg.diff_config.params.loss_type],
            diff_pl_module=self,
        )

    def forward(self, images):
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(images.shape[0],))
            .long()
            .to(self.device)
        )
        images = images * self.cfg.normalization_factor
        x_t, e = self.diff.q_sample(images, t)
        x_t = x_t.float()
        e = e.float()
        return self.model(x_t, t), e

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=Config.get("lr"))
        if self.cfg.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.cfg.scheduler_step, gamma=0.9
            )
            return [optimizer], [scheduler]
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # Extract input_data (either voxel or weight) which is the first element of the tuple
        input_data = train_batch[0]

        # At the first step output first element in the dataset as a sanit check
        if "hyper" in self.method and self.trainer.global_step % 50 == 0:
            curr_weights = Config.get("curr_weights")
            img = input_data[0].flatten()[:curr_weights]
            mlp = generate_mlp_from_weights(img, self.mlp_kwargs)
            model_input = get_mgrid(128, 2).unsqueeze(0)
            model_input = {'coords': model_input}
            result = mlp(model_input)
            img = result['model_out'][0].view(1, 128, 128, 3)
            img_min = img.min().item()
            img_max = img.max().item()
            print(img_min, img_max)
            img = (img + 1) / 2
            img = (img * 255).byte().permute(0, 3, 1, 2)
            # print(img.shape)
            # images = wandb.Image(img, caption="")
            # wandb.log({"examples": images})
            self.logger.log_image("train", [img])
            #sdf_decoder = SDFDecoder(
            #    self.mlp_kwargs.model_type,
            #    None,
            #    "nerf" if self.mlp_kwargs.model_type == "nerf" else "mlp",
            #    self.mlp_kwargs,
            #)
            #sdf_decoder.model = mlp.cuda()
            # if not self.mlp_kwargs.move:
            #     sdf_meshing.create_mesh(
            #         sdf_decoder,
            #         "meshes/first_mesh",
            #         N=128,
            #         level=0.5 if self.mlp_kwargs.output_type == "occ" else 0,
            #     )

            #print("Input images shape:", input_data.shape)

        # Output statistics every 100 step
        if self.trainer.global_step % 10 == 0:
            print(input_data.shape)
            print(
                "Orig weights[0].stats",
                input_data.min().item(),
                input_data.max().item(),
                input_data.mean().item(),
                input_data.std().item(),
            )

        # Sample a diffusion timestep
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(input_data.shape[0],))
            .long()
            .to(self.device)
        )
        centered_vector = input_data - torch.tensor(self.pca.mean_).cuda()
        pca_components = torch.tensor(self.pca.components_).cuda().t()

        projected_vector = torch.matmul(centered_vector, pca_components)

        # Execute a diffusion forward pass
        loss_terms = self.diff.training_losses(
            self.model,
            projected_vector * self.cfg.normalization_factor,
            t,
            self.mlp_kwargs,
            self.logger,
            model_kwargs=None,
        )

        loss_mse = loss_terms["loss"].mean()
        self.log("train_loss", loss_mse)

        loss = loss_mse
        return loss

    def validation_step(self, val_batch, batch_idx):
        x_0s = self.diff.ddim_sample_loop(
            self.model, (16, *self.image_size[1:]), clip_denoised=False
        )
        x_0s = (x_0s / self.cfg.normalization_factor)
        x_0s = torch.matmult(x_0s, torch.tensor(self.pca.components_).cuda()) + torch.tensor(self.pca.mean_).cuda()

        print(x_0s.shape)
        print(
            "Sampled weights[0].stats",
            x_0s.min().item(),
            x_0s.max().item(),
            x_0s.mean().item(),
            x_0s.std().item(),
        )

        weights = x_0s[0].view(-1)
        print(weights)
        print(weights.shape)
        siren = generate_mlp_from_weights(weights, self.mlp_kwargs)
        #print(self.mlp_kwargs.model_type)

        model_input = get_mgrid(128, 2).unsqueeze(0)
        model_input = {'coords': model_input}
        result = siren(model_input)
        img = result['model_out'][0].view(1, 128, 128, 3)
        #print(img[0, :, :, 0])
        #print('!!!')
        #print(img[0, :, :, 1])
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)
        img = (img * 255).byte().permute(0, 3, 1, 2)
        #print(img)
        #print('!!!')
        #print(img.shape)
        #images = wandb.Image(img, caption="")
        #wandb.log({"examples": images})
        self.logger.log_image("val", [img])
        #metric_fn = (
        #    self.calc_metrics_4d
        #    if self.cfg.mlp_config.params.move
        #    else self.calc_metrics
        #)
        #metrics = metric_fn("train")
        #for metric_name in metrics:
        #    self.log("train/" + metric_name, metrics[metric_name])
        #metrics = metric_fn("val")
        #for metric_name in metrics:
        #    self.log("val/" + metric_name, metrics[metric_name])

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        epoch_loss = sum(output["loss"] for output in outputs) / len(outputs)
        self.log("epoch_loss", epoch_loss)

    def print_summary(self, flat, func):
        var = func(flat, dim=0)
        print(
            var.shape,
            var.mean().item(),
            var.std().item(),
            var.min().item(),
            var.max().item(),
        )
        print(var.shape, func(flat))

    def test_step(self, *args, **kwargs):
        # Then, sampling some new shapes -> outputting and rendering them
        x_0s = self.diff.ddim_sample_loop(
            self.model, (16, *self.image_size[1:]), clip_denoised=False
        )
        x_0s = x_0s / self.cfg.normalization_factor

        print(
            "x_0s[0].stats",
            x_0s.min().item(),
            x_0s.max().item(),
            x_0s.mean().item(),
            x_0s.std().item(),
        )
