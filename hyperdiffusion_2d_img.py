import copy
import os

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from collections import defaultdict, abc as container_abcs
from itertools import chain
from functools import partial
from six.moves import map, zip

import wandb
from diffusion.gaussian_diffusion import (GaussianDiffusion, LossType,
                                          ModelMeanType, ModelVarType)
from hd_utils import (Config, calculate_fid_3d, generate_mlp_from_weights,
                      render_mesh, render_meshes)
from siren import sdf_meshing, dataio
from siren.dataio import anime_read, get_mgrid, get_grid
from siren.experiment_scripts.test_sdf import SDFDecoder


class HyperDiffusion_2d_img(torch.nn.Module):
    def __init__(
        self, model, train_dt, val_dt, test_dt, mlp_kwargs, image_shape, method, cache_size, cfg
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.method = method
        self.mlp_kwargs = mlp_kwargs
        self.train_dt = train_dt
        self.test_dt = test_dt
        self.cache_size = cache_size
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

        if cache_size > 0:
            split_points = np.round(np.linspace(0, cache_size, num=2)).astype(np.int64)
            inds = np.arange(start=split_points[0], stop=split_points[1])
            self.cache = {ind: None for ind in inds}
        else:
            self.cache = None
        self.cache_loaded = False

    def optimizer_set_state(self, optimizer, state_dict):
        groups = optimizer.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                if key != "step":
                    if param.is_floating_point():
                        value = value.to(param.dtype)
                    value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v, key=k) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        optimizer.__setstate__({'state': state})

    def load_cache(self, data):
        #device = get_module_device(self)
        num_scenes = len(data['scene_id'])

        if self.cache is not None:
            if not self.cache_loaded:
                cache_load_from = self.train_cfg.get('cache_load_from', None)
                loaded = False
                if cache_load_from is not None:
                    cache_files = os.listdir(cache_load_from)
                    cache_files.sort()
                    if len(cache_files) > 0:
                        assert len(cache_files) == self.cache_size
                        for ind in self.cache.keys():
                            self.cache[ind] = torch.load(
                                os.path.join(cache_load_from, cache_files[ind]), map_location='cpu')
                        loaded = True
                        print('Loaded cache files from ' + cache_load_from)
                if not loaded:
                    print('Initialize codes from scratch.')
                self.cache_loaded = True
            cache_list = [self.cache[scene_id_single] for scene_id_single in data['scene_id']]
        else:
            cache_list = [None for _ in range(num_scenes)]
        code_list_ = []
        for scene_state_single in cache_list:
            if scene_state_single is None:
                code_list_.append(self.get_init_code_(None, device))
            else:
                if 'code_' in scene_state_single['param']:
                    code_ = scene_state_single['param']['code_'].to(dtype=torch.float32, device=device)
                else:
                    assert 'code' in scene_state_single['param']
                    code_ = self.code_activation.inverse(
                        scene_state_single['param']['code'].to(dtype=torch.float32, device=device))
                code_list_.append(code_.requires_grad_(True))

        code_optimizers = self.build_optimizer(code_list_, self.train_cfg)
        for ind, scene_state_single in enumerate(cache_list):
            if scene_state_single is not None and 'optimizer' in scene_state_single:
                self.optimizer_set_state(code_optimizers[ind], scene_state_single['optimizer'])
        return code_list_, code_optimizers

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

    def training_step(self, train_batch, global_step):
        # Extract input_data (either voxel or weight) which is the first element of the tuple
        input_data = train_batch[0].cuda()

        log_interval = int(Config.get("log_interval"))

        # At the first step output first element in the dataset as a sanit check
        if "hyper" in self.method and global_step % 50 == 0 and global_step % log_interval == 0:
            curr_weights = Config.get("curr_weights")
            img = input_data[0].flatten()[:curr_weights]
            mlp = generate_mlp_from_weights(img, self.mlp_kwargs)
            model_input = get_grid(64, 64, b=0).unsqueeze(0)#get_mgrid(128, 2).unsqueeze(0)

            model_input = {'coords': model_input}
            result = mlp(model_input)
            #print(img)
            img = dataio.lin2img(result['model_out'], (64, 64))
            img = dataio.rescale_img((img + 1) / 2, mode='clamp')
            img = (img * 255).byte()
            # print(img.shape)
            images = wandb.Image(img, caption="")
            # wandb.log({"examples": images})
            self.logger.log({"global_step": global_step / log_interval, "train": images})
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

        # Output statistics every 10 step
        if global_step % 10 == 0:
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
            .cuda()
        )

        # Execute a diffusion forward pass
        loss_terms = self.diff.training_losses(
            self.model,
            input_data * self.cfg.normalization_factor,
            t,
            self.mlp_kwargs,
            self.logger,
            model_kwargs=None,
        )

        loss_mse = loss_terms["loss"].mean()
        self.logger.log({"global_step": global_step, "train_loss": loss_mse})

        loss = loss_mse
        return loss

    def validation_step(self, epoch):

        x_0s = self.diff.ddim_sample_loop(
            self.model, (16, *self.image_size[1:]), clip_denoised=False
        )
        x_0s = (x_0s / self.cfg.normalization_factor)

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

        model_input = get_grid(64, 64, b=0).unsqueeze(0)
        model_input = {'coords': model_input}
        result = siren(model_input)
        print(result['model_out'].shape)
        # print(img)
        img = dataio.lin2img(result['model_out'], (64, 64))
        img = dataio.rescale_img((img + 1) / 2, mode='clamp')
        img = (img * 255).byte()
        #print(img)
        #print('!!!')
        #print(img.shape)
        images = wandb.Image(img, caption="")
        #wandb.log({"examples": images})
        self.logger.log({"global_step": epoch, "val": images})
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
        self.log({"epoch_loss": epoch_loss})

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
