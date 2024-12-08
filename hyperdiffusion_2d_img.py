import copy
import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh
import math
import skimage.measure as measure
from einops import rearrange
import torchvision.utils as vutils
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from collections import defaultdict, abc as container_abcs
from itertools import chain
from functools import partial
from six.moves import map, zip

import wandb
from diffusion.gaussian_diffusion import (GaussianDiffusion, LossType,
                                          ModelMeanType, ModelVarType)
from ema import ExponentialMovingAverage
from hd_utils import (Config, calculate_fid_3d, generate_mlp_from_weights,
                      render_mesh, render_meshes)
from mlp_models import ImplicitMLP, ParallelImplicitMLP, GaussianFourierFeatureTransform, ImplicitMLPShort, \
    ParallelImplicitShortMLP, image_psnr, image_mse
from reg_loss import RegLoss
from siren import sdf_meshing, dataio
from siren.dataio import anime_read, get_mgrid, get_grid
from siren.experiment_scripts.test_sdf import SDFDecoder


class HyperDiffusion_2d_img(torch.nn.Module):
    def __init__(
        self, model, train_dt, val_dt, test_dt, mlp_kwargs, image_shape, method, cache_size, cfg
    ):
        super().__init__()
        self.model = model
        if Config.get('use_ema'):
            self.ema_model = ExponentialMovingAverage(model, decay=0.999, rampup_rate=0.05, rampup_kimg=4, batch_size=16, eps=1e-8)
        else:
            self.ema_model = None
        self.cfg = cfg
        self.method = method
        self.mlp_kwargs = mlp_kwargs
        self.train_dt = train_dt
        self.test_dt = test_dt
        self.loaded_B = torch.load(Config.get("B_path"))
        self.gff = GaussianFourierFeatureTransform(B=self.loaded_B, mapping_dim=128).cuda()
        self.cache_size = cache_size
        self.file_queues = None
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

        self.register_buffer('norm_factor', torch.ones(1, dtype=torch.float))
        self.momentum = Config.config["norm_momentum"]

        # Initialize diffusion utiities
        self.diff = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType[cfg.diff_config.params.model_mean_type],
            model_var_type=ModelVarType[cfg.diff_config.params.model_var_type],
            loss_type=LossType[cfg.diff_config.params.loss_type],
            diff_pl_module=self,
        )

        self.reg_loss = None #RegLoss(loss_weight=4e-3)

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

    def get_init_code_(self, device=None):
        #print('!!!!')
        model = ImplicitMLP(B=self.loaded_B)
        #checkpoint_path = "/data/pwojcik/siren/logs/033013.jpg/checkpoints/model_epoch_14500.pth"
        #checkpoint = torch.load(checkpoint_path, map_location=device)
        #model.load_state_dict(checkpoint)

        state_dict = model.state_dict()
        weights = []
        shapes = []
        for weight in state_dict:
            shapes.append(np.prod(state_dict[weight].shape))
            weights.append(state_dict[weight].flatten().cpu())
        weights = torch.hstack(weights).requires_grad_()

        return weights

    def build_optimizer(self, mlp, cfg):
        optimizer_cfg = dict(type=cfg['code_optimizer'], lr=cfg['code_lr'])
        optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
        code_optimizer = optimizer_class(mlp.parameters(), **optimizer_cfg)
        return code_optimizer

    def load_cache(self, data):
        #device = get_module_device(self)
        num_scenes = len(data['scene_id'])

        if self.cache is not None:
            if not self.cache_loaded:
                #cache_load_from = self.cfg.get('cache_load_from', None)
                loaded = False
                cache_dir = Config.get('cache_dir')
                cache_files = [f for f in os.listdir(cache_dir) if 'code' in f]
                cache_files.sort(key=lambda x: int(x.replace('code_', '').split('.')[0]))
                if len(cache_files) > 0:
                    assert len(cache_files) == self.cache_size
                    for ind in self.cache.keys():
                        self.cache[ind] = torch.load(
                            os.path.join(cache_dir, cache_files[ind]), map_location='cpu')
                    loaded = True
                    print('Loaded cache files from ' + cache_dir + '.')

                if not loaded:
                    print('Initialize codes from scratch.')
                self.cache_loaded = True
            cache_list = [self.cache[scene_id_single] for scene_id_single in data['scene_id']]
        else:
            cache_list = [None for _ in range(num_scenes)]
        code_list_ = []
        for scene_state_single in cache_list:
            if scene_state_single is None:
                code_list_.append(self.get_init_code_(None))
            else:
                assert 'code_' in scene_state_single['param']
                code_ = scene_state_single['param']['code_'].to(dtype=torch.float32)
                code_list_.append(code_.requires_grad_(True))
        return code_list_

    def optimizer_state_to(self, state_dict, device=None, dtype=None):
        assert dtype.is_floating_point
        out = dict(state=dict())
        for key_state_single, state_single in state_dict['state'].items():
            state_single_out = dict()
            for key, val in state_single.items():
                if isinstance(val, torch.Tensor):
                    if key != 'step' and val.dtype != dtype:
                        val = val.clamp(min=torch.finfo(dtype).min, max=torch.finfo(dtype).max)
                    state_single_out[key] = val.to(
                        device=device, dtype=None if key == 'step' else dtype)
                else:
                    state_single_out[key] = val
            out['state'][key_state_single] = state_single_out
        #print(out['state']['step'])
        return out

    def out_dict_to(self, d, device=None, code_dtype=torch.float32):
        assert code_dtype.is_floating_point
        return dict(
            scene_id=d['scene_id'],
            param=dict(
                code_=d['param']['code_'].clamp(
                    min=torch.finfo(code_dtype).min, max=torch.finfo(code_dtype).max
                ).to(device=device, dtype=code_dtype)))

    def load_tensor_to_dict(self, d, key, value, device=None, dtype=None):
        assert dtype.is_floating_point
        if isinstance(value, torch.Tensor):
            if key not in ['step'] and value.dtype != dtype:
                value = value.clamp(min=torch.finfo(dtype).min, max=torch.finfo(dtype).max)
            if key in d:
                d[key].copy_(value)
            else:
                d[key] = value.to(
                    device=device, dtype=None if key in ['step'] else dtype)
        else:
            d[key] = value

    def optimizer_state_copy(self, d_src, d_dst, device=None, dtype=None):
        for key_state_single, state_single in d_src['state'].items():
            if key_state_single not in d_dst['state']:
                d_dst['state'][key_state_single] = dict()
            for key, val in state_single.items():
                self.load_tensor_to_dict(d_dst['state'][key_state_single], key, val,
                                         device=device, dtype=dtype)

    def save_cache(self, code_list_, scene_name, save_to_disk):
        code_dtype = code_list_[0].dtype
        if Config.get('cache_dir') is not None:
            save_dir = Config.get('cache_dir')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = None

        for ind, code_single_ in enumerate(code_list_):
            out = dict(
                scene_id=scene_name[ind],
                param=dict(
                    code_=code_single_.data))
            if self.cache is not None:
                scene_name_single = scene_name[ind]
                if self.cache[scene_name_single] is None:
                    self.cache[scene_name_single] = self.out_dict_to(
                        out, device='cpu', code_dtype=code_dtype)
                else:
                    if 'scene_id' not in self.cache[scene_name_single]:
                        self.cache[scene_name_single]['scene_id'] = out['scene_id']
                    if 'code_' in self.cache[scene_name_single]['param']:
                        del self.cache[scene_name_single]['param']['code_']
                    for key, val in out['param'].items():
                        self.load_tensor_to_dict(self.cache[scene_name_single]['param'], key, val,
                                                 device='cpu', dtype=code_dtype)
                    # if 'optimizer' in self.cache[scene_name_single]:
                    #     self.optimizer_state_copy(out['optimizer'], self.cache[scene_name_single]['optimizer'],
                    #                              device='cpu', dtype=optimizer_dtype)
                    # else:
                    # #     self.cache[scene_name_single]['optimizer'] = self.optimizer_state_to(
                    #         out['optimizer'], device='cpu', dtype=optimizer_dtype)
                if save_dir is not None and save_to_disk:
                    if self.file_queues is not None:
                        self.file_queues[ind // self.num_file_writers].put(
                            self.out_dict_to(out, device='cpu', code_dtype=code_dtype))
                    else:
                        torch.save(
                            self.out_dict_to(out, device='cpu', code_dtype=code_dtype),
                            os.path.join(save_dir, f"code_{scene_name[ind]}" + '.pth'))

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

    def inverse_code_1b1(self, gt_imgs, grids, code_, optimizer_state, prior_grad, cfg):
        n_inverse_steps = cfg['inverse_steps']

        x = grids[0].unsqueeze(0).cuda()
        x = self.gff(x)
        x = rearrange(x, "b c h w -> (b h w) c")

        mlps = [generate_mlp_from_weights(code_single, self.mlp_kwargs, self.loaded_B, short=True) for code_single in code_]
        mlp = ParallelImplicitShortMLP(mlps)
        #mlp = torch.nn.DataParallel(mlp, device_ids=list(range(torch.cuda.device_count())))
        #mlp_without_ddp = mlp#.module
        #grids = grids.cuda()
        gt_imgs = gt_imgs.cuda()
        code_optimizer = self.build_optimizer(mlp, cfg)
        #for sidx, state in enumerate(code_optimizer_states):
        if optimizer_state is not None:
            optim = code_optimizer.state_dict()
            optim['state'] = optimizer_state['state']
            code_optimizer.load_state_dict(optim)
        code_optimizer.zero_grad()

        if n_inverse_steps == 0:
            n_inverse_steps = 1
            update_grad = False
        else:
            update_grad = True

        #start = time.time()
        for inverse_step_id in range(n_inverse_steps):
            #psnr = []
            #mse_loss, psnr = mlp(x.clone(), gt_imgs.clone())
            outputs = mlp(x.clone())
            #torch.cuda.synchronize()
            #start = time.time()
            # num_gpus = torch.cuda.device_count()
            # chunk_size = outputs.size(0) // num_gpus
            # chunks = torch.split(outputs, chunk_size, dim=0)
            # processed_chunks = []  # List to store processed chunks
            # for chunk in chunks:
            #     processed_chunks.append(chunk)
            # outputs = torch.cat(processed_chunks, dim=1)

            mse_loss = image_mse(mask=None, model_output=outputs, gt=gt_imgs)['img_loss']
            mse_loss = mse_loss * Config.get('code_loss_weight')

            code_reg = None
            if self.reg_loss is not None:
                code_reg = self.reg_loss(mlp)
                #print(self.reg_loss(mlp))
                mse_loss = mse_loss + code_reg

            psnr = image_psnr(outputs, gt_imgs)['img_psnr']

            if update_grad:
                #grad_inner = torch.autograd.grad(
                #    mse_loss,
                #    mlp.parameters(),
                #    create_graph=False,
                #    retain_graph=False
                #)

                mse_loss.backward()

                prior_grad_ = torch.cat(prior_grad, dim=0).cuda()
                current_idx = 0
                for param in mlp.parameters():
                    grad = param.grad
                    grad = grad.to('cuda:0')
                    param = param.to('cuda:0')
                    grad_shape = grad.shape
                    num_params = np.product(list(grad.shape))
                    grad = grad.view(-1)
                    grad = grad + prior_grad_[current_idx:current_idx + num_params]
                    grad = grad.view(grad_shape)
                    param.grad = torch.zeros_like(param).to('cuda:0')
                    current_idx += num_params
                    param.grad.copy_(grad)
                #assert (current_idx == prior_grad_.shape[0])
                code_optimizer.step()

        #print()
        #end = time.time()
        #print(f"grad and optim {round(end - start, 3)} seconds")
        for idx, _mlp in enumerate(mlp.models):
            state_dict = _mlp.state_dict()
            weights = []
            for weight in state_dict:
                weights.append(state_dict[weight].flatten())
            code_[idx] = torch.hstack(weights)

        #psnr = torch.mean(torch.hstack(psnr))
        optim_state = code_optimizer.state_dict()
        del optim_state['param_groups']
        #print('state: ', code_optimizer.state_dict()['state'][0]['step'])
        return mse_loss, code_reg, psnr, optim_state

    def deep_copy_dict(self, input_dict):
        copied_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                # For tensors, use .clone() to make an independent copy
                copied_dict[key] = value.clone()
            elif isinstance(value, dict):
                # Recursively copy nested dictionaries
                copied_dict[key] = self.deep_copy_dict(value)
            else:
                # For other types, use copy.deepcopy to handle immutable and mutable types
                copied_dict[key] = copy.deepcopy(value)
        return copied_dict

    def training_step(self, train_batch, optimizer, code_optimizer_state, global_step, save_to_disk):
        # Extract input_data (either voxel or weight) which is the first element of the tuple
        input_img = train_batch['gt_img'][0].clone().detach().view(64, 64, 3).permute(2, 0, 1).cuda()

        log_interval = int(Config.get("log_interval"))

        if 'code_optimizer' in self.cfg:
            code_list_ = self.load_cache(train_batch)
            code = torch.stack(code_list_, dim=0).cuda()

        optimizer.zero_grad()
        # Sample a diffusion timestep
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(code.shape[0],))
            .long()
            .cuda()
        )

        #start_time = time.time()
        # Execute a diffusion forward pass
        loss_terms = self.diff.training_losses(
            self.model,
            code * self.cfg.normalization_factor,
            t,
            self.mlp_kwargs,
            self.logger,
            model_kwargs=None,
        )
        #print('!!! loss shape', loss_terms["loss"].shape)
        #end_time = time.time()
        #print(f"Time taken: {end_time - start_time:.4f} seconds")

        #print('!!!')
        #print(loss_terms["loss"].std())
        #print(loss_terms["loss"])
        #print(code.detach().square().std())

        loss_mse = loss_terms["loss"].mean()

        #norm_factor = code.detach().square().mean()
        #self.norm_factor[:] = (1 - self.momentum) * self.norm_factor \
        #                      + self.momentum * norm_factor
        #if global_step <= 1700:
        #    warmup_factor = (2.0 - (math.cos(math.pi * global_step / 1700) + 1.0)) / 2.0
        #else:
        #    warmup_factor = 1.0
        loss_mse = Config.get("loss_weight") * loss_mse #/ self.norm_factor

        loss_mse.backward()  # Backpropagation
        optimizer.step()
        if self.ema_model is not None:
            self.ema_model.update()
        #print(code)
        prior_grad = [code_.grad.data.clone() for code_ in code_list_]
        #for code_ in code_list_:
        #    print('!!', code_.grad)
        #print('before inverse code')
        start = time.time()
        inv_loss, code_reg, psnr, code_optim_state_ = self.inverse_code_1b1(train_batch['gt_img'], train_batch['coords'], code_list_,
                                                                            self.deep_copy_dict(code_optimizer_state),
                                                                            prior_grad, self.cfg)
        elapsed_time = time.time() - start
        print(f"Time taken for inverse_code_1b1 execution: {elapsed_time:.4f} seconds")
        #for code_ in code_list_:
        #    print(code_.grad)
        #code_optimizer_state_ = self.deep_copy_dict(optim_state)

        if "hyper" in self.method and global_step % 50 == 0 and global_step % log_interval == 0:
            mlp = generate_mlp_from_weights(code_list_[0], self.mlp_kwargs, self.loaded_B)
            #model_input = {'coords': model_input}
            input = train_batch['coords'][0].unsqueeze(0)
            inr_output = mlp({'coords': input})['model_out'][0].view(64, 64, 3).permute(2, 0, 1)

            #psnr_test = image_psnr(mlp({'coords': input})['model_out'], train_batch['gt_img'][16].unsqueeze(0))['img_psnr']
            #print('!!!!')
            #print(psnr_test)

            images = wandb.Image(input_img, caption="")
            inr_images = wandb.Image(inr_output, caption="")
            #inr_init_img = wandb.Image(inr_output_init, caption="")
            # wandb.log({"examples": images})
            self.logger.log({"global_step": global_step / log_interval, "gt": images})
            self.logger.log({"global_step": global_step / log_interval, "inr": inr_images})
            #self.logger.log({"global_step": global_step / log_interval, "inr_init": inr_init_img})


        # ==== save cache ====
        self.save_cache(code_list_, train_batch['scene_id'], save_to_disk)
        self.logger.log({"global_step": global_step, "diff_train_loss": loss_mse})
        self.logger.log({"global_step": global_step, "psnr": psnr})
        self.logger.log({"global_step": global_step, "inr_train_loss": inv_loss})
        self.logger.log({"global_step": global_step, "code_norm": code.detach().square().mean()})
        #self.logger.log({"global_step": global_step, "code_reg": code_reg})

        return loss_mse, code_optim_state_

    def test_step(self):
        if Config.get("use_ema"):
            model = self.ema_model.ema_model
        else:
            model = self.model

        image_list = []
        image_folder = "/data/pwojcik/CelebAHQ_5k/CelebAHQ/"
        # Define a transformation pipeline to load images as tensors
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor (values in range [0, 1])
        ])

        # Loop over all JPG files in the folder
        for image_file in os.listdir(image_folder):
            if image_file.endswith(".jpg"):
                # Load the image
                img_path = os.path.join(image_folder, image_file)
                img = Image.open(img_path).convert("RGB")  # Ensure RGB format

                # Apply the transform
                img_tensor = transform(img)  # Shape: [3, 64, 64]

                # Reshape and convert to byte format
                reshaped_tensor = img_tensor.permute(1, 2, 0).reshape(-1, 3) * 255  # Shape: [4096, 3]
                reshaped_tensor = reshaped_tensor.byte()  # Convert to byte

                # Append to the list
                image_list.append(reshaped_tensor)
        image_tensors = torch.stack(image_list)

        x_0s = self.diff.ddim_sample_loop(
            model, (256, *self.image_size[1:]), clip_denoised=False
        )
        x_0s = (x_0s / self.cfg.normalization_factor)

        images = []

        for img_id in range(x_0s.shape[0]):
            weights = x_0s[img_id].view(-1)
            siren = generate_mlp_from_weights(weights, self.mlp_kwargs, self.loaded_B)
            #print(self.mlp_kwargs.model_type)

            input = get_grid(64, 64, b=0).unsqueeze(0)
            result = siren({'coords': input})
            #print(result['model_out'].shape)
            img = dataio.lin2img(result['model_out'], (64, 64))
            img = dataio.rescale_img((img + 1) / 2, mode='clamp')
            img = (img * 255).byte()
            images.append(img)

        # Initialize a list to store the reshaped images
        images = torch.cat(images, dim=0)
        flattened_images = images.permute(0, 2, 3, 1).view(images.size(0), -1) / 255
        flattened_gt = image_tensors.view(image_tensors.size(0), -1) / 255
        #print('!!! ', flattened_images.shape, flattened_gt.shape)

        dot_products = torch.mm(flattened_images, flattened_gt.t())  # Shape: [N, N]
        norms1 = flattened_images.pow(2).sum(dim=1, keepdim=True)  # Shape: [N, 1]
        norms2 = flattened_gt.pow(2).sum(dim=1, keepdim=True)  # Shape: [N, 1]
        distances = norms1 - 2 * dot_products + norms2.t()  # Shape: [N, N]
        distances.fill_diagonal_(float('inf'))
        nearest_indices = distances.argmin(dim=1)  # Shape: [N]

        gt_grid = []

        # Output results
        for i, nearest_idx in enumerate(nearest_indices):
            print(
                f"Image {i} is closest to Image {nearest_idx} with L2 distance {distances[i, nearest_idx].item():.2f}")
            gt_grid.append(image_list[nearest_idx])
        gt_grid = torch.stack(gt_grid)
        print('!!! ', images.shape, gt_grid.shape)

        grid = vutils.make_grid(images, nrow=32, padding=0, normalize=False)
        gt_grid = vutils.make_grid(gt_grid.permute(0, 2, 1).view(images.shape), nrow=32, padding=0, normalize=False)
        print("Grid shape:", grid.shape)
        # Convert list of images to a grid
        #wandb_images = [wandb.Image(img.permute(1, 2, 0).cpu().numpy()) for img in images[:256]]
        wandb.log({"generated_images_grid": wandb.Image(grid, caption="generated")})
        wandb.log({"gt_images_grid": wandb.Image(gt_grid, caption="GT")})


    def validation_step(self, epoch):
        if Config.get("use_ema"):
            model = self.ema_model.ema_model
        else:
            model = self.model

        x_0s = self.diff.ddim_sample_loop(
            model, (16, *self.image_size[1:]), clip_denoised=False
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
        siren = generate_mlp_from_weights(weights, self.mlp_kwargs, self.loaded_B)
        #print(self.mlp_kwargs.model_type)

        input = get_grid(64, 64, b=0).unsqueeze(0)
        result = siren({'coords': input})
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

