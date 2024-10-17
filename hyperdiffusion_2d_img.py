import copy
import os
import time

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
                      render_mesh, render_meshes, image_mse, generate_mlp_from_weights_trainable)
from mlp_models import ImplicitMLP
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

    def get_init_code_(self, device=None):
        model = ImplicitMLP(B_path=self.mlp_kwargs['B_path'])
        state_dict = model.state_dict()
        weights = []
        shapes = []
        for weight in state_dict:
            shapes.append(np.prod(state_dict[weight].shape))
            weights.append(state_dict[weight].flatten().cpu())
        weights = torch.hstack(weights).requires_grad_()

        return weights

    def build_optimizer(self, mlps, cfg):
        optimizer_cfg = dict(type=cfg['code_optimizer'], lr=cfg['code_lr'])
        optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
        code_optimizer = [
            optimizer_class(mlp.parameters(), **optimizer_cfg) for mlp in mlps]
        return code_optimizer

    def load_cache(self, data):
        #device = get_module_device(self)
        num_scenes = len(data['scene_id'])

        if self.cache is not None:
            if not self.cache_loaded:
                cache_load_from = self.cfg.get('cache_load_from', None)
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
        optimizer_states = []
        for scene_state_single in cache_list:
            if scene_state_single is None:
                code_list_.append(self.get_init_code_(None))
                optimizer_states.append(None)
            else:
                assert 'code_' in scene_state_single['param']
                code_ = scene_state_single['param']['code_'].to(dtype=torch.float32)
                code_list_.append(code_.requires_grad_(True))
                if 'optimizer' in scene_state_single:
                    optimizer_states.append(scene_state_single['optimizer'])
                else:
                    optimizer_states.append(None)
        return code_list_, optimizer_states


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
        return out

    def out_dict_to(self, d, device=None, code_dtype=torch.float32, optimizer_dtype=torch.float32):
        assert code_dtype.is_floating_point and optimizer_dtype.is_floating_point
        return dict(
            scene_id=d['scene_id'],
            param=dict(
                code_=d['param']['code_'].clamp(
                    min=torch.finfo(code_dtype).min, max=torch.finfo(code_dtype).max
                ).to(device=device, dtype=code_dtype)),
            optimizer=self.optimizer_state_to(d['optimizer'], device=device, dtype=optimizer_dtype))

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

    def save_cache(self, code_list_, code_optimizers, scene_name):
        code_dtype = code_list_[0].dtype
        optimizer_dtype = torch.float32
        for ind, code_single_ in enumerate(code_list_):
            out = dict(
                scene_id=scene_name[ind],
                param=dict(
                    code_=code_single_.data),
                optimizer=code_optimizers[ind])
            if self.cache is not None:
                scene_name_single = scene_name[ind]
                if self.cache[scene_name_single] is None:
                    self.cache[scene_name_single] = self.out_dict_to(
                        out, device='cpu', code_dtype=code_dtype, optimizer_dtype=optimizer_dtype)
                else:
                    if 'scene_id' not in self.cache[scene_name_single]:
                        self.cache[scene_name_single]['scene_id'] = out['scene_id']
                    if 'code_' in self.cache[scene_name_single]['param']:
                        del self.cache[scene_name_single]['param']['code_']
                    for key, val in out['param'].items():
                        self.load_tensor_to_dict(self.cache[scene_name_single]['param'], key, val,
                                                 device='cpu', dtype=code_dtype)
                    if 'optimizer' in self.cache[scene_name_single]:
                        self.optimizer_state_copy(out['optimizer'], self.cache[scene_name_single]['optimizer'],
                                                 device='cpu', dtype=optimizer_dtype)
                    else:
                        self.cache[scene_name_single]['optimizer'] = self.optimizer_state_to(
                            out['optimizer'], device='cpu', dtype=optimizer_dtype)


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

    def inverse_code(self, gt_imgs, grids, code_, code_optimizer_states, prior_grad, cfg):
        n_inverse_steps = cfg['inverse_steps']

        mlps = [generate_mlp_from_weights(code_single, self.mlp_kwargs) for code_single in code_]
        code_optimizers = self.build_optimizer(mlps, cfg)
        for sidx, state in enumerate(code_optimizer_states):
            if state is not None:
                optim = code_optimizers[sidx].state_dict()
                optim['state'] = state
                code_optimizers[sidx].load_state_dict(optim)

        for code_optimizer in code_optimizers:
            code_optimizer.zero_grad()

        for inverse_step_id in range(n_inverse_steps):
            mse_loss = []
            for code_idx, code_single in enumerate(code_):
                #if code_idx == 2:
                #   print(code_single)
                mlp = mlps[code_idx]
                #mlp_params = [param for name, param in mlp.named_parameters()]
                input = grids[code_idx].unsqueeze(0)
                output = mlp({'coords': input})

                loss_inner = image_mse(mask=None, model_output=output, gt=gt_imgs[code_idx].unsqueeze(0))
                mse_loss.append(loss_inner['img_loss'])
            mse_loss = torch.mean(torch.hstack(mse_loss))
            joint_parameters = []
            for model in mlps:
                ps = list(model.parameters())
                joint_parameters += ps
                num_parameters = len(ps)
            start = time.time()
            grad_inner = torch.autograd.grad(mse_loss,
                                             joint_parameters,
                                             create_graph=False)
            current_idx = 0
            code_idx = 0
            for ii, (grad, param) in enumerate(zip(grad_inner, joint_parameters)):
                grad_shape = grad.shape
                num_params = np.product(list(grad.shape))
                grad = grad.view(-1)
                grad = grad + prior_grad[code_idx][current_idx:current_idx + num_params]
                grad = grad.view(grad_shape)
                param.grad = torch.zeros_like(param)
                current_idx += num_params
                param.grad.copy_(grad)

                if ((ii + 1) % num_parameters) == 0:
                    code_idx += 1
                    current_idx = 0

            for code_optim in code_optimizers:
                code_optim.step()
            end = time.time()
            print(f"one step took {round(end - start, 3)} seconds")
        for idx, mlp in enumerate(mlps):
            state_dict = mlp.state_dict()
            weights = []
            for weight in state_dict:
                weights.append(state_dict[weight].flatten())
            code_[idx] = torch.hstack(weights)
            optim_state = code_optimizers[idx].state_dict()
            del optim_state['param_groups']
            code_optimizer_states[idx] = code_optimizers[idx].state_dict()

        return mse_loss

    def inverse_code_1b1(self, gt_imgs, grids, code_, code_optimizer_states, prior_grad, cfg):
        n_inverse_steps = cfg['inverse_steps']

        mlps = [generate_mlp_from_weights(code_single, self.mlp_kwargs) for code_single in code_]
        code_optimizers = self.build_optimizer(mlps, cfg)
        for sidx, state in enumerate(code_optimizer_states):
            if state is not None:
                optim = code_optimizers[sidx].state_dict()
                optim['state'] = state
                code_optimizers[sidx].load_state_dict(optim)

        for code_optimizer in code_optimizers:
            code_optimizer.zero_grad()

        mse_loss = []
        for inverse_step_id in range(n_inverse_steps):
            for code_idx, code_single in enumerate(code_):
                #if code_idx == 2:
                #   print(code_single)
                mlp = mlps[code_idx]
                #mlp_params = [param for name, param in mlp.named_parameters()]
                input = grids[code_idx].unsqueeze(0)
                output = mlp({'coords': input})
                start = time.time()
                loss_inner = image_mse(mask=None, model_output=output, gt=gt_imgs[code_idx].unsqueeze(0))['img_loss']
                grad_inner = torch.autograd.grad(loss_inner,
                                                 list(mlp.parameters()),
                                                 create_graph=False)

                end_grad = time.time()
                print(f"grad inner step took {round(end_grad - start, 3)} seconds")
                mse_loss.append(loss_inner)

                current_idx = 0
                for grad, param in zip(grad_inner, mlp.parameters()):
                    grad_shape = grad.shape
                    num_params = np.product(list(grad.shape))
                    grad = grad.view(-1)
                    grad = grad + prior_grad[code_idx][current_idx:current_idx + num_params]
                    grad = grad.view(grad_shape)
                    param.grad = torch.zeros_like(param)
                    current_idx += num_params
                    param.grad.copy_(grad)

                for code_optim in code_optimizers:
                    code_optim.step()
                end = time.time()
                print(f"one step took {round(end - start, 3)} seconds")
        for idx, mlp in enumerate(mlps):
            state_dict = mlp.state_dict()
            weights = []
            for weight in state_dict:
                weights.append(state_dict[weight].flatten())
            code_[idx] = torch.hstack(weights)
            optim_state = code_optimizers[idx].state_dict()
            del optim_state['param_groups']
            code_optimizer_states[idx] = code_optimizers[idx].state_dict()

        mse_loss = torch.mean(torch.hstack(mse_loss))
        return mse_loss


    def training_step(self, train_batch, optimizer, global_step):
        # Extract input_data (either voxel or weight) which is the first element of the tuple
        input_img = train_batch['gt_img'][0].view(64, 64, 3).permute(2, 0, 1).cuda()

        log_interval = int(Config.get("log_interval"))

        if 'code_optimizer' in self.cfg:
            code_list_, code_optimizers = self.load_cache(train_batch)
            code = torch.stack(code_list_, dim=0).cuda()

        optimizer.zero_grad()
        # Sample a diffusion timestep
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(code.shape[0],))
            .long()
            .cuda()
        )

        # Execute a diffusion forward pass
        loss_terms = self.diff.training_losses(
            self.model,
            code * self.cfg.normalization_factor,
            t,
            self.mlp_kwargs,
            self.logger,
            model_kwargs=None,
        )

        loss_mse = loss_terms["loss"].mean()

        loss_mse.backward()  # Backpropagation
        optimizer.step()
        #print(code)
        prior_grad = [code_.grad.data.clone() for code_ in code_list_]

        #print('before inverse code')
        #start = time.time()
        inv_loss = self.inverse_code_1b1(train_batch['gt_img'], train_batch['coords'], code_list_, code_optimizers, prior_grad, self.cfg)
        # At the first step output first element in the dataset as a sanit check
        if "hyper" in self.method and global_step % 50 == 0 and global_step % log_interval == 0:
            mlp = generate_mlp_from_weights(code_list_[0], self.mlp_kwargs)
            #model_input = {'coords': model_input}
            input = train_batch['coords'][0].unsqueeze(0)
            inr_output = mlp({'coords': input})['model_out'][0].view(64, 64, 3).permute(2, 0, 1)
            #print(inr_output.shape)

            images = wandb.Image(input_img, caption="")
            inr_images = wandb.Image(inr_output, caption="")
            # wandb.log({"examples": images})
            self.logger.log({"global_step": global_step / log_interval, "gt": images})
            self.logger.log({"global_step": global_step / log_interval, "inr": inr_images})


        # ==== save cache ====
        self.save_cache(code_list_, code_optimizers, train_batch['scene_id'])
        self.logger.log({"global_step": global_step, "diff_train_loss": loss_mse})
        self.logger.log({"global_step": global_step, "inr_train_loss": inv_loss})

        return loss_mse

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
