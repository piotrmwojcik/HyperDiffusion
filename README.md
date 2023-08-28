# HyperDiffusion
Official code repository of "HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion" @ ICCV 2023

[Paper](https://arxiv.org/abs/2303.17015) | [Project website](https://www.ziyaerkoc.com/hyperdiffusion) | [Data](https://drive.google.com/drive/folders/1CuNVa92jcKlGBiHEuCQK2-juAB6Q6QPx?usp=sharing)


## News
I'll release rest of the weights/checkpoints after post-refactor tests are complete. You can see here for what's uploaded.  
- [XX.09.2023] Code and airplane weights/checkpoints released
## Method Overview

![Overview image](/static/overview.svg)


## Code Map
### Directories
- **Pointnet_Pointnet2_pytorch**: Includes Pointnet2 definition and weights for 3D FID calculation.
- **configs**: Containing training and overfitting configs.
- **data**: Downloaded point cloud files including train-val-test splits go here (see [Get Started](#get-started)) 
- **diffusion**: Contains all the diffusion logic. Borrowed from [OpenAI](https://github.com/openai/guided-diffusion) .
- **ldm**: Latent diffusion codebase for Voxel baseline. Borrowed from [official LDM repo](https://github.com/CompVis/latent-diffusion).
- **mlp_weights**: Includes overfitted MLP weights should be downloaded to here (see [Get Started](#get-started)).
- **siren**: Modified [SIREN](https://github.com/vsitzmann/siren) codebase. Includes shape overfitting logic.
- **static**: Images for README file.
### Generated Directories
- **lightning_checkpoints**: This will be created once you start training for the first time. It will include checkpoints of the diffusion model, the sub-folder names will be the unique name assigned by the Weights & Biases in addition to timestamp.
- **outputs**: Hydra creates this folder to store the configs but we mainly send our outputs to Weights & Biases, so, it's not that special.
- **orig_meshes**: Here we put generated weights as .pth and sometimes generated meshes.
- **wandb**: Weights & Biases will create this folder to store outputs before sending them to server.
### Files
- **augment.py**: Including some augmentation methods, though we don't use them in the main paper.
- **dataset.py**: `WeightDataset` and `VoxelDataset` definitions which are `torch.Dataset` descendants. Former one is related to our HyperDiffusion method, while the latter one is for Voxel baseline.
- **embedder.py**: Positional encoding definition.
- **evaluation_metrics_3d.py**: Methods to calculate MMD, COV and 1-NN from [DPC](https://github.com/luost26/diffusion-point-cloud). Both for 3D and 4D.
- **hd_utils.py**: Many utility methods ranging from rendering to flattening MLP weights.
- **hyperdiffusion.py**: Definition of our method, it includes training, testing and validation logics in the form of a Pytorch Lightning module.
- **hyperdiffusion_env.yaml**: Conda environment file (see [Get Started](#get-started) section).
- **main.py**: Entry point of our codebase.
- **mlp_models.py**: Definition of ReLU MLPs with positional encoding.
- **torchmetrics_fid.py**: Modified torchmetrics fid implementation to calculate 3D-FID.
- **transformer.py**: GPT definition from [G.pt paper](https://github.com/wpeebles/G.pt).

## Data
All the data needed to train and evaluate HyperDiffusion is in [this Drive folder](https://drive.google.com/drive/folders/1CuNVa92jcKlGBiHEuCQK2-juAB6Q6QPx?usp=sharing).
There are three main folders there:
- **Checkpoints** contains trained diffusion model for each category, you'll need them for [evaluation](#evaluation)
- **MLP Weights** involves already overfitted MLP weights.
- **Point Clouds (2048)** has the set of 2048 points sampled from meshes to be used for metric calculation and baseline training.  

## Get Started

We have environment file that you can create a conda environment from. Simply run,

```commandline
conda env create --file hyperdiffusion_env.yaml
conda activate hyper-diffusion
```

_We specify our runtime parameters using .yaml files which are inside configs folder. There are different yaml files for each category and task._

Then, download **MLP Weights** from [our Drive](#data) and put it into **mlp_weights** folder. Config files assume that weights are in that folder.

For 3D, download **Point Clouds (2048)** folder from [Drive](#data) and save its content to **data** folder. Eventually, **data** folder should look like this:
```
data
|-- 02691156
|-- 02691156_2048_pc
|-- 02958343
|-- 02958343_2048_pc
|-- 03001627
|-- 03001627_2048_pc
|-- animals
```
_Note: Category id to name conversion is as follows: 02691156 -> airplane, 02958343 -> car, 03001627 -> chair_


### Training
To start training, airplane category:
```commandline
python main.py --config-name=train_plane
```
car category:
```commandline
python main.py --config-name=train_car
```
chair category:
```commandline
python main.py --config-name=train_chair
```

We are using [hydra](https://hydra.cc/), you can either specify them from corresponding yaml file or directly modify
the parameters from terminal. For instance, to change the number of epochs:

```commandline
python main.py --config-name=train_plane epochs=1
```
### Evaluation
Download **Checkpoints** folder from [Drive section](#data). Assign the path of that checkpoint to the `best_model_save_path` parameter.

to start evaluating, airplane category:
```commandline
python main.py --config-name=train_plane mode=test best_model_save_path=<path/to/checkpoint>
```
car category:
```commandline
python main.py --config-name=train_car mode=test best_model_save_path=<path/to/checkpoint>
```
chair category (we have special operations for chair, see our Supplementary Material for details):
```commandline
python main.py --config-name=train_chair mode=test best_model_save_path=<path/to/checkpoint> test_sample_mult=2 dedup=True
```

### Ovefitting
We already provide overfitted shapes but if you want to do it yourself. Make sure that you put downloaded [ShapeNet](https://shapenet.org/) shapes into **data** folder. Then, you can run the following:
```commandline
python siren/experiment_scripts/train_sdf.py --config-name=overfit_plane
```
## Training Plots

We share training plots for better reproducibility. Links take you to Weights & Biases reports.

[Plane](https://api.wandb.ai/links/ziyaer/9korb518) | [Car](https://api.wandb.ai/links/ziyaer2/s528ygbt) | [Chair](https://api.wandb.ai/links/ziyaer2/y9pbdzwh) | [4D Animals](https://api.wandb.ai/links/ziyaer2/2xzc3fcn)

## Dependencies

* Python 3.7
* PyTorch 1.13.0
* CUDA 11.7
* Weights & Biases (We heavily rely on it for visualization and monitoring)

For full list please see [hyperdiffusion_env.yaml file](/hyperdiffusion_env.yaml)

## Acknowledgment

We mainly used codebases of [SIREN](https://github.com/vsitzmann/siren), [G.pt](https://github.com/wpeebles/G.pt) papers to build our repository. We also referred to [DPC](https://github.com/luost26/diffusion-point-cloud) for codes like evaluation metrics. We used [OpenAI Guided Diffusion](https://github.com/openai/guided-diffusion) as our diffusion backbone. [LDM](https://github.com/CompVis/latent-diffusion) codebase was useful for us to implement our voxel baseline.

## Citation
If you find our work useful, please cite using the following BibTex entry:
```
@misc{erkoç2023hyperdiffusion,
  title={HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion}, 
  author={Ziya Erkoç and Fangchang Ma and Qi Shan and Matthias Nießner and Angela Dai},
  year={2023},
  eprint={2303.17015},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```