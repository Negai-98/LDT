# Latent Diffusion Transformer for Point Cloud Generation

This repository contains a PyTorch implementation of the paper:

## Introduction
Diffusion models have been successfully applied to point cloud generation tasks recently. The main notion is using a forward process to progressively add noises into point clouds and then use a reverse process to generate point clouds by denoising these noises. We propose a latent diffusion model based on Transformers for point cloud generation. Instead of directly building a diffusion process based on the original points, we first propose a latent compressor to convert original point clouds into a set of latent tokens before feeding them into diffusion models. Converting point clouds as latent tokens not only improves expressiveness but also exhibits better flexibility since they can adapt to various downstream tasks. We carefully design the latent compressor based on an attention-based auto-encoder architecture to capture global structures in point clouds. Then, we propose to use Transformers as the backbones of the latent diffusion module to maintain global structures. The powerful feature extraction ability of Transformers guarantees the high quality and smoothness of generated point clouds. Experiments show that our method achieves superior performance in both unconditional generation on ShapeNet and multi-modal point cloud completion on ShapeNet-ViPC.

## Dependencies
### Create conda environment with torch 1.13.1 and CUDA 11.6
conda env create -f environment.yml
conda activate LDT

### Download the lib
Please Download the "extern" and place this folder in the root directory: [link](https://drive.google.com/drive/folders/1FRRKDBFNQTW_HdDglNro4ufJVPsD8zcz?usp=drive_link)

### Compile the evaluation metrics and pointnet2 lib

cd evaluation/pytorch_structural_losses/

python setup.py install

cd extern/pointnet2_ops_lib

python setup.py install

cd extern/emd

python setup.py install

## Dataset

Please follow the instruction from PointFlow to set-up the dataset ShapeNetCore.v2.PC15k for generation: [link](https://github.com/stevenygd/PointFlow). And place ShapeNetCore.v2.PC15k in data/

Please Download the "ViPC" dataset split files and place this folder in the data/datasets: [link](https://drive.google.com/drive/folders/1FRRKDBFNQTW_HdDglNro4ufJVPsD8zcz?usp=drive_link)

Please follow the instruction from ViPC to set-up the dataset ShapeNet-ViPc for completion: [link](https://github.com/Hydrogenion/ViPC). 
And place ShapeNetViPC-Dataset in data/

## Samples
Please Download the "result" and place this folder in the root directory: [link](https://drive.google.com/drive/folders/1FRRKDBFNQTW_HdDglNro4ufJVPsD8zcz?usp=drive_link)
Generated samples are available in test/smp

### Evaluate these samples
#### python val_sample.py --dataset<dataset_type> --sample <samples_filename>

python val_sample.py --dataset airplane --smp.npy  

python val_sample.py --dataset car --smp.npy  

python val_sample.py --dataset chair --smp.npy  

## Training
Only Single GPU Training supported

We provide the best result config in experiments/...Triner/config.yaml
#### Usage:
#### First stage training
python train_Compressor.py <config>
#### Second stage training (modify the pretrain path in config.yaml)
python train_Latent_Diffusion.py <config>
#### Hybrid training (modify the pretrain path in config.yaml)
python train_Hybrid.py <config>
