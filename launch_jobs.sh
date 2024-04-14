#!/bin/bash
#SBATCH --job-name=denoising_backmapping
#SBATCH --time=14-00:00:00
#SBATCH --partition=simatlab-gpu
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giovanni_michel.cherchi@uca.fr
#SBATCH --gres=gpu:a100:1


poetry run python src/train.py train=diffusion
