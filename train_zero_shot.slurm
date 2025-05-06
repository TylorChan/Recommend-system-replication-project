#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=20g
#SBATCH --tmp=20g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7751@umn.edu
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1


export PATH =/users/9/chen7751/miniconda3/envs/myenv:$PATH


python inference_knnlm_zero_shot.py