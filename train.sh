#!/bin/bash 
#SBATCH -N 1 -c 2
#SBATCH --mem=128G
#SBATCH --time=72:00:00 ## this requests one day of walltime
#SBATCH --gres=gpu:1
#SBATCH -e
#SBATCH -o

module load python/3.9.2_torch_gpu
cd your_env

python3 ./train_ae_128_cond_is.py
# python3 ./train_transformer_cond.py
