#!/bin/bash

#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=3-00:00:00
#SBATCH --output=training_log_%j.out

source ~/.bashrc

workon dslab
cd src
python scripts/train.py
