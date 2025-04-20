#!/bin/bash

#SBATCH --job-name=train
#SBATCH -n 4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=training_log_%j.out

source ~/.bashrc

workon dslab
cd dslab-project4/src
python scripts/train.py
