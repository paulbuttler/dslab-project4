# Data Science Lab: Developing Next-Gen 3D Human Pose Estimators

This repository implements the method presented in the paper "Look Ma, no markers: holistic performance capture without the hassle" using synthetic data provided by the authors.

## Setup
After cloning this repository and navigating to its root folder, create a virtual environment and install all required packages:
```
conda create --name dslab python=3.10
conda activate dslab
pip install -r requirements.txt
```

## Datasets
Download the SynthBody, -Hand and -Face datasets from [here](https://github.com/microsoft/SynthMoCap/blob/main/DATASETS.md) and place them in the `data/` directory.

