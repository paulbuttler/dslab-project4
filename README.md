# Look Ma, No Markers – Implementation for 3D Body and Hand Pose Estimation

This repository implements hand and body pose estimation as presented in the paper [Look Ma, no markers](https://microsoft.github.io/SynthMoCap/), using synthetic [data](https://github.com/microsoft/SynthMoCap) provided by the authors.

We use the extended SMPL+H body model from [MANO](https://mano.is.tue.mpg.de/) as a substitute for the SOMA model used in the paper, which is not publicly available.

## Setup
After cloning this repository and navigating to its root folder, create a virtual environment and install all required packages:
```
conda create --name dslab python=3.10
conda activate dslab
pip install -r requirements.txt
```

Note: Developer versions of the smplx and aitviewer packages are required for full functionality. In particular, edit `aitviewer/models/smpl.py` and uncomment lines 39–40 to use the neutral SMPL+H model.

Ensure the `src` directory is added to your Python path.

### Datasets
The `tools` directory contains scripts provided by the authors of the paper. See [DATASETS.md](tools/DATASETS.md) for detailed instructions.

Download the SynthBody and SynthHand datasets into `data/raw` using the script provided in `tools/download`.

To generate 2d landmarks used for training the body and hand DNNs, run:
```bash
python tools/preprocess/generate_ldmks.py
```
Repeat with the `--hand` flag to generate hand landmarks.
```bash
python tools/preprocess/generate_ldmks.py --hand
```

This process can take a significant amount of time to complete.

### Parametric Human Body Model
Download the MANO and extended SMPL+H models from the [MANO download page](https://mano.is.tue.mpg.de/download.php). 

Follow the instructions [here](https://github.com/vchoutas/smplx/blob/main/tools/README.md) to merge the neutral SMPL+H model used in AMASS with MANO parameters. Create the folder `src/models/smplx/params/smplh/` and place the created `SMPLH_NEUTRAL.pkl` file inside.

## Visualization

### 2D Visualization

The `tools/visualize` directory contains the original 2D visualization script provided by the paper authors.
This can be used to inspect the dataset's annotations directly in image space.  

### 3D Visualization

To interactively explore the synthetic datasets in 3D, use our custom script based on the aitviewer:

```bash
python src/visualization/visualize_aitv.py
```
This script visualizes our generated dense 3D landmarks, which are projected into image space for training the body and hand networks on dense 2D landmarks.

Optional flags include

- `--joints`    Visualize the original 2D/3D joint and landmark annotations
- `--hand`      Visualize samples from the hand dataset
- `--sidx`      Specify a subject (body identity) index to display
- `--fidx`      Specify a frame index for visualization

## Training
For training, a subset of the earlier-installed dependencies is sufficient.
We recommend creating a new environment using the dependencies listed in `requirements_euler.txt`.

Navigate to the `src` directory and set training parameters in `training/body.yaml` or `training/hand.yaml` respectively. 
Parameters documented in the paper are marked with comments; the rest may require further tuning.

To start training the body model:
```
python training/train.py
```
To train the hand model include the flag `--hand`.

## Acknowledgments
We thank the authors of [Look Ma, no markers](https://microsoft.github.io/SynthMoCap/) for providing synthetic datasets and code that supports this implementation.