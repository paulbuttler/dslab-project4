# SynthMoCap Datasets

The following datasets are released for **non-commercial** use, refer to [LICENSE](LICENSE) for more details.

In all datasets samples are indexed by identity and frame - there are approximately 20,000 identities in each dataset
with 5 frames each. Indices are formatted with leading zeros, for example `img_0000123_004.jpg` for identity 123, frame 4.

Some pose data is sourced from the [AMASS](https://amass.is.tue.mpg.de/) and [MANO](https://mano.is.tue.mpg.de/)
datasets and is not directly redistributed by us. This data will be downloaded and spliced into the full dataset as part
of the `download_data.py` script. You therefore need valid logins to https://amass.is.tue.mpg.de/ and
https://mano.is.tue.mpg.de/ which you will be prompted for when running the script.

Once downloaded, you can use `python visualize_data.py [path_to_dataset]` to visualize the data
including some ground-truth annotations.

First setup your environment by running `pip install -r requirements.txt` using python 3.10 and installing
`wget` on your system if it isn't already. Our server requires TLSv1.2 which some old versions of wget do not support.
We have successfully tested 1.21.4 of this [this build](https://eternallybored.org/misc/wget/) on windows.

## SynthBody

![body_data](img/body_data.jpg)

### Download Instructions

The following command will download the dataset to `YOUR_DATA_DIRECTORY/synth_body/`:

```bash
python download_data.py --dataset body --output-dir YOUR_DATA_DIRECTORY/
```

If you want just a single identity (500KB) you can add the `--single_id` flag, or for a single chunk (380MB) add `--single_chunk`.
The total size of the dataset is approximately 10GB.

### Contents

#### Image Data

| Data Type | File Name |
|-|-|
|RGB image| `img_XXXXXXX_XXX.jpg`|
| Grayscale beard segmentation| `segm_beard_XXXXXXX_XXX.png`|
| Grayscale eyebrows segmentation| `segm_eyebrows_XXXXXXX_XXX.png`|
| Grayscale eyelashes segmentation| `segm_eyelashes_XXXXXXX_XXX.png`|
| Grayscale facewear segmentation| `segm_facewear_XXXXXXX_XXX.png`|
| Grayscale glasses segmentation| `segm_glasses_XXXXXXX_XXX.png`|
| Grayscale head hair segmentation| `segm_hair_XXXXXXX_XXX.png`|
| Grayscale headwear segmentation| `segm_headwear_XXXXXXX_XXX.png`|
| Integer body parts segmentation| `segm_parts_XXXXXXX_XXX.png`|

#### Segmentation parts indices

|Class|Index|
|-|-|
|BACKGROUND|0|
|FACE|1|
|LEFT_UPPER_TORSO|2|
|LEFT_LOWER_TORSO|3|
|RIGHT_UPPER_TORSO|4|
|RIGHT_LOWER_TORSO|5|
|LEFT_UPPER_LEG|6|
|LEFT_LOWER_LEG|7|
|LEFT_FOOT|8|
|RIGHT_UPPER_LEG|9|
|RIGHT_LOWER_LEG|10|
|RIGHT_FOOT|11|
|LEFT_UPPER_ARM|12|
|LEFT_LOWER_ARM|13|
|LEFT_HAND|14|
|RIGHT_UPPER_ARM|15|
|RIGHT_LOWER_ARM|16|
|RIGHT_HAND|17|

#### Metadata

```json
{
    "camera": {
        "world_to_camera": [ "4x4 array of camera extrinsics" ],
        "camera_to_image": [ "3x3 array of camera intrinsics" ],
        "resolution": [
            512,
            512
        ]
    },
    "pose": [ " 52x3 array of SMPL-H pose parameters" ],
    "translation": [ "3 element array for SMPL-H translation" ],
    "body_identity": [ "16 element array of neutral SMPL-H shape parameters" ],
    "landmarks": {
        "3D_world": [ "52x3 array of 3D landmarks in world-space corresponding to SMPL-H joints" ],
        "3D_cam": [ "52x3 array of 3D landmarks in camera-space corresponding to SMPL-H joints" ],
        "2D": [ "52x2 array of 2D landmarks in image-space corresponding to SMPL-H joints" ]
    }
}
```

#### Landmarks

![body landmark definition](img/body_ldmks.png)

#### Notes

The dataset includes some images with secondary "distractor" people in the background, the ground-truth data does
not include annotations for these people, only the primary person. These images help with robustness to occlusions and
cases where people are close together in real-world scenarios.

As detailed in the paper, clothing is modeled using displacement maps. Segmentation ground-truth includes the effect
of these displacements, but landmarks are not displaced and instead lie directly on the surface of the body mesh.

## SynthFace

![face_data](img/face_data.jpg)

### Download Instructions

The following command will download the dataset to `YOUR_DATA_DIRECTORY/synth_face/`:

```bash
python download_data.py --dataset face --output-dir /YOUR_DATA_DIRECTORY/
```

If you want just a single identity (500KB) you can add the `--single_id` flag, or for a single chunk (500MB) add `--single_chunk`.
The total size of the dataset is approximately 11GB.

### Contents

#### Image Data

| Data Type | File Name |
|-|-|
|RGB image|`img_XXXXXXX_XXX.jpg`|
|Grayscale beard segmentation|`segm_beard_XXXXXXX_XXX.png`|
|Grayscale clothing segmentation|`segm_clothing_XXXXXXX_XXX.png`|
|Grayscale eyebrows segmentation|`segm_eyebrows_XXXXXXX_XXX.png`|
|Grayscale eyelashes segmentation|`segm_eyelashes_XXXXXXX_XXX.png`|
|Grayscale facewear segmentation|`segm_facewear_XXXXXXX_XXX.png`|
|Grayscale glasses segmentation|`segm_glasses_XXXXXXX_XXX.png`|
|Grayscale head hair segmentation|`segm_hair_XXXXXXX_XXX.png`|
|Grayscale headwear segmentation|`segm_headwear_XXXXXXX_XXX.png`|
|Integer face parts segmentation|`segm_parts_XXXXXXX_XXX.png`|

#### Segmentation parts indices

|Class|Index|
|-|-|
|BACKGROUND|0|
|SKIN|1|
|NOSE|2|
|RIGHT_EYE|3|
|LEFT_EYE|4|
|RIGHT_BROW|5|
|LEFT_BROW|6|
|RIGHT_EAR|7|
|LEFT_EAR|8|
|MOUTH_INTERIOR|9|
|TOP_LIP|10|
|BOTTOM_LIP|11|
|NECK|12|

#### Metadata

```json
{
    "camera": {
        "world_to_camera": [ "4x4 array of camera extrinsics" ],
        "camera_to_image": [ "3x3 array of camera intrinsics" ],
        "resolution": [
            512,
            512
        ]
    },
    "head_pose": [ "3x3 rotation matrix of the head" ],
    "left_eye_pose": [ "3x3 rotation matrix of the left eye"],
    "right_eye_pose": [ "3x3 rotation matrix of the right eye" ],
    "landmarks": {
        "2D": [ "70x2 array of landmarks in image space" ]
    }
}
```

#### Landmarks

![face landmark definition](img/face_ldmks.png)

#### Notes

The dataset includes some images with secondary "distractor" faces in the background, the ground-truth data does
not include annotations for these faces, only the primary face. These images help with robustness to occlusions and
cases where faces are close together in real-world scenarios.

## SynthHand

![hand_data](img/hand_data.jpg)

### Download Instructions

The following command will download the dataset to `YOUR_DATA_DIRECTORY/synth_hand/`:

```bash
python download_data.py --dataset hand --output-dir /YOUR_DATA_DIRECTORY/
```

If you want just a single identity (250KB) you can add the `--single_id` flag, or for a single chunk (300MB) add `--single_chunk`.
The total size of the dataset is approximately 7GB.

### Contents

#### Image Data

| Data Type | File Name |
|-|-|
|RGB image|`img_XXXXXXX_XXX.jpg`|

#### Metadata

```json
{
    "camera": {
        "world_to_camera": [ "4x4 array of camera extrinsics" ],
        "camera_to_image": [ "3x3 array of camera intrinsics" ],
        "resolution": [
            512,
            512
        ]
    },
    "pose": [ " 52x3 array of SMPL-H pose parameters" ],
    "translation": [ "3 element array for SMPL-H translation" ],
    "body_identity": [ "16 element array of neutral SMPL-H shape parameters" ],
    "landmarks": {
        "3D_world": [ "21x3 array of 3D landmarks in world-space - first 15 elements are MANO joints, last 5 are finger tips" ],
        "3D_cam": [ "21x3 array of 3D landmarks in camera-space - first 15 elements are MANO joints, last 5 are finger tips" ],
        "2D": [ "21x2 array of 2D landmarks in image-space - first 15 elements are MANO joints, last 5 are finger tips" ]
    }
}
```

#### Landmarks

![hand landmark definition](img/hand_ldmks.png)

#### Notes

Our parametric body model uses a 300 component SMPL-H shape basis and adds the MANO shape basis to the hands, as well as
incorporating skin displacement maps. The reposed SMPL-H meshes therefore do not exactly match the rendered images, this
difference is only significant for some hand images.
