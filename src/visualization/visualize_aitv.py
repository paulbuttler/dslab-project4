import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import cv2
import numpy as np
from src import utils
from pathlib import Path
from aitviewer.models.smpl import SMPLLayer  # type: ignore
from aitviewer.renderables.smpl import SMPLSequence  # type: ignore
from aitviewer.renderables.billboard import Billboard  # type: ignore
from aitviewer.scene.camera import OpenCVCamera  # type: ignore
from aitviewer.renderables.spheres import Spheres  # type: ignore
from aitviewer.viewer import Viewer  # type: ignore


LDMK_CONN = {
    "body": [
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 1],
        [5, 2],
        [6, 3],
        [7, 4],
        [8, 5],
        [9, 6],
        [10, 7],
        [11, 8],
        [12, 9],
        [13, 9],
        [14, 9],
        [15, 12],
        [16, 13],
        [17, 14],
        [18, 16],
        [19, 17],
        [20, 18],
        [21, 19],
        [22, 20],
        [23, 22],
        [24, 23],
        [25, 20],
        [26, 25],
        [27, 26],
        [28, 20],
        [29, 28],
        [30, 29],
        [31, 20],
        [32, 31],
        [33, 32],
        [34, 20],
        [35, 34],
        [36, 35],
        [37, 21],
        [38, 37],
        [39, 38],
        [40, 21],
        [41, 40],
        [42, 41],
        [43, 21],
        [44, 43],
        [45, 44],
        [46, 21],
        [47, 46],
        [48, 47],
        [49, 21],
        [50, 49],
        [51, 50],
    ], "hand": [
        [1, 0],
        [2, 1],
        [3, 2],
        [4, 0],
        [5, 4],
        [6, 5],
        [7, 0],
        [8, 7],
        [9, 8],
        [10, 0],
        [11, 10],
        [12, 11],
        [13, 0],
        [14, 13],
        [15, 14],
        [16, 3],
        [17, 6],
        [18, 9],
        [19, 12],
        [20, 15],
    ],
}

def draw_func(kp2d):
    """Returns a function that overlays 2D landmarks onto the Billboard image."""

    def draw_2d_kp(img, frame=0):
        """Modifies the image to include 2D keypoints."""
        current_kp2d = kp2d.copy()

        # Draw landmarks using provided function by the paper authors.
        utils.draw_landmarks(img, current_kp2d, LDMK_CONN["body"])

        return img

    return draw_2d_kp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="data/synth_body")
    parser.add_argument("--sidx", type=int, default=0)
    parser.add_argument("--fidx", type=int, default=0)
    args = parser.parse_args()

    meta_file = args.data_dir / f"metadata_{args.sidx:07d}_{args.fidx:03d}.json"
    img_file = args.data_dir / f"img_{args.sidx:07d}_{args.fidx:03d}.jpg"

    with open(meta_file, "r") as f:
        metadata = json.load(f)

    # Convert json metadata to NumPy arrays.
    ldmks_2d = np.asarray(metadata["landmarks"]["2D"])
    ldmks_3d_world = np.asarray(metadata["landmarks"]["3D_world"])
    ldmks_3d_camera = np.asarray(metadata["landmarks"]["3D_cam"])
    body_identity = np.asarray(metadata["body_identity"])
    pose = np.asarray(metadata["pose"])
    translation = np.asarray(metadata["translation"])
    world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
    camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])

    print("ldmks_2d:", ldmks_2d.shape)
    print("ldmks_3d_world:", ldmks_3d_world.shape)
    print("body_identity:", body_identity.shape)
    print("pose:", pose.shape)
    print("translation:", translation.shape)
    print("world_to_camera:", world_to_camera.shape)
    print("camera_to_image:", camera_to_image.shape)

    # Extract pose and shape parameters.
    global_orient = pose[0].reshape(1, -1)
    body_pose = pose[1:22].reshape(1, -1)
    left_hand_pose = pose[22:37].reshape(1, -1)
    right_hand_pose = pose[37:].reshape(1, -1)
    body_shape = body_identity[:10].reshape(1, -1)

    # Create a SMPL sequence.
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral")
    smpl_seq = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=body_pose,
        betas=body_shape,
        poses_root=global_orient,
        poses_left_hand=left_hand_pose,
        poses_right_hand=right_hand_pose,
        trans=translation.reshape(1, -1),
    )

    # Load the input image.
    input_img = cv2.imread(img_file)
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    cols, rows = input_img.shape[1], input_img.shape[0]

    v = Viewer(size=(cols, rows))

    # Create an OpenCV camera.
    camera = OpenCVCamera(camera_to_image, world_to_camera[:3], cols, rows, viewer=v)

    # Load the reference image and create a Billboard.
    billboard = Billboard.from_camera_and_distance(camera, 5.0, cols, rows, [img_rgb], draw_func(ldmks_2d))

    # Display 3D landmarks.
    spheres = Spheres(ldmks_3d_world, name="Joints", radius=0.008, color=(1.0, 0.0, 1.0, 1.0))

    v.scene.add(billboard, spheres, smpl_seq, camera)
    v.set_temp_camera(camera)

    # Display the set of generated vertices for the SMPL-H model.
    vertex_indices = np.int64(np.load("src/visualization/vertices/body_vertices.npy"))

    # Extract the positions of the specified vertices
    vertex_positions = smpl_seq.vertices[:, vertex_indices] + smpl_seq.position[np.newaxis]
    print("vertex_positions:", vertex_positions.shape)

    vertices = Spheres(vertex_positions, name="Body_Vertices", radius=0.007, color=(0.0, 0.0, 1.0, 1.0))

    v.scene.add(vertices)

    # Viewer settings.
    v.scene.floor.enabled = False
    v.scene.origin.enabled = False
    v.shadows_enabled = False

    v.run()
