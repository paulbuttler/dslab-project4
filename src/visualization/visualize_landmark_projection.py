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



def draw_func(img, frame=0):
    img = img.copy()
    utils.draw_landmarks(
        img,
        verts_2d,
        connectivity=[],
        thickness=1,
        color=(255, 255, 255)
    )
    return img



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
    input_img = cv2.imread(str(img_file))
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    ########
    cv2.imwrite("tools/results/projected_landmark_overlay.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    cols, rows = input_img.shape[1], input_img.shape[0]

    v = Viewer(size=(cols, rows))

    # Create an OpenCV camera.
    camera = OpenCVCamera(camera_to_image, world_to_camera[:3], cols, rows, viewer=v)

    # Load the set of generated vertex indices (e.g., body vertices)
    vertex_indices = np.int64(np.load("src/visualization/vertices/body36_vertices.npy"))

    # # Uncomment HERE to combine 3 files
    # body_indices = np.load("src/visualization/vertices/body_vertices.npy")
    # hand_indices = np.load("src/visualization/vertices/hand_vertices.npy")
    # head_indices = np.load("src/visualization/vertices/head_vertices.npy")

    # # Concatenate and remove duplicates
    # vertex_indices = np.unique(np.concatenate([body_indices, hand_indices, head_indices])).astype(np.int64)


    # Extract vertex positions from the SMPL mesh
    vertex_positions = smpl_seq.vertices[:, vertex_indices] + smpl_seq.position[np.newaxis]  # (1, N, 3)
    vertex_positions = vertex_positions[0]  # (N, 3)

    # === Project 3D to 2D (make verts_2d global) ===
    verts_cam = (world_to_camera[:3, :3] @ vertex_positions.T + world_to_camera[:3, 3:4]).T  # (N, 3)
    verts_2d_homo = (camera_to_image @ verts_cam.T).T
    verts_2d = verts_2d_homo[:, :2] / verts_2d_homo[:, 2:]

    # === Save for reuse ===
    output_path = Path("tools/results/projected_2d_landmarks.npy")
    np.save(output_path, verts_2d)


    billboard = Billboard.from_camera_and_distance(
        camera, 5.0, cols, rows, [img_rgb], draw_func
    )


    # Visualize the 3D points
    vertices = Spheres(vertex_positions, name="Body36_Vertices", radius=0.007, color=(0.0, 0.0, 1.0, 1.0))


    # Add to viewer
    v.scene.add(billboard, smpl_seq, vertices, camera)
    v.set_temp_camera(camera)

    # Viewer settings.
    v.scene.floor.enabled = False
    v.scene.origin.enabled = False
    v.shadows_enabled = False


    v.run()
