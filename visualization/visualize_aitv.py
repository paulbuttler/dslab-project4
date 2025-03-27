import argparse
import json
import cv2
from pathlib import Path
import numpy as np
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.billboard import Billboard
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.renderables.spheres import Spheres
from aitviewer.viewer import Viewer

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

    # Convert json metadata to NumPy arrays
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

    # Extract pose and shape parameters
    global_orient = pose[0].reshape(1, -1)
    body_pose = pose[1:22].reshape(1, -1)
    left_hand_pose = pose[22:37].reshape(1, -1)
    right_hand_pose = pose[37:].reshape(1, -1)
    body_shape = body_identity[:10].reshape(1, -1)

    # Create SMPL sequence
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

    # Load the input image
    input_img = cv2.imread(img_file)
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    cols, rows = input_img.shape[1], input_img.shape[0]

    v = Viewer(size=(cols, rows))

    # Create an OpenCV camera.
    camera = OpenCVCamera(camera_to_image, world_to_camera[:3], cols, rows, viewer=v)

    # Load the reference image and create a Billboard.
    pc = Billboard.from_camera_and_distance(camera, 5.0, cols, rows, [img_rgb])

    v.scene.add(pc, smpl_seq, camera)
    v.set_temp_camera(camera)
    v.run()
