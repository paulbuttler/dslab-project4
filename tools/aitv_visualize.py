import argparse
import json
import cv2
import utils
from pathlib import Path
import numpy as np
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--sidx", type=int, default=0)
    parser.add_argument("--fidx", type=int, default=0)
    args = parser.parse_args()

    meta_file = args.data_dir / f"metadata_{args.sidx:07d}_{args.fidx:03d}.json"
    img_file = args.data_dir / f"img_{args.sidx:07d}_{args.fidx:03d}.jpg"

    with open(meta_file, "r") as f:
        metadata = json.load(f)

    # Convert json metadata to NumPy arrays
    ldmks_2d = np.asarray(metadata["landmarks"]["2D"])
    body_identity = np.asarray(metadata["body_identity"])
    pose = np.asarray(metadata["pose"])
    translation = np.asarray(metadata["translation"])
    world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
    camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])

    print("ldmks_2d:", ldmks_2d.shape)
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
    smpl_layer = SMPLLayer(model_type="smplh", gender="male")
    smpl_seq = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=body_pose,
        betas=body_shape,
        poses_root=global_orient,
        poses_left_hand=left_hand_pose,
        poses_right_hand=right_hand_pose,
        trans=translation.reshape(1, -1),
    )

    # Display the image and SMPL sequence
    frame = cv2.imread(str(img_file))
    cv2.imshow("Image", frame)

    v = Viewer()
    v.scene.add(smpl_seq)

    utils.set_camera_from_extrinsics(v.scene.camera, world_to_camera)

    v.run()
