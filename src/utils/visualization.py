import os
import cv2
import json
import torch
import numpy as np
from models.smplx import SMPLHLayer
from aitviewer.models.smpl import SMPLLayer  # type: ignore
from aitviewer.renderables.smpl import SMPLSequence  # type: ignore
from aitviewer.renderables.billboard import Billboard  # type: ignore
from aitviewer.scene.camera import OpenCVCamera  # type: ignore
from aitviewer.viewer import Viewer  # type: ignore


def test_smpl_layer(pose_rot, pose_aa, shape, translation, gt_joints=None, part="body"):

    smplh_layer = SMPLHLayer(
        model_path="src/models/smplx/params/smplh",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=16,
    )

    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=16)
    B = pose_rot.shape[0]
    smpl_output = smplh_layer(
        betas=shape,
        global_orient=pose_rot[:, 0],
        body_pose=pose_rot[:, 1:22],
        left_hand_pose=pose_rot[:, 22:37],
        right_hand_pose=pose_rot[:, 37:],
        transl=translation,
        return_verts=False,
        return_full_pose=False,
    )
    smplh_joints = smpl_output.joints
    vertices, joints = smpl_layer.fk(
        betas=shape.reshape(B, -1),
        poses_root=pose_aa[:, 0].reshape(B, -1),
        poses_body=pose_aa[:, 1:22].reshape(B, -1),
        poses_left_hand=pose_aa[:, 22:37].reshape(B, -1),
        poses_right_hand=pose_aa[:, 37:].reshape(B, -1),
        trans=translation.reshape(B, -1),
    )

    if torch.allclose(joints[0, :52], smplh_joints[0, :52]):
        print("SMPLH layer used for training matches smpl_layer from aitviewer")
    else:
        print("SMPLH layer used for training does not match smpl_layer from aitviewer")

    if gt_joints is not None:
        if part == "body" or part == "full":
            diff = np.linalg.norm(gt_joints - smplh_joints[0, :52], axis=1)
            print(f"Mean distance of 3d gt body joints used for training to actual gt-joints: {diff[:22].mean()}")
            print(f"Mean distance of 3d gt hand joints used for training to actual gt-joints: {diff[22:].mean()}")
        elif part == "hand":
            diff = np.linalg.norm(gt_joints[:15] - smplh_joints[0, 22:37], axis=1)
            print(f"Mean distance of 3d gt hand joints used for training to actual gt-joints: {diff.mean()}")


def visualize_pose_and_shape(
    uids,
    gt_pose,
    gt_shape,
    pred_pose,
    pred_shape,
    translation,
    part="body",
    dataset="synth",
    billboard=True,
):

    print(f"\nVisualizing {part} pose" + " and shape" * (part != "hand"))
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=pred_shape.shape[1])

    gt_shape = gt_shape[:, : pred_shape.shape[1]]

    for idx in np.random.choice(len(uids), size=2, replace=False):
        if dataset == "synth":
            if part == "body" or part == "full":
                img_file = f"data/synth_body/img_{uids[idx]}.jpg"
                meta_file = f"data/synth_body/metadata_{uids[idx]}.json"
            elif part == "hand":
                img_file = f"data/synth_hand/img_{uids[idx]}.jpg"
                meta_file = f"data/synth_hand/metadata_{uids[idx]}.json"

            if billboard:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)

                # Get camera data
                world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
                camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])

        elif dataset == "ehf":
            img_file = os.path.join("data/EHF", f"{uids[idx]:02d}_img.jpg")

            if billboard:
                # Get camera data
                camera_to_image = np.array(
                    [[1498.22426237, 0.0, 790.263706], [0.0, 1498.22426237, 578.90334], [0.0, 0.0, 1.0]]
                )
                rvec = np.array([[-2.98747896], [0.01172457], [-0.05704687]])
                tvec = np.array([[-0.03609917], [0.43416458], [2.37101226]])
                R, _ = cv2.Rodrigues(rvec)
                world_to_camera = np.hstack((R, tvec))

        else:
            raise ValueError("Dataset not supported")

        g_pose = gt_pose[idx]  # [52, 3]
        pose = pred_pose[idx]  # [52, 3]
        g_shape = gt_shape[idx]  # [16]
        shape = pred_shape[idx]  # [16]
        transl = translation[idx]  # [3]

        cat_pose = torch.stack([g_pose, pose], dim=0)
        cat_shape = torch.stack([g_shape, shape], dim=0)
        cat_transl = transl.unsqueeze(0).repeat(2, 1)

        smpl_seq = SMPLSequence(
            smpl_layer=smpl_layer,
            betas=cat_shape,
            poses_root=cat_pose[:, 0].reshape(2, -1),
            poses_body=cat_pose[:, 1:22].reshape(2, -1),
            poses_left_hand=cat_pose[:, 22:37].reshape(2, -1),
            poses_right_hand=cat_pose[:, 37:].reshape(2, -1),
            trans=cat_transl,
        )

        input_img = cv2.imread(img_file)
        img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        cols, rows = input_img.shape[1], input_img.shape[0]

        v = Viewer(size=(cols, rows))

        if billboard:
            camera = OpenCVCamera(camera_to_image, world_to_camera[:3], cols, rows, viewer=v)
            billboard = Billboard.from_camera_and_distance(camera, 5.0, cols, rows, [img_rgb])
            v.scene.add(billboard, camera)
            v.set_temp_camera(camera)

        v.scene.add(smpl_seq)
        v.scene.floor.enabled = False
        v.scene.origin.enabled = False
        v.shadows_enabled = False
        v.run()


def visualize_prediction(images, pose, shape, cam_int, cam_ext):
    """
    Visualizes the predicted pose and shape in 3D using aitviewer.

    Args:
        images (torch.Tensor): Batch of input images.
        pose (torch.Tensor): Predicted pose parameters.
        shape (torch.Tensor): Predicted shape parameters.
        cam_int (torch.Tensor): Camera intrinsics.
        cam_ext (torch.Tensor): Camera extrinsics.
    """
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=shape.shape[1])
    B = pose.shape[0]

    images = images.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
    images = (images * 255).astype(np.uint8)

    for idx in range(B):
        smpl_seq = SMPLSequence(
            smpl_layer=smpl_layer,
            betas=shape[idx],
            poses_root=pose[idx, 0].reshape(1, -1),
            poses_body=pose[idx, 1:22].reshape(1, -1),
            poses_left_hand=pose[idx, 22:37].reshape(1, -1),
            poses_right_hand=pose[idx, 37:].reshape(1, -1),
        )

        img_np = images[idx]
        cols, rows = img_np.shape[1], img_np.shape[0]
        v = Viewer(size=(cols, rows))
        camera = OpenCVCamera(cam_int[idx].numpy(), cam_ext[idx].numpy(), cols, rows, viewer=v)
        billboard = Billboard.from_camera_and_distance(camera, 5.0, cols, rows, [img_np])
        v.scene.add(billboard, smpl_seq, camera)
        v.set_temp_camera(camera)

        v.scene.floor.enabled = False
        v.scene.origin.enabled = False
        v.shadows_enabled = False
        v.run()
