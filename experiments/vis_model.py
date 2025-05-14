import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.config import ConfigManager
from models.model import MultiTaskDNN
from models.smplx import SMPLHLayer
from models.load import get_val_dataloader, get_predictions, get_full_shape_and_pose
from aitviewer.models.smpl import SMPLLayer  # type: ignore
from aitviewer.renderables.smpl import SMPLSequence  # type: ignore
from aitviewer.renderables.billboard import Billboard  # type: ignore
from aitviewer.scene.camera import OpenCVCamera  # type: ignore
from aitviewer.viewer import Viewer  # type: ignore
from utils.rottrans import aa2rot, rot2aa


def visualize_samples(images, pred_ldmks, gt_ldmks=None, std=None, n=100, save_dir="./experiments/body/visualization"):
    """
    Visualizes a subset of predicted landmarks (and optionally ground truth and uncertainty)
    on a batch of images. Saves each visualization as a separate file.

    Args:
        images (Tensor): Batch of input images, shape [B, C, H, W].
        pred_ldmks (array-like): Predicted landmarks, shape [B, N, 2].
        gt_ldmks (array-like, optional): Ground truth landmarks, shape [B, N, 2].
        std (array-like, optional): Standard deviations for uncertainty visualization, shape [B, N].
        n (int): Number of landmarks to randomly sample and visualize.
        save_dir (str): Directory to save the output images.
    """
    N = pred_ldmks.shape[1]

    # Randomly sample landmarks
    sample_indices = torch.randperm(N)[:n] if n < N else torch.arange(N)  # [n]
    pred_ldmks = np.asarray(pred_ldmks[:, sample_indices, :], dtype=int)  # [B, n, 2]

    if gt_ldmks is not None:
        gt_ldmks = np.asarray(gt_ldmks[:, sample_indices, :], dtype=int)  # [B, n, 2]
    if std is not None:
        std = np.asarray(std[:, sample_indices])  # [B, n]

    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0).cpu().numpy()

        pred_points = pred_ldmks[i]  # [n, 2]

        # Visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        ax = plt.gca()
        ax.axis("off")

        # Unified color scheme
        gt_color = "lime"
        pred_color = "red"
        uncertainty_alpha = 0.15

        ax.scatter(
            pred_points[:, 0],
            pred_points[:, 1],
            c=pred_color,
            s=7,
            edgecolor="white",
            label="Prediction",
            alpha=0.8,
            linewidths=0.3,
        )
        if gt_ldmks is not None:
            gt_points = gt_ldmks[i]  # [n, 2]:
            ax.scatter(
                gt_points[:, 0],
                gt_points[:, 1],
                c=gt_color,
                s=7,
                edgecolor="black",
                label="Ground Truth",
                alpha=0.8,
                linewidths=0.3,
            )

        if gt_ldmks is not None or std is not None:
            for j, (x, y) in enumerate(pred_points):

                if std is not None:
                    ax.add_patch(plt.Circle((x, y), std[i, j], color=pred_color, alpha=uncertainty_alpha))
                if gt_ldmks is not None:
                    ax.plot([x, gt_points[j, 0]], [y, gt_points[j, 1]], color="gray", linewidth=0.8, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize=9)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"sample_{i+1}.pdf")
        plt.savefig(save_path, bbox_inches="tight")
        print(f"\nSaved visualization: {save_path}")
        plt.close()


def test_smpl_layer(pose_rot, pose_aa, shape, translation, gt_joints=None):

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
    print(f"Joints shape: {smplh_joints.shape}")

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
        diff = np.linalg.norm(gt_joints - smplh_joints[0, :52], axis=1)
        print(
            f"Mean distance of 3d ground-truth body joints used for training to actual gt-joints: {diff[:22].mean()}"
        )
        print(
            f"Mean distance of 3d ground-truth hand joints used for training to actual gt-joints: {diff[22:].mean()}"
        )


if __name__ == "__main__":
    run_name = "0509-2106_Run_3_cont_b79aa"
    data_root = "./data/raw/synth_body"
    meta_file = "./data/annotations/body_meta.pkl"

    config_manager = ConfigManager(f"./experiments/body/checkpoints/config_{run_name}.yaml")
    config = config_manager.get_config()

    model = MultiTaskDNN(
        backbone_name=config.backbone_name,
        pretrained=False,
        num_landmarks=config.num_landmarks,
        num_pose_params=config.num_pose_params,
        num_shape_params=config.num_shape_params,
        backbone_feat_dim=config.backbone_feat_dim,
        mlp_head_hidden_dim=config.mlp_head_hidden_dim,
    ).to(config.device)
    checkpoint = torch.load(f"./experiments/body/checkpoints/model_epoch_400_{run_name}.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint['epoch']} epoch")

    val_loader = get_val_dataloader(config, data_root, meta_file, 10)
    batch = next(iter(val_loader))

    (
        images,
        uids,
        translation,
        ldmks,
        gt_shape,
        gt_pose,
        pred_ldmks,
        pred_std,
        pred_shape,
        pred_pose_rot,
    ) = get_predictions(model, batch, config.device)

    gt_pose_rot = aa2rot(gt_pose)
    pred_pose_aa = rot2aa(pred_pose_rot)
    full_shape, full_pose = get_full_shape_and_pose(pred_shape, pred_pose_rot, gt_shape, gt_pose_rot)

    print("\nVisualizing landmarks")
    visualize_samples(images, pred_ldmks, None, None, n=1100)

    print("\nVisualizing body pose")
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=16)

    for idx in range(5, 10):
        meta_file = os.path.join(data_root, f"metadata_{uids[idx]}.json")
        img_file = os.path.join(data_root, f"img_{uids[idx]}.jpg")
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        # Get camera data
        world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
        camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])
        gt_joints = torch.tensor(metadata["landmarks"]["3D_world"], dtype=torch.float32)

        print("Test SMPLH layer")
        test_smpl_layer(
            gt_pose_rot[idx].unsqueeze(0),
            gt_pose[idx].unsqueeze(0),
            gt_shape[idx].unsqueeze(0),
            translation[idx].unsqueeze(0),
            gt_joints,
        )

        pose = gt_pose[idx]  # [52, 3]
        pr_pose = rot2aa(full_pose[idx])  # [52, 3]
        shape = gt_shape[idx]  # [16]
        pr_shape = full_shape[idx]  # [16]
        transl = translation[idx]  # [3]

        cat_pose = torch.stack([pose, pr_pose], dim=0)
        cat_shape = torch.stack([shape, pr_shape], dim=0)
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
        camera = OpenCVCamera(camera_to_image, world_to_camera[:3], cols, rows, viewer=v)
        billboard = Billboard.from_camera_and_distance(camera, 4.5, cols, rows, [img_rgb])
        v.scene.add(billboard, smpl_seq, camera)
        v.set_temp_camera(camera)
        v.run()
