import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.config import ConfigManager
from models.model import MultiTaskDNN
from models.load_model import get_val_dataloader, get_predictions
from aitviewer.models.smpl import SMPLLayer  # type: ignore
from aitviewer.renderables.smpl import SMPLSequence  # type: ignore
from aitviewer.renderables.billboard import Billboard  # type: ignore
from aitviewer.scene.camera import OpenCVCamera  # type: ignore
from aitviewer.viewer import Viewer  # type: ignore


def visualize_samples(
    images,
    pred_ldmks,
    gt_ldmks=None,
    std=None,
    n=100,
    save_dir="./experiments/body/visualization",
):
    """
    Visualizes a subset of predicted landmarks (and optionally ground truth and uncertainty)
    on input images from a batch. Saves each visualization as a separate file.

    Args:
        images (Tensor): Batch of input images, shape [B, C, H, W].
        pred_ldmks (array-like): Predicted landmarks, shape [B, 1100, 2].
        gt_ldmks (array-like, optional): Ground truth landmarks, shape [B, 1100, 2].
        std (array-like, optional): Standard deviations for uncertainty visualization, shape [B, 1100].
        n (int): Number of landmarks to randomly sample and visualize.
        save_dir (str): Directory to save the output images.
    """

    # Randomly sample landmarks
    sample_indices = torch.randperm(1100)[:n]  # [n]
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
        ax.axis('off')

        # Unified color scheme
        gt_color = 'lime'
        pred_color = 'red'
        uncertainty_alpha = 0.15

        ax.scatter(
            pred_points[:, 0],
            pred_points[:, 1],
            c=pred_color,
            s=10,
            edgecolor="white",
            label="Prediction",
            alpha=0.8,
            linewidths=0.3,
        )
        if gt_ldmks is not None:
            gt_points = gt_ldmks[i]  # [n, 2]:
            ax.scatter(gt_points[:, 0], gt_points[:, 1], c=gt_color, s=10, edgecolor='black', label='Ground Truth', alpha=0.8, linewidths=0.3)

        if gt_ldmks is not None or std is not None:
            for j, (x, y) in enumerate(pred_points):

                if std is not None:
                    ax.add_patch(
                        plt.Circle(
                            (x, y), std[i, j], color=pred_color, alpha=uncertainty_alpha
                        )
                    )
                if gt_ldmks is not None:
                    ax.plot([x, gt_points[j, 0]], [y, gt_points[j, 1]], color='gray', linewidth=0.8, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize=9)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'sample_{i+1}.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\nSaved visualization: {save_path}")
        plt.close()


if __name__ == "__main__":
    run_name = "0503-2012_Run_2_cont_74664"
    data_root = "./data/raw/synth_body"
    meta_file = "./data/annotations/body_meta.pkl.gz"

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
    checkpoint = torch.load(f"./experiments/body/checkpoints/best_model_{run_name}.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint['epoch']} epoch")

    val_loader = get_val_dataloader(config, data_root, meta_file)
    batch = next(iter(val_loader))

    (
        images,
        uids,
        ldmks,
        global_orient,
        body_pose,
        left_hand_pose,
        right_hand_pose,
        body_shape,
        pred_ldmks,
        pred_std,
        pred_pose,
        pred_shape,
    ) = get_predictions(model, batch, config.device)

    print("\nVisualizing landmarks")
    visualize_samples(images, pred_ldmks, None, None, n=1100)

    #     # Ground truth pose in global rotation matrix format
    #     gt_global_rot = local_to_global(
    #         gt_pose.reshape(-1, 52 * 3),
    #         part="body",
    #         output_format="rotmat",
    #         input_format="aa",
    #     ).reshape(-1, 52, 3, 3)
    #     pelvis_gt = gt_global_rot[:, 0, :, :].reshape(-1, 1, 3, 3)  # [B, 1, 3, 3]
    #     hand_gt = gt_global_rot[:, 22:, :, :].reshape(-1, 30, 3, 3)  # [B, 30, 3, 3]

    #     # Compute predicted pose in local angle axis format
    #     pred_pose = rotation_6d_to_matrix(pred_pose)  # [B, 21, 3, 3]
    #     cat_pose = torch.cat([pelvis_gt, pred_pose, hand_gt], dim=1)  # [B, 52, 3, 3]
    #     pred_pose = global_to_local(
    #         cat_pose, part="body", output_format="aa"
    #     )  # [B, 52, 3]

    #     body_pose = gt_pose[:, 1:22, :].reshape(-1, 21 * 3)  # [B, 21*3]
    #     pred_pose = pred_pose[:, 1:22, :].reshape(-1, 21 * 3)

    print("\nVisualizing body pose and shape")

    for idx in range(5):
        meta_file = os.path.join(data_root, f"metadata_{uids[idx]}.json")
        img_file = os.path.join(data_root, f"img_{uids[idx]}.jpg")
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        # Get camera data
        translation = np.asarray(metadata["translation"])
        world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
        camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])

        pose = np.asarray(body_pose[idx]).reshape(1, -1)  # [1, 21*3]
        pr_pose = np.asarray(pred_pose[idx]).reshape(1, -1)  # [1, 21*3]
        shape = np.asarray(body_shape[idx]).reshape(1, -1)  # [1, 10]
        pr_shape = np.asarray(pred_shape[idx]).reshape(1, -1)  # [1, 10]
        root_orient = np.asarray(global_orient[idx]).reshape(1, -1)  # [1, 3]
        translation = translation.reshape(1, -1)  # [1, 3]

        cat_pose = np.concatenate([pose, pr_pose], axis=0)  # [2, 21*3]
        cat_shape = np.concatenate([shape, pr_shape], axis=0)  # [2, 10]
        cat_orient = np.concatenate([root_orient, root_orient], axis=0)  # [2, 3]
        cat_trans = np.concatenate([translation, translation], axis=0)  # [2, 3]

        # Create a SMPL sequence.
        smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=10)
        smpl_seq = SMPLSequence(
            smpl_layer=smpl_layer,
            poses_body=cat_pose,
            betas=cat_shape,
            poses_root=cat_orient,
            trans=cat_trans,
        )

        input_img = cv2.imread(img_file)
        img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        cols, rows = input_img.shape[1], input_img.shape[0]

        v = Viewer(size=(cols, rows))
        camera = OpenCVCamera(
            camera_to_image, world_to_camera[:3], cols, rows, viewer=v
        )
        billboard = Billboard.from_camera_and_distance(
            camera, 4.5, cols, rows, [img_rgb]
        )
        v.scene.add(billboard, smpl_seq, camera)
        v.set_temp_camera(camera)
        v.run()
