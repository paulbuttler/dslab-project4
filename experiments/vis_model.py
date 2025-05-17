import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from models.smplx import SMPLHLayer
from models.load import get_val_dataloader, get_predictions, get_full_shape_and_pose, load_model
from aitviewer.models.smpl import SMPLLayer  # type: ignore
from aitviewer.renderables.smpl import SMPLSequence  # type: ignore
from aitviewer.renderables.billboard import Billboard  # type: ignore
from aitviewer.scene.camera import OpenCVCamera  # type: ignore
from aitviewer.viewer import Viewer  # type: ignore
from utils.rottrans import aa2rot, rot2aa, rotation_6d_to_matrix
from utils.roi import compute_roi
from datasets.transforms import apply_roi_transform, normalize, denormalize


def visualize_landmarks(
    images, pred_ldmks, gt_ldmks=None, std=None, n=100, save_dir="./experiments/body/visualization"
):
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

    print("\nVisualizing landmarks")
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
            print(
                f"Mean distance of 3d ground-truth body joints used for training to actual gt-joints: {diff[:22].mean()}"
            )
            print(
                f"Mean distance of 3d ground-truth hand joints used for training to actual gt-joints: {diff[22:].mean()}"
            )
        elif part == "hand":
            diff = np.linalg.norm(gt_joints[:15] - smplh_joints[0, 22:37], axis=1)
            print(f"Mean distance of 3d ground-truth hand joints used for training to actual gt-joints: {diff.mean()}")


def visualize_pose_and_shape(
    uids,
    gt_pose,
    gt_shape,
    full_pose,
    full_shape,
    translation,
    part="body",
    data_root="data/raw/synth_body",
):

    print(f"\nVisualizing {part} pose" + "and shape" * (part == "body" or part == "full"))
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=full_shape.shape[1])
    # gt_pose_rot = aa2rot(gt_pose)

    gt_shape = gt_shape[:, : full_shape.shape[1]]

    for idx in range(len(uids)):
        meta_file = os.path.join(data_root, f"metadata_{uids[idx]}.json")
        img_file = os.path.join(data_root, f"img_{uids[idx]}.jpg")
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        # Get camera data
        world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
        camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])

        # print("Test SMPLH layer")
        # gt_joints = torch.tensor(metadata["landmarks"]["3D_world"], dtype=torch.float32)
        # test_smpl_layer(
        #     gt_pose_rot[idx].unsqueeze(0),
        #     gt_pose[idx].unsqueeze(0),
        #     gt_shape[idx].unsqueeze(0),
        #     translation[idx].unsqueeze(0),
        #     gt_joints,
        #     part,
        # )

        pose = gt_pose[idx]  # [52, 3]
        pred_pose = full_pose[idx]  # [52, 3]
        shape = gt_shape[idx]  # [16]
        pred_shape = full_shape[idx]  # [16]
        transl = translation[idx]  # [3]

        cat_pose = torch.stack([pose, pred_pose], dim=0)
        cat_shape = torch.stack([shape, pred_shape], dim=0)
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
        billboard = Billboard.from_camera_and_distance(camera, 5.0, cols, rows, [img_rgb])
        v.scene.add(billboard, smpl_seq, camera)
        v.set_temp_camera(camera)
        v.scene.floor.enabled = False
        v.scene.origin.enabled = False
        v.shadows_enabled = False
        v.run()


def visualize_batch_of_images(images):
    from torchvision.utils import make_grid

    grid = make_grid(images, nrow=5, padding=2)  # [C, H', W']
    grid_np = grid.permute(1, 2, 0).cpu().numpy()  # [H', W', C]
    grid_bgr = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("Batch Grid", grid_bgr)
    cv2.waitKey(0)


def initial_pose_estimation(images, body_model, hand_model, device):
    """
    Perform initial pose, shape and refined landmark estimation using trained body and hand models.
    """

    normalized_images = normalize(images)

    body_model.eval()
    with torch.no_grad():
        outputs = body_model(normalized_images)
        ldmks = (outputs["landmarks"][..., :2] * 256.0).cpu()
        std = (torch.exp(0.5 * outputs["landmarks"][..., 2])).cpu()
        body_pose = rotation_6d_to_matrix(outputs["pose"]).cpu()
        shape = outputs["shape"].cpu()

    left_hand_ldmks = ldmks[:, 665:802]
    right_hand_ldmks = ldmks[:, 802:939]

    left_hand_roi = compute_roi(left_hand_ldmks, None, 0.15, images.shape[-2]).to(device, dtype=torch.float32)
    right_hand_roi = compute_roi(right_hand_ldmks, None, 0.15, images.shape[-2]).to(device, dtype=torch.float32)

    left_hand_img, M1 = apply_roi_transform(images, None, left_hand_roi, "inference", 128.0)
    right_hand_img, M2 = apply_roi_transform(images, None, right_hand_roi, "inference", 128.0)

    # visualize_batch_of_images(left_hand_img)
    # visualize_batch_of_images(torch.flip(right_hand_img, [-1]))

    left_hand_img = normalize(left_hand_img)
    right_hand_flipped = normalize(torch.flip(right_hand_img, [-1]))

    hand_model.eval()
    with torch.no_grad():
        left_out = hand_model(left_hand_img)
        right_out = hand_model(right_hand_flipped)

        left_hand_ldmks_crop = left_out["landmarks"][..., :2] * 128.0
        left_hand_std = (torch.exp(0.5 * left_out["landmarks"][..., 2])).cpu()
        left_hand_pose = rotation_6d_to_matrix(left_out["pose"]).cpu()

        right_hand_ldmks_crop = right_out["landmarks"][..., :2] * 128.0
        right_hand_std = (torch.exp(0.5 * right_out["landmarks"][..., 2])).cpu()
        right_hand_pose_flipped = rotation_6d_to_matrix(right_out["pose"]).cpu()

        # Flip landmarks and pose
        right_hand_ldmks_crop[..., 0] = 128.0 - right_hand_ldmks_crop[..., 0]

    def warp_back(ldmks_crop, M):
        B, N, _ = ldmks_crop.shape
        ldmks_crop_h = torch.cat([ldmks_crop, torch.ones_like(ldmks_crop[..., :1])], dim=-1)  # [B, N, 3]
        M_h = torch.cat([M, torch.tensor([0.0, 0.0, 1.0], device=M.device).view(1, 1, 3).expand(B, -1, -1)], dim=1)
        M_inv = torch.inverse(M_h)  # [B, 3, 3]
        ldmks_orig = torch.bmm(ldmks_crop_h, M_inv.transpose(1, 2))  # [B, N, 3]
        return ldmks_orig[..., :2].cpu()

    left_hand_ldmks_orig = warp_back(left_hand_ldmks_crop, M1)
    right_hand_ldmks_orig = warp_back(right_hand_ldmks_crop, M2)

    # Replace the original hand landmarks with the refined ones
    ldmks[:, 665:802] = left_hand_ldmks_orig
    ldmks[:, 802:939] = right_hand_ldmks_orig
    std[:, 665:802] = left_hand_std
    std[:, 802:939] = right_hand_std

    # Concatenate pose parameters and flip the right hand pose
    pose = torch.cat([body_pose, left_hand_pose, right_hand_pose_flipped], dim=1)
    pose = rot2aa(pose)
    pose[:, -15:] *= torch.tensor([1.0, -1.0, -1.0], device=pose.device)

    return ldmks, std, pose, shape


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["body", "hand", "full"], default="body")
    args = parser.parse_args()

    part = args.part

    if part == "body" or part == "hand":
        data_root = f"data/raw/synth_{part}"
        meta_file = f"data/annotations/{part}_meta.pkl"

        model, config = load_model(part)
        val_loader = get_val_dataloader(config, data_root, meta_file, 15)
        batch = next(iter(val_loader))

        (
            images,
            uids,
            translation,
            landmarks,
            gt_shape,
            gt_pose,
            pred_ldmks,
            pred_std,
            pred_shape,
            pred_pose_rot,
        ) = get_predictions(model, batch, config.device)

        # Concatenate missing ground truth parameters
        full_shape, full_pose = get_full_shape_and_pose(pred_shape, rot2aa(pred_pose_rot), gt_shape, gt_pose)

        # Visualize landmarks, pose and shape
        visualize_landmarks(images, pred_ldmks, None, None, pred_ldmks.shape[1], f"experiments/synth_data/{part}")
        visualize_pose_and_shape(uids, gt_pose, gt_shape, full_pose, full_shape, translation, part, data_root)

    elif part == "full":
        data_root = f"data/raw/synth_body"
        meta_file = f"data/annotations/body_meta.pkl"

        body_model, config = load_model("body")
        hand_model, _ = load_model("hand")
        val_loader = get_val_dataloader(config, data_root, meta_file, 15)

        images, targets, uids = next(iter(val_loader))
        images = images.to(config.device)
        images = denormalize(images)

        # Perform pose, shape and refined landmark estimation
        ldmks, std, pose, shape = initial_pose_estimation(images, body_model, hand_model, config.device)

        # Concatenate global orientation
        gt_pose = targets["pose"]
        full_pose = torch.cat([gt_pose[:, 0].unsqueeze(1), pose], dim=1)

        # Visualize landmarks, pose and shape
        visualize_landmarks(images, ldmks, None, None, n=ldmks.shape[1], save_dir=f"experiments/synth_data/{part}")
        visualize_pose_and_shape(
            uids, targets["pose"], targets["shape"], full_pose, shape, targets["translation"], part, data_root
        )
