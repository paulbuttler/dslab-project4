import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from models.load import get_val_dataloader, get_predictions, get_full_shape_and_pose, load_model
from utils.rottrans import aa2rot, rot2aa
from inference.initialization import initial_pose_estimation
from utils.visualization import visualize_pose_and_shape
from datasets.EHF_dataset import EHF_Dataset
from torch.utils.data import DataLoader


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["body", "hand", "full"], default="body")
    parser.add_argument("--dataset", choices=["synth", "ehf"], default="synth")
    args = parser.parse_args()

    dataset = args.dataset
    part = args.part if dataset == "synth" else "full"

    if part == "body" or part == "hand":
        data_root = f"data/synth_{part}"
        meta_file = f"data/annot/{part}_meta.pkl"

        model, config = load_model(part)
        val_loader = get_val_dataloader(config, data_root, meta_file, 10)
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
        visualize_landmarks(
            images,
            pred_ldmks,
            landmarks,
            None,
            pred_ldmks.shape[1],
            f"experiments/{dataset}/{part}",
        )
        visualize_pose_and_shape(
            uids, gt_pose, gt_shape, full_pose, full_shape, translation, part, dataset, billboard=True
        )

    elif part == "full":

        body_model, config = load_model("body")
        hand_model, _ = load_model("hand")

        if dataset == "synth":
            data_root = "data/raw/synth_body"
            meta_file = "data/annot/body_meta.pkl"
            test_loader = get_val_dataloader(config, data_root, meta_file, 10, "test")

        elif dataset == "ehf":
            test_dataset = EHF_Dataset(data_dir=Path("data/raw/EHF"))
            test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

        images, targets, uids = next(iter(test_loader))
        roi = targets["roi"].to(config.device)
        images = images.to(config.device)

        # Perform pose, shape and refined landmark estimation
        ldmks, std, pose, shape = initial_pose_estimation(images, roi, body_model, hand_model, config.device)

        # Concatenate global orientation
        gt_pose = targets["pose"]
        full_pose = torch.cat([gt_pose[:, 0].unsqueeze(1), pose], dim=1)

        # Visualize landmarks, pose and shape
        visualize_landmarks(
            images,
            ldmks,
            None,
            None,
            ldmks.shape[1],
            save_dir=f"experiments/{dataset}/" + (f"{part}" if dataset == "synth" else "landmarks"),
        )
        visualize_pose_and_shape(
            uids,
            targets["pose"],
            targets["shape"],
            full_pose,
            shape,
            targets["translation"],
            part,
            dataset,
            billboard=True,
        )
