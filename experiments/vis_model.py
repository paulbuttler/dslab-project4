import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.config import ConfigManager
from models.model import MultiTaskDNN
from datasets.transforms import denormalize
from datasets.synth_dataset import SynDataset
from torch.utils.data import DataLoader, random_split

def get_val_dataloader(config, batch_size=16):
    """Initialize datasets with splits"""
    full_length = 95575

    # Create dataset splits
    val_size = int(full_length * config.val_ratio)
    test_size = int(full_length * config.test_ratio)
    train_size = full_length - val_size - test_size

    all_indices = torch.randperm(full_length, generator=torch.Generator().manual_seed(config.seed))
    val_indices = all_indices[train_size:train_size + val_size]

    val_set = SynDataset(
        img_dir=config.data_root,
        body_meta_dir=config.meta_file,
        indices=val_indices,
        mode="test",
        device=config.device,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
        
    return val_loader



def visualize_samples(model, batch,  n=100, draw_gt=True, draw_std=False, save_dir="./experiments/body/visualization"):
    """
    Visualize sampled predictions from a batch of images.
    
    Args:
        model: The PyTorch model.
        batch: A tuple (images, targets) from the DataLoader.
        n: Number of landmarks to sample.
        draw_gt: Whether to draw ground truth landmarks.
        draw_std: Whether to draw uncertainty circles.
        save_dir: Directory to save visualizations.
    """

    model.eval()
    images, targets, _ = batch

    h, w = images.shape[-2:]

    with torch.no_grad():
        outputs = model(images)

        pred_kp = (outputs['landmarks'][..., :2] *h).cpu().numpy()  # [B, N, 2]
        pred_std = (torch.exp(0.5 * outputs['landmarks'][..., 2]) *h).cpu().numpy()  # [B, N]

    images = denormalize(images) # [B, 3, H, W]
    gt_kp = (targets["landmarks"] *h).cpu().numpy() # [B, N, 2]


    # Randomly sample landmarks
    sample_indices = np.random.permutation(1100)[:n] # [n]
    gt_kp = gt_kp[:, sample_indices, :].astype(int) # [B, n, 2]
    pred_kp = pred_kp[:, sample_indices, :].astype(int) # [B, n, 2]
    pred_std = pred_std[:, sample_indices] # [B, n]

    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0).cpu().numpy()

        gt_points = gt_kp[i] # [n, 2]
        pred_mu = pred_kp[i] # [n, 2]
        pred_sigma = pred_std[i] # [n]

        # Print sampled points info
        print(f"\nSample {i+1} - Sampled {n} landmarks:")
        for j in range(n):
            print(f"Landmark {j}: GT ({gt_points[j, 0]:.1f}, {gt_points[j, 1]:.1f}) | "
                  f"Pred ({pred_mu[j, 0]:.1f}, {pred_mu[j, 1]:.1f}) ± {pred_sigma[j]:.2f}")
        
        # Visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        ax = plt.gca()
        ax.axis('off')
        
        # Unified color scheme
        gt_color = 'lime'
        pred_color = 'red'
        uncertainty_alpha = 0.15
        
        ax.scatter(pred_mu[:, 0], pred_mu[:, 1], c=pred_color, s=10, edgecolor='white', label='Prediction', alpha=0.8, linewidths=0.3)
        if draw_gt:
            ax.scatter(gt_points[:, 0], gt_points[:, 1], c=gt_color, s=10, edgecolor='black', label='Ground Truth', alpha=0.8, linewidths=0.3)
        
        if draw_std or draw_gt:
            for j, (x, y) in enumerate(pred_mu):

                if draw_std:
                    ax.add_patch(plt.Circle(
                        (x, y), pred_sigma[j], 
                        color=pred_color, alpha=uncertainty_alpha
                    ))
                if draw_gt:
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
    run_name = "0420-2306_Test_Run_4f4df"

    config_manager = ConfigManager(f"./experiments/body/checkpoints/config_{run_name}.yaml")
    config = config_manager.get_config()

    val_loader = get_val_dataloader(config, batch_size=5)
    batch = next(iter(val_loader))

    model = MultiTaskDNN(pretrained=False).to(config.device)
    model.load_state_dict(torch.load(f"./experiments/body/checkpoints/best_model_{run_name}.pth")['model_state_dict'])

    print("\nVisualizing random samples...")
    visualize_samples(model, batch, draw_gt=True, draw_std=False, n=550)