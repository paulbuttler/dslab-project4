import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import torch

from utils.transforms import denormalize

def plot_landmarks_with_uncertainty(image_tensor, gt_landmarks, pred_landmarks, target_size, save_path=None):
    """
    Visualize 2D landmarks with uncertainty
    
    Params:
        image_tensor: [C, H, W]
        gt_landmarks: [num_landmarks, 2]
        pred_landmarks: [num_landmarks, 3] (mu_x, mu_y, logvar)
        target_size: (H, W)
        save_path:
    """
    denorm_img = denormalize(image_tensor.unsqueeze(0))[0]
    img = denorm_img.permute(1, 2, 0).cpu().numpy()
    
    h, w = target_size
    scale = torch.tensor([w, h])
    
    gt_points = gt_landmarks * scale
    
    pred_mu = pred_landmarks[:, :2] * scale
    pred_std = torch.exp(0.5 * pred_landmarks[:, 2]) * scale.mean()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.axis('off')
    
    ax.scatter(gt_points[:, 0], gt_points[:, 1], c='green', s=50, 
               edgecolors='white', label='Ground Truth', alpha=0.7)
    
    for i, (x, y) in enumerate(pred_mu):
        color = 'red' if i < 15 else 'blue'
        circle = plt.Circle((x, y), pred_std[i], 
                          color=color, fill=True, alpha=0.3)
        ax.add_patch(circle)
        ax.scatter(x, y, c=color, s=50, 
                 edgecolors='white', label=f'Pred {i}' if i==0 else "")
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(),
             loc='upper right', fontsize=8)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()