import sys
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ConfigManager
from utils.datasets import SynDataset
from scripts.train import Trainer
from utils.transforms import denormalize

def visualize_samples(trainer, samples_indices=range(3)):
    '''Visualize on samples from test set'''
    os.makedirs(trainer.config.vis_dir, exist_ok=True)
    device = trainer.device
    model = trainer.model
    dataset = trainer.test_set
    
    indices = samples_indices
    
    for i, idx in enumerate(indices):
        image, targets = dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            pred_landmarks = outputs['landmarks'].view(-1, 3).cpu()  # [21, 3]
            gt_landmarks = targets['landmarks'].view(-1, 2).cpu()  # [21, 2]
            # print(f"pred_landmarks shape:{pred_landmarks.shape}")
            # print(f"gt_landmarks shape:{gt_landmarks.shape}")
        
        img = denormalize(image)[0].permute(1, 2, 0).cpu().numpy()
        
        h, w = image.shape[-2:]
        assert(h == w)
        
        gt_points = gt_landmarks * h
        pred_mu = pred_landmarks[:, :2] * h
        pred_std = torch.exp(0.5 * pred_landmarks[:, 2]) * h

        for j in range(pred_mu.shape[0]):
            print(f"----------- \n {j}-th landmark \n ground truth: ({gt_points[j, 0]}, {gt_points[j, 1]}) \n pred mu: \
                    ({pred_mu[j, 0]}, {pred_mu[j, 1]}); pred sigma: {pred_std[j]}")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.axis('off')
        
        for j, (x, y) in enumerate(gt_points):
            color = 'red' if j < 15 else 'blue'
            ax.scatter(x, y, c=color, s=3, edgecolors='black', label='Ground Truth' if j==0 else "")
        
        for j, (x, y) in enumerate(pred_mu):
            color = 'red' if j < 15 else 'blue'
            ax.add_patch(plt.Circle((x, y), pred_std[j], 
                                  color=color, alpha=0.1))
            ax.scatter(x, y, c=color, s=1, 
                       edgecolors='gray', label='Predicted' if j==0 else "")
            ax.plot([x, gt_points[j, 0]], [y, gt_points[j, 1]], linewidth=0.8, alpha=0.5)
        
        ax.legend(loc='upper right')
        
        save_path = os.path.join(trainer.config.vis_dir, f'sample_{i+1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        plt.close()
        print(f"Saved visualization: {save_path}")

if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.get_config()
    trainer = Trainer(config)

    trainer.test()
    
    print("\nVisualizing random samples...")
    visualize_samples(trainer)