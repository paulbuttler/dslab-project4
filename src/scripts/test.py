import os
import torch
import matplotlib.pyplot as plt
from utils.config import ConfigManager
from scripts.train import Trainer
from datasets.transforms import denormalize

import wandb

def visualize_samples(trainer, samples_indices=range(3), n=40):
    '''Visualize on samples from test set (with landmark sampling)'''
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
            # Process all landmarks
            all_pred_landmarks = outputs['landmarks'].view(-1, 3).cpu()  # [N, 3]
            all_gt_landmarks = targets['landmarks'].view(-1, 2).cpu()    # [N, 2]
            
            # Random sampling
            num_landmarks = all_gt_landmarks.shape[0]
            n = min(n, num_landmarks)
            sample_indices = torch.randperm(num_landmarks)[:n]
            
            # Apply sampling
            gt_landmarks = all_gt_landmarks[sample_indices]
            pred_landmarks = all_pred_landmarks[sample_indices]
        
        img = denormalize(image)[0].permute(1, 2, 0).cpu().numpy()
        h, w = image.shape[-2:]
        assert h == w, "Assuming square images"
        
        scale_factor = h
        gt_points = gt_landmarks * scale_factor
        pred_mu = pred_landmarks[:, :2] * scale_factor
        pred_std = torch.exp(0.5 * pred_landmarks[:, 2]) * scale_factor

        # Print sampled points info
        print(f"\nSample {i+1} - Sampled {n} landmarks:")
        for j in range(n):
            print(f"Landmark {j}: GT ({gt_points[j, 0]:.1f}, {gt_points[j, 1]:.1f}) | "
                  f"Pred ({pred_mu[j, 0]:.1f}, {pred_mu[j, 1]:.1f}) ± {pred_std[j]:.2f}")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.axis('off')
        
        # Unified color scheme
        gt_color = 'lime'
        pred_color = 'red'
        uncertainty_alpha = 0.15
        
        # Plot ground truth
        ax.scatter(gt_points[:, 0], gt_points[:, 1], 
                   c=gt_color, s=15, edgecolor='black', 
                   label='Ground Truth', alpha=0.8)
        
        # Plot predictions with uncertainty
        for j, (x, y) in enumerate(pred_mu):
            ax.add_patch(plt.Circle(
                (x, y), pred_std[j], 
                color=pred_color, alpha=uncertainty_alpha
            ))
            
            label = 'Predicted (±1σ)' if j == 0 else None
            ax.scatter(x, y, c=pred_color, s=12, 
                      edgecolor='white', label=label, alpha=0.8)
            
            ax.plot(
                [x, gt_points[j, 0]], 
                [y, gt_points[j, 1]], 
                color='gray', linewidth=0.8, alpha=0.3
            )
        
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in zip(handles, labels)}
        ax.legend(unique_labels.values(), unique_labels.keys(), 
                 loc='upper right', fontsize=9)
        
        save_path = os.path.join(trainer.config.vis_dir, f'sample_{i+1}_sampled.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"\nSaved visualization: {save_path}")

if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.get_config()
    trainer = Trainer(config)

    trainer.test()
    wandb.finish()
    
    print("\nVisualizing random samples...")
    visualize_samples(trainer, n=20)
