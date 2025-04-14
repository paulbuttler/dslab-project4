import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.dataset import SynDataset
from models.model import MultiTaskDNN
from models.smplx import SMPLHLayer
from utils.config import ConfigManager
from utils.transforms import get_timm_transform
from utils.losses import DNNMultiTaskLoss
from tqdm import tqdm
import yaml

import wandb
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import uuid


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        self.train_history = {'total': [], 'rotation': [], 'translation': [], 
                            'landmark': [], 'pose': [], 'shape': []}
        self.val_history = {'total': [], 'rotation': [], 'translation': [], 
                           'landmark': [], 'pose': [], 'shape': []}
        self.wandb_logger = wandb_logger
        
        # Initialize components
        self._init_datasets()
        self._init_model()
        self._init_optimizer()
        self._init_scheduler()
        self._init_smplh_layer()
        self.criterion = DNNMultiTaskLoss(config, self.smplh_layer)

    def _init_smplh_layer(self):
        """Initialize SMPLH layer for calculating loss"""
        self.smplh_layer = SMPLHLayer(
            model_path=self.config.smplh_model_path,
            gender='neutral',
            use_pca=False,
            num_betas=10,
            dtype=torch.float32
        ).to(self.device)
    
    def _init_datasets(self):
        """Initialize datasets with splits"""
        full_dataset = SynDataset(
            img_dir=self.config.data_root,
            body_meta_dir=config.meta_file,
            model=self.config.backbone_name,
            mode='train',
            device=self.config.device,
        )
        
        # Split dataset
        val_size = int(len(full_dataset) * self.config.val_ratio)
        test_size = int(len(full_dataset) * self.config.test_ratio)
        train_size = len(full_dataset) - val_size - test_size
        
        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        self.train_set.mode = 'train'
        self.val_set.mode = 'test' # no augmentation!
        self.test_set.mode = 'test' # no augmentation!
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=False
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers
        )
    
    def _init_model(self):
        """Initialize model and move to device"""
        self.model = MultiTaskDNN(
            backbone_name=self.config.backbone_name,
            pretrained=self.config.pretrained,
            num_landmarks=self.config.num_landmarks,
            num_pose_params=self.config.num_pose_params,
            num_shape_params=self.config.num_shape_params,
            num_backbone_features=self.config.num_backbone_features,
            mlp_head_hidden_dim=self.config.mlp_head_hidden_dim,
        ).to(self.device)
        
        if self.config.freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
    
    def _init_optimizer(self):
        """Initialize optimizer with parameter groups"""
        params = [
            {'params': self.model.backbone.parameters(), 'lr': self.config.backbone_lr}, # Do we need to freeze backbone during training?
            {'params': self.model.feature_adapter.parameters()},
            {'params': self.model.landmark_head.parameters()},
            {'params': self.model.pose_head.parameters()},
            {'params': self.model.shape_head.parameters()}
        ]
        self.optimizer = AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.step_per_period,
                T_mult=1,
                eta_min=self.config.min_lr,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
    
    def _compute_loss(self, outputs, targets):
        """Compute multi-task loss"""
        device_outputs = {k: v.to(self.device) for k, v in outputs.items()}
        device_targets = {k: v.to(self.device) for k, v in targets.items()}
        return self.criterion(device_outputs, device_targets)
    
    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = {'total': 0, 'landmark': 0, 'pose': 0, 'shape': 0, 'translation': 0, 'rotation': 0}
        running_loss = {'total': 0, 'landmark': 0, 'pose': 0, 'shape': 0, 'translation': 0, 'rotation': 0}
        iter_count = 0
        log_interval = self.config.log_interval
        val_interval = self.config.val_interval
        scheduler_interval = self.config.iteration_per_scheduler_step
        
        for batch_idx, (images, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)):
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss_dict = self._compute_loss(outputs, targets)
            
            loss_dict['total'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            for k in epoch_loss:
                epoch_loss[k] += loss_dict[k].item()

            # Accumulate losses
            for k in running_loss:
                running_loss[k] += loss_dict[k].item()
            iter_count += 1

            # log
            if iter_count % log_interval == 0:
                avg_total = running_loss['total'] / log_interval
                avg_landmark = running_loss['landmark'] / log_interval
                avg_pose = running_loss['pose'] / log_interval
                avg_translation = running_loss['translation'] / log_interval
                avg_shape = running_loss['shape'] / log_interval
                avg_rotation = running_loss['rotation'] / log_interval

                if self.wandb_logger:
                    self.wandb_logger.log_metrics({
                        "train/total_loss": avg_total,
                        "train/landmark_loss": avg_landmark,
                        "train/pose_loss": avg_pose,
                        "train/shape_loss": avg_shape,
                        "train/translation_loss": avg_translation,
                        "train/rotation_loss": avg_rotation,
                        "epoch": epoch,
                        "step": iter_count
                    })

                '''
                print(f"Epoch {epoch} Iteration {iter_count}: "
                      f"Avg Total Loss: {avg_total:.4f}, "
                      f"landmark Loss: {avg_landmark:.4f}, pose Loss: {avg_pose:.4f}, shape Loss: {avg_shape:.4f}, "
                      f"rotation Loss: {avg_rotation:.4f}, translation Loss: {avg_translation:.4f}")
                '''
                self.train_history['total'].append(avg_total)
                self.train_history['landmark'].append(avg_landmark)
                self.train_history['pose'].append(avg_pose)
                self.train_history['shape'].append(avg_shape)
                self.train_history['translation'].append(avg_translation)
                self.train_history['rotation'].append(avg_rotation)
                running_loss = {'total': 0, 'landmark': 0, 'pose': 0, 'shape': 0, 'translation': 0, 'rotation': 0}

            # val
            if iter_count % val_interval == 0:
                val_loss = self._validate()
                if self.wandb_logger:
                    self.wandb_logger.log_metrics({
                        "val/total_loss": val_loss['total'],
                        "val/landmark_loss": val_loss['landmark'],
                        "val/pose_loss": val_loss['pose'],
                        "val/shape_loss": val_loss['shape'],
                        "val/translation_loss": val_loss['translation'],
                        "val/rotation_loss": val_loss['rotation'],
                        "epoch": epoch,
                        "step": iter_count
                    })
                '''
                print(f"Validation at Epoch {epoch} Iter {iter_count}: ")
                for _, (key, value) in enumerate(val_loss.items()):
                    print(f"{key} loss: {value:.4f}")
                '''
                self.val_history['total'].append(avg_total)
                self.val_history['landmark'].append(avg_landmark)
                self.val_history['pose'].append(avg_pose)
                self.val_history['shape'].append(avg_shape)
                self.val_history['translation'].append(avg_translation)
                self.val_history['rotation'].append(avg_rotation)

            # lr scheduler step
            if iter_count % scheduler_interval == 0:
                self.scheduler.step()
        
        # Average losses
        for k in epoch_loss:
            epoch_loss[k] /= len(self.train_loader)
        return epoch_loss
    
    def _validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = {'total': 0, 'landmark': 0, 'pose': 0, 'shape': 0, 'translation': 0, 'rotation': 0}
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                outputs = self.model(images)
                loss_dict = self._compute_loss(outputs, targets)
                
                for k in val_loss:
                    val_loss[k] += loss_dict[k].item()
        
        # Average losses
        for k in val_loss:
            val_loss[k] /= len(self.val_loader)
        return val_loss
    
    def train(self):
        """Full training loop"""
        best_loss = float('inf')
        
        for epoch in range(1, self.config.epochs+1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            
            # Train & validate
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate()
            
            # Print metrics
            if self.wandb_logger:
                self.wandb_logger.log_metrics({
                    "epoch_summary/train_total_loss": train_loss['total'],
                    "epoch_summary/val_total_loss": val_loss['total'],
                    "epoch": epoch
                })
            '''
            print(f"Epoch {epoch} Summary:")
            print("Train Loss:")
            for _, (key, value) in enumerate(train_loss.items()):
                print(f"{key} loss: {value:.4f}")
            print("Validating Loss:")
            for _, (key, value) in enumerate(val_loss.items()):
                print(f"{key} loss: {value:.4f}")
            '''

            if val_loss['total'] < best_loss:
                best_loss = val_loss['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': best_loss,
                }, os.path.join(self.config.save_dir, 'best_model.pth'))


    def test(self):
        """Final test on test set"""
        checkpoint = torch.load(os.path.join(self.config.save_dir, 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        test_loss = {'total': 0, 'landmark': 0, 'pose': 0, 'shape': 0, 'translation': 0, 'rotation': 0}
        
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                outputs = self.model(images)
                loss_dict = self._compute_loss(outputs, targets)
                
                for k in test_loss:
                    test_loss[k] += loss_dict[k].item()
        
        # Average losses
        for k in test_loss:
            test_loss[k] /= len(self.test_loader)
        
        print("\nFinal Test Results:")
        for _, (key, value) in enumerate(test_loss.items()):
            print(f"{key} loss: {value:.4f}")

if __name__ == "__main__":
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Save config for reproducibility
    with open(os.path.join(config.save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(config), f)

    # Generate unique run name
    timestamp = datetime.now().strftime('%m%d-%H%M')
    run_name = f"{timestamp}_{config.name}_{str(uuid.uuid4())[:5]}"

    # Init wandb logger
    wandb_logger = WandbLogger(
        name=run_name,
        project=config.project,
        entity=config.entity,
    )
    wandb_logger.experiment.config.update(vars(config))  # log all hyperparams

    
    # Initialize and run trainer
    trainer = Trainer(config)
    trainer.train()
    trainer.test()
    wandb.finish()