import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
)
from datasets.synth_dataset import SynDataset
from models.model import MultiTaskDNN
from models.smplx import SMPLHLayer
from utils.config import ConfigManager
from training.losses import DNNMultiTaskLoss
from tqdm import tqdm
import yaml
import joblib
import wandb
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import uuid
import multiprocessing as mp

mp.set_start_method("spawn", force=True)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.body = "body" in self.config.meta_file
        # logs and monitor
        self.train_history = {
            "total": [],
            "rotation": [],
            "translation": [],
            "landmark": [],
            "pose": [],
            "landmark_mse": [],
            "mean_var": [],
        }
        self.val_history = {
            "total": [],
            "rotation": [],
            "translation": [],
            "landmark": [],
            "pose": [],
            "landmark_mse": [],
            "mean_var": [],
        }
        if self.body:
            self.train_history["shape"] = []
            self.val_history["shape"] = []
        # Generate unique run name
        timestamp = datetime.now().strftime("%m%d-%H%M")
        self.run_name = f"{timestamp}_{config.name}_{str(uuid.uuid4())[:5]}"
        # Init wandb logger
        wandb_logger = WandbLogger(
            name=self.run_name,
            project=config.project,
            entity=config.entity,
        )
        wandb_logger.experiment.config.update(vars(config))  # log all hyperparams
        self.wandb_logger = wandb_logger

        # Initialize components
        self._init_datasets()
        self._init_model()
        self._init_optimizer()
        self._init_scheduler()
        self._init_smplh_layer()
        self.criterion = DNNMultiTaskLoss(config, self.smplh_layer).to(self.device)

    def _init_smplh_layer(self):
        """Initialize SMPLH layer for calculating loss"""
        self.smplh_layer = SMPLHLayer(
            model_path=self.config.smplh_model_path,
            gender="neutral",
            use_pca=False,
            flat_hand_mean=True,
            num_betas=16,
            dtype=torch.float32,
        ).to(self.device)

    def _init_datasets(self):
        """Initialize datasets with splits"""

        full_length = 95575 if self.body else 99200

        # Create dataset splits
        val_size = int(full_length * self.config.val_ratio)
        test_size = int(full_length * self.config.test_ratio)
        train_size = full_length - val_size - test_size

        all_indices = torch.randperm(full_length, generator=torch.Generator().manual_seed(self.config.seed))
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size : train_size + val_size]

        meta = joblib.load(self.config.meta_file)

        # Create datasets
        train_set = SynDataset(
            img_dir=self.config.data_root,
            metadata=meta,
            aug=config.aug,
            indices=train_indices,
            mode="train",
            device=self.config.device,
        )
        val_set = SynDataset(
            img_dir=self.config.data_root,
            metadata=meta,
            aug=config.aug,
            indices=val_indices,
            mode="test",
            device=self.config.device,
        )
        if test_size > 0:
            test_indices = all_indices[train_size + val_size :]
            self.test_set = SynDataset(
                img_dir=self.config.data_root,
                metadata=meta,
                aug=config.aug,
                indices=test_indices,
                mode="test",
                device=self.config.device,
            )

        # Create dataloaders
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers // 2,
            drop_last=False,
            pin_memory=True,
        )
        if test_size > 0:
            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.config.val_batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
        else:
            self.test_loader = None

    def _init_model(self):
        """Initialize model and move to device"""
        self.model = MultiTaskDNN(
            backbone_name=self.config.backbone_name,
            pretrained=(self.config.pretrained if self.config.checkpoint == "None" else False),
            num_landmarks=self.config.num_landmarks,
            num_pose_params=self.config.num_pose_params,
            num_shape_params=self.config.num_shape_params,
            backbone_feat_dim=self.config.backbone_feat_dim,
            mlp_head_hidden_dim=self.config.mlp_head_hidden_dim,
        ).to(self.device)

    def _init_optimizer(self):
        """Initialize optimizer with model parameters"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        if self.config.scheduler_type == "cosinewarmrest":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.step_per_period,
                T_mult=1,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler_type == "None":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint["loss"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def _compute_loss(self, outputs, targets):
        """Compute multi-task loss"""
        device_outputs = {k: v.to(self.device) for k, v in outputs.items()}
        device_targets = {k: v.to(self.device) for k, v in targets.items()}
        return self.criterion(device_outputs, device_targets)

    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = {
            "total": 0,
            "landmark": 0,
            "pose": 0,
            "translation": 0,
            "rotation": 0,
            "landmark_mse": 0,
            "mean_var": 0,
        }
        if self.body:
            running_loss["shape"] = 0
        last_loss = {}
        log_interval = self.config.log_interval
        val_interval = self.config.val_interval
        accumulation_steps = self.config.accumulation_steps

        self.optimizer.zero_grad()

        for batch_idx, (images, targets, _) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)
        ):
            images = images.to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}

            outputs = self.model(images)
            loss_dict = self._compute_loss(outputs, targets)

            loss = loss_dict["total"] / accumulation_steps
            loss.backward()

            # simulate larger batch size
            if (batch_idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Accumulate running loss
            for k in running_loss:
                running_loss[k] += loss_dict[k].item()

            # log training loss
            if (batch_idx + 1) % log_interval == 0:
                for k in running_loss:
                    last_loss[k] = running_loss[k] / log_interval

                if self.wandb_logger:
                    log_data = {
                            "train/total_loss": last_loss["total"],
                            "train/landmark_loss": last_loss["landmark"],
                            "train/pose_loss": last_loss["pose"],
                            "train/translation_loss": last_loss["translation"],
                            "train/rotation_loss": last_loss["rotation"],
                            "train/landmark_mse_loss": last_loss["landmark_mse"],
                            "train/landmark_mean_var": last_loss["mean_var"],
                            "step": (epoch - 1) * len(self.train_loader) + batch_idx + 1,
                        }
                    if self.body:
                        log_data["train/shape_loss"] = last_loss["shape"]
                    self.wandb_logger.log_metrics(log_data)

                self.train_history["total"].append(last_loss["total"])
                self.train_history["landmark"].append(last_loss["landmark"])
                self.train_history["pose"].append(last_loss["pose"])
                self.train_history["translation"].append(last_loss["translation"])
                self.train_history["rotation"].append(last_loss["rotation"])
                self.train_history["landmark_mse"].append(last_loss["landmark_mse"])
                self.train_history["mean_var"].append(last_loss["mean_var"])

                # reset running loss
                running_loss = {
                    "total": 0,
                    "landmark": 0,
                    "pose": 0,
                    "translation": 0,
                    "rotation": 0,
                    "landmark_mse": 0,
                    "mean_var": 0,
                }
                if self.body:
                    self.train_history["shape"].append(last_loss["shape"])
                    running_loss["shape"] = 0

            # log validation loss
            if (batch_idx + 1) % val_interval == 0:
                val_loss = self._validate()
                if self.wandb_logger:
                    log_data = {
                            "val/total_loss": val_loss["total"],
                            "val/landmark_loss": val_loss["landmark"],
                            "val/pose_loss": val_loss["pose"],
                            "val/translation_loss": val_loss["translation"],
                            "val/rotation_loss": val_loss["rotation"],
                            "val/landmark_mse_loss": val_loss["landmark_mse"],
                            "val/landmark_mean_var": val_loss["mean_var"],
                            "step": (epoch - 1) * len(self.train_loader) + batch_idx + 1,
                        }
                    if self.body: 
                        log_data["val/shape_loss"] = val_loss["shape"]
                        
                    self.wandb_logger.log_metrics(log_data)

                self.val_history["total"].append(val_loss["total"])
                self.val_history["landmark"].append(val_loss["landmark"])
                self.val_history["pose"].append(val_loss["pose"])
                self.val_history["translation"].append(val_loss["translation"])
                self.val_history["rotation"].append(val_loss["rotation"])
                self.val_history["landmark_mse"].append(val_loss["landmark_mse"])
                self.val_history["mean_var"].append(val_loss["mean_var"])

                if self.body: 
                    self.val_history["shape"].append(val_loss["shape"]) 

                self.model.train()

        # step the scheduler after each epoch and log the learning rate
        if self.wandb_logger:
            wandb.log({"epoch": epoch, "learning_rate": self.optimizer.param_groups[0]["lr"]})
        if self.scheduler is not None:
            self.scheduler.step()

        return last_loss

    def _validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = {
            "total": 0,
            "landmark": 0,
            "pose": 0,
            "translation": 0,
            "rotation": 0,
            "landmark_mse": 0,
            "mean_var": 0,
        }
        if self.body:
            val_loss["shape"] = 0

        with torch.no_grad():
            for images, targets, _ in tqdm(self.val_loader, desc="Validating", leave=False):
                images = images.to(self.device, non_blocking=True)
                targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
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
        if self.config.checkpoint != "None":
            self._load_checkpoint(self.config.checkpoint)
        else:
            self.start_epoch = 1
            self.best_loss = float("inf")

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")

            # Train & validate
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate()

            # Log epoch summary
            if self.wandb_logger:
                self.wandb_logger.log_metrics(
                    {
                        "epoch_summary/train_total_loss": train_loss["total"],
                        "epoch_summary/val_total_loss": val_loss["total"],
                        "epoch": epoch,
                    }
                )

            if val_loss["total"] < self.best_loss or epoch % 50 == 0:
                if val_loss["total"] < self.best_loss:
                    model_name = "best_model"
                    self.best_loss = val_loss["total"]
                else:
                    model_name = f"model_epoch_{epoch}"

                loss = val_loss["total"]
                torch.save(
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler is not None else None),
                    },
                    os.path.join(self.config.save_dir, f"{model_name}_{self.run_name}.pth"),
                )

    def test(self):
        """Final test on test set"""
        checkpoint = torch.load(os.path.join(self.config.save_dir, f"best_model{self.run_name}.pth"))
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        test_loss = {
            "total": 0,
            "landmark": 0,
            "pose": 0,
            "translation": 0,
            "rotation": 0,
            "landmark_mse": 0,
            "mean_var": 0,
        }
        if self.body:
            test_loss["shape"] = 0

        with torch.no_grad():
            for images, targets, _ in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device, non_blocking=True)
                targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}

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

    # Initialize trainer
    trainer = Trainer(config)

    # Save config for reproducibility
    with open(os.path.join(config.save_dir, f"config_{trainer.run_name}.yaml"), "w") as f:
        yaml.dump(vars(config), f)

    # Run training
    trainer.train()
    if trainer.test_loader is not None:
        trainer.test()

    wandb.finish()
