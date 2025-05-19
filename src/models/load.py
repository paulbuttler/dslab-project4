import torch
from models.model import MultiTaskDNN
from utils.config import ConfigManager
from datasets.transforms import denormalize
from utils.rottrans import rotation_6d_to_matrix, aa2rot
from datasets.synth_dataset import SynDataset
from torch.utils.data import DataLoader


def load_model(part="body"):
    """
    Load latest trained hand or body model from checkpoint.
    """
    if part == "body":
        run_name = "0509-2106_Run_3_cont_b79aa"
        epoch = 400
    elif part == "hand":
        run_name = "0514-2042_Run_1_f87ca"
        epoch = 300
    config_manager = ConfigManager(f"src/models/checkpoints/config_{run_name}.yaml")
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
    checkpoint = torch.load(f"src/models/checkpoints/model_epoch_{epoch}_{run_name}.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded latest {part} model from epoch {checkpoint['epoch']}")

    return model, config


def get_predictions(model, batch, device):
    """
    Get ground truth data and model predictions from a dataset batch.
    """
    images, targets, uids = batch

    h = images.shape[-2]
    images = images.to(device)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        pred_ldmks = (outputs["landmarks"][..., :2] * h).cpu()
        pred_std = (torch.exp(0.5 * outputs["landmarks"][..., 2]) * h).cpu()
        pred_pose = rotation_6d_to_matrix(outputs["pose"]).cpu()  # [B, J, 3, 3]
        pred_shape = outputs["shape"].cpu() if "shape" in outputs else None

    images = denormalize(images).cpu()  # [B, 3, H, W]

    ldmks = targets["landmarks"] * h
    translation = targets["translation"]
    gt_shape = targets["shape"]
    gt_pose = targets["pose"]

    return (
        images,
        uids,
        translation,
        ldmks,
        gt_shape,
        gt_pose,
        pred_ldmks,
        pred_std,
        pred_shape,
        pred_pose,
    )


def get_full_shape_and_pose(pred_shape, pred_pose, gt_shape, gt_pose):
    """Use ground-truth values for any missing pose and shape parameters"""
    if pred_shape is None:  # Hand model
        full_shape = gt_shape
        full_pose = torch.cat([gt_pose[:, :22], pred_pose, gt_pose[:, 37:]], dim=1)
    else:  # Body model
        full_shape = torch.cat([pred_shape, gt_shape[:, -6:]], dim=1)
        full_pose = torch.cat([gt_pose[:, 0].unsqueeze(1), pred_pose, gt_pose[:, 22:]], dim=1)

    return full_shape, full_pose


def get_val_dataloader(config, data_root, meta_file, batch_size=16, mode="val"):
    """
    Initialize validation dataloader that was used for training.
    """
    body = "body" in meta_file
    full_length = 95575 if body else 99200

    # Create dataset splits
    val_size = int(full_length * config.val_ratio)
    test_size = int(full_length * config.test_ratio)
    train_size = full_length - val_size - test_size

    all_indices = torch.randperm(full_length, generator=torch.Generator().manual_seed(config.seed))
    val_indices = all_indices[train_size : train_size + val_size]

    val_set = SynDataset(
        img_dir=data_root,
        metadata=meta_file,
        aug=config.aug,
        indices=val_indices,
        mode=mode,
        device=config.device,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return val_loader


if __name__ == "__main__":
    part = "body"  # Change to "hand" for hand model

    data_root = f"./data/raw/synth_{part}"
    meta_file = f"./data/annotations/{part}_meta.pkl"

    model, config = load_model(part)

    val_loader = get_val_dataloader(config, data_root, meta_file)
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
        pred_pose,
    ) = get_predictions(model, batch, config.device)

    # Concatenated ground truth shape parameters and pose rotation matrices
    full_shape, full_pose = get_full_shape_and_pose(pred_shape, pred_pose, gt_shape, aa2rot(gt_pose))
