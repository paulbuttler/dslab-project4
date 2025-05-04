import torch
from models.model import MultiTaskDNN
from utils.config import ConfigManager
from datasets.transforms import denormalize
from utils.rottrans import rotation_6d_to_matrix, rot2aa
from datasets.synth_dataset import SynDataset
from torch.utils.data import DataLoader


def load_model(run_name) -> MultiTaskDNN:
    """
    Load the model from a checkpoint.
    """
    config_manager = ConfigManager(f"./src/models/checkpoints/config_{run_name}.yaml")
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

    return model, config

def get_predictions(model, batch, device):
    """
    Get ground truth data and model predictions from a dataset batch.
    """
    images, targets, uids = batch

    h = images.shape[-1]
    images = images.to(device)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        pred_ldmks = (outputs["landmarks"][..., :2] * h).cpu()
        pred_std = (torch.exp(0.5 * outputs["landmarks"][..., 2]) * h).cpu()
        pred_pose = outputs["pose"].cpu()
        pred_shape = outputs["shape"].cpu()

        pred_pose = rotation_6d_to_matrix(pred_pose)  # [B, 21, 3, 3]
        pred_pose = rot2aa(pred_pose)  # [B, 21, 3]

    images = denormalize(images).cpu()  # [B, 3, H, W]

    ldmks = targets["landmarks"] * h
    global_orient = targets["pose"][:, 0]
    body_pose = targets["pose"][:, 1:22]
    left_hand_pose = targets["pose"][:, 22:37]
    right_hand_pose = targets["pose"][:, 37:]
    shape = targets["shape"]

    return images, uids, ldmks, global_orient, body_pose, left_hand_pose, right_hand_pose, shape, pred_ldmks, pred_std, pred_pose, pred_shape

def get_val_dataloader(config, data_root, meta_file, batch_size=16):
    """
    Initialize validation dataloader that was used for training.
    """
    full_length = 95575

    # Create dataset splits
    val_size = int(full_length * config.val_ratio)
    test_size = int(full_length * config.test_ratio)
    train_size = full_length - val_size - test_size

    all_indices = torch.randperm(full_length, generator=torch.Generator().manual_seed(config.seed))
    val_indices = all_indices[train_size:train_size + val_size]

    val_set = SynDataset(
        img_dir=data_root,
        metadata=meta_file,
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


if __name__ == "__main__":
    run_name = "0503-2012_Run_2_cont_74664"
    data_root = "./data/raw/synth_body"
    meta_file = "./data/annotations/body_meta.pkl.gz"

    model, config = load_model(run_name)
    
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
        shape,
        pred_ldmks,
        pred_std,
        pred_pose,
        pred_shape,
    ) = get_predictions(model, batch, config.device)