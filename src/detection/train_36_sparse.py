import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gzip
import pickle
import timm
import wandb
import matplotlib.pyplot as plt
import io
from datasets.transforms import random_roi_transform, apply_roi_transform, AppearanceAugmentation
import kornia.augmentation as K
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.patches as patches

# -------------------- Model --------------------
class LightweightLandmarkDNN(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, num_landmarks=36, backbone_feat_dim=512, mlp_head_hidden_dim=256):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=False,
            num_classes=backbone_feat_dim,
        )
        self.landmark_head = nn.Sequential(
            nn.Linear(backbone_feat_dim, mlp_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_head_hidden_dim, num_landmarks * 3)  # (mu_x, mu_y, logvar)
        )
        self.num_landmarks = num_landmarks

    def forward(self, x):
        features = self.backbone(x)
        B = features.shape[0]
        landmarks = self.landmark_head(features).reshape(B, self.num_landmarks, 3)
        return {"landmarks": landmarks}

# -------------------- Dataset --------------------
class Landmark36WithROIAugDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, landmark_file, meta_file, uids=None, train=True, device='cuda'):
        self.img_dir = img_dir
        self.train = train
        self.device = device
        self.appearance_aug = AppearanceAugmentation().to(device) if train else None
        self.normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        ).to(device)

        with gzip.open(landmark_file, 'rb') as f:
            self.ldmks_dict = pickle.load(f)

        with gzip.open(meta_file, 'rb') as f:
            self.meta = pickle.load(f)

        self.uids = uids if uids is not None else list(self.ldmks_dict.keys())

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        img_path = os.path.join(self.img_dir, f"img_{uid}.jpg")

        image = read_image(img_path).float() / 255.0
        landmarks = torch.tensor(self.ldmks_dict[uid], dtype=torch.float32)
        roi = torch.tensor(self.meta[uid]["roi"], dtype=torch.float32)

        image = image.to(self.device).unsqueeze(0)
        landmarks = landmarks.to(self.device).unsqueeze(0)
        roi = roi.to(self.device).unsqueeze(0)

        if self.train:
            image, landmarks = random_roi_transform(image, landmarks, roi, mode="train")
            image = self.appearance_aug(image)
        else:
            image, landmarks = apply_roi_transform(image, landmarks, roi, mode="test")

        image = self.normalize(image)
        landmarks = landmarks / image.shape[-1]

        return image.squeeze(0).cpu(), landmarks.squeeze(0).cpu()

# -------------------- Training Script --------------------
def train():
    wandb.init(project="lightweight-dnn-landmarks", config={
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "backbone": "resnet18",
        "val_split": 0.1
    })

    config = wandb.config

    img_dir = "data/raw/synth_body"
    landmark_file = "data/annotations/body_ldmks_roi.pkl.gz"
    meta_file = "data/annotations/body_meta.pkl.gz"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with gzip.open(landmark_file, 'rb') as f:
        ldmks_dict = pickle.load(f)
    with gzip.open(meta_file, 'rb') as f:
        meta = pickle.load(f)

    all_uids = list(ldmks_dict.keys())
    val_size = int(len(all_uids) * config.val_split)
    train_size = len(all_uids) - val_size

    indices = torch.randperm(len(all_uids))
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    train_uids = [all_uids[i] for i in train_indices]
    val_uids = [all_uids[i] for i in val_indices]

    train_dataset = Landmark36WithROIAugDataset(img_dir, landmark_file, meta_file, train_uids, train=True, device=device)
    val_dataset = Landmark36WithROIAugDataset(img_dir, landmark_file, meta_file, val_uids, train=False, device=device)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = LightweightLandmarkDNN()
    model = model.to(device)

    def probabilistic_landmark_loss(pred, target):
        pred_mu = pred[..., :2]
        log_var = pred[..., 2]
        sq_diff = (pred_mu - target).pow(2).sum(dim=-1)
        return (log_var + 0.5 * sq_diff * torch.exp(-log_var)).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)["landmarks"]
            loss = probabilistic_landmark_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)["landmarks"]
                loss = probabilistic_landmark_loss(outputs, targets)
                val_loss += loss.item()

                # ---- Log sample visualizations every epoch ----
                num_vis = min(4, images.size(0))
                vis_images = []
                for i in range(num_vis):
                    img = images[i].cpu()
                    pred_pts = outputs[i, :, :2].cpu() * img.shape[-1]
                    gt_pts = targets[i, :, :2].cpu() * img.shape[-1] if targets.dim() == 3 else targets[i].cpu() * img.shape[-1]
                    logvar = outputs[i, :, 2].cpu()
                    std = torch.exp(0.5 * logvar)

                    img_np = TF.to_pil_image(img)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(img_np)
                    ax.scatter(gt_pts[:, 0], gt_pts[:, 1], c='lime', s=10, label='GT')
                    ax.scatter(pred_pts[:, 0], pred_pts[:, 1], c='red', s=10, label='Pred')

                    # Draw confidence circles
                    for (x, y), s in zip(pred_pts, std):
                        radius = 20 * s.item()
                        circle = patches.Circle((x, y), radius=radius, edgecolor='blue', facecolor='blue', alpha=0.2)
                        ax.add_patch(circle)

                    ax.axis('off')
                    ax.legend()

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    img_pil = Image.open(buf).convert("RGB")
                    vis_images.append(wandb.Image(img_pil, caption=f"Epoch {epoch+1} - Sample {i+1}"))
                    plt.close()

                wandb.log({"Val Predictions": vis_images}, step=epoch)
                break

        val_loss /= len(val_loader)
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "lightweight_dnn_36.pth")
    wandb.finish()

if __name__ == "__main__":
    train()
