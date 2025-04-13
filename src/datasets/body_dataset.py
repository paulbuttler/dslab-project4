import torch
import gzip
import pickle
from pathlib import Path
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from transforms import random_roi_transform, apply_roi_transform, AppearanceAugmentation


class BodyDataset(Dataset):

    def __init__(self, img_dir, body_meta_dir, mode="train", device="cuda"):
        self.img_dir = img_dir
        with gzip.open(body_meta_dir, "rb") as f:
            self.body_meta = pickle.load(f)
        self.uids = list(self.body_meta.keys())
        self.mode = mode
        self.device = device
        self.appearance_aug = AppearanceAugmentation()

    def __len__(self):
        return len(self.body_meta)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        img_path = self.img_dir / f"img_{uid}.jpg"

        img = (decode_image(img_path).float().to(self.device) / 255.0).unsqueeze(0)

        kp2d = torch.tensor(
            self.body_meta[uid]["ldmks_2d"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        roi = torch.tensor(
            self.body_meta[uid]["roi"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        if self.mode == "train":
            img, kp2d = random_roi_transform(img, kp2d, roi, "train")
            img = self.appearance_aug(img)
        else:
            img, kp2d = apply_roi_transform(img, kp2d, roi, "test")
        pose = self.body_meta[uid]["pose"]
        shape = self.body_meta[uid]["shape"]

        return img.squeeze(0), kp2d.squeeze(0), pose, shape, uid

# Test the dataset
if __name__ == "__main__":
    dataset = BodyDataset(
        img_dir=Path("data/raw/synth_body"),
        body_meta_dir="data/annotations/body_meta.pkl.gz",
        mode="train",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dataloader = DataLoader(dataset, batch_size=15, shuffle=False)

    images, kp2d, pose, shape, uid = next(iter(dataloader))
    print(f"Image shape: {images.shape}, Keypoints shape: {kp2d.shape}")

    # Visualize a batch of images
    import cv2
    import numpy as np

    img_list = []
    for i in range(15):
        img = images[i].permute(1, 2, 0).numpy()  # [H, W, C], float32
        img = (img * 255).astype(np.uint8)  # [0, 255], uint8
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw keypoints
        # for x, y in kp2d[i]:
        #     x, y = int(x.item()), int(y.item())
        #     cv2.circle(img, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0  # Back to [C, H, W], float
        img_list.append(img)

    img_tensor = torch.stack(img_list)  # [B, C, H, W]
    grid = make_grid(img_tensor, nrow=5, padding=2)

    # 5. Convert to numpy and show using OpenCV
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    grid_bgr = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)

    cv2.imshow("Batch Grid", grid_bgr)
    cv2.waitKey(0)
