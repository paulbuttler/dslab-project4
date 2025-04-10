import torch
import gzip
import pickle
from pathlib import Path
from torchvision.io import decode_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transforms import random_roi_transform, apply_roi_transform
from transforms import AppearanceAugmentation

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
            img, kp2d = random_roi_transform(img, kp2d, roi, crop_size=256.0)
            # img, kp2d = self.appearance_aug(img, kp2d)
        else:
            img, kp2d = apply_roi_transform(img, kp2d, roi, crop_size=256.0)
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
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    img, kp2d, pose, shape, uid = next(iter(dataloader))

    print(
        f"Image shape: {img.shape}, Keypoints shape: {kp2d.shape}, Pose: {pose.shape}, Shape: {shape.shape}, UID: {uid}"
    )

    # Visualize the first image and keypoints
    import cv2

    idx = 2

    img = img[idx].permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Visualize ROI and landmarks on the image
    for x, y in kp2d[idx].cpu().numpy():
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.imshow("Square ROI", img)
    cv2.waitKey(0)
