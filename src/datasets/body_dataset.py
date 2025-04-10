import torch
import gzip
import pickle
from torchvision.io import decode_image
from torch.utils.data import Dataset
from src.datasets.transforms import random_roi_transform
from src.datasets.transforms import AppearanceAugmentation

class BodyDataset(Dataset):

    def __init__(self, img_dir, body_meta_dir, mode="train", device="cuda"):
        self.img_dir = img_dir
        with gzip.open(body_meta_dir, "rb") as f:
            self.body_meta = pickle.load(f)
        self.uids = list(self.body_meta.keys())
        self.mode = mode
        self.device = device
        self.appearance_aug = AppearanceAugmentation().to(device)

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
            img, kp2d = random_roi_transform(
                img, kp2d, roi, crop_size=(256, 256), device=self.device
            )
            img, kp2d = self.appearance_aug(img, kp2d)
        pose = self.body_meta[uid]["pose"]
        shape = self.body_meta[uid]["shape"]

        return img.squeeze(0), kp2d.squeeze(0), pose, shape, uid
