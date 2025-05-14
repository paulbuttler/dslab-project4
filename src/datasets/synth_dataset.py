import os
import torch
import joblib
from pathlib import Path
import kornia.augmentation as K
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from datasets.transforms import denormalize, random_roi_transform, apply_roi_transform, AppearanceAugmentation


class SynDataset(Dataset):

    def __init__(self, img_dir: str, metadata, aug: dict, indices=None, mode="train", device="cuda"):

        self.device = device
        self.img_dir = img_dir
        self.train = mode == "train"
        self.crop_size = aug["crop_size"]

        if self.train:
            self.roi_aug = aug["roi"]
            self.appearance_aug = AppearanceAugmentation(aug["appearance"]).to(device)

        self.normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])
        ).to(device)

        if isinstance(metadata, str):
            metadata = joblib.load(metadata)

        all_uids = list(metadata.keys())

        self.uids = [all_uids[i] for i in indices] if indices is not None else all_uids
        self.metadata = {uid: metadata[uid] for uid in self.uids} if indices is not None else metadata

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        img_path = os.path.join(self.img_dir, f"img_{uid}.jpg")

        img = decode_image(img_path).float() / 255.0
        kp2d = self.metadata[uid]["ldmks_2d"]
        roi = self.metadata[uid]["roi"]
        pose = self.metadata[uid]["pose"]  # Local axis angle representation
        shape = self.metadata[uid]["shape"]
        translation = self.metadata[uid]["translation"]

        # Move to device for augmentation
        img = img.to(self.device).unsqueeze(0)
        kp2d = torch.from_numpy(kp2d).to(self.device, dtype=torch.float32).unsqueeze(0)
        roi = torch.from_numpy(roi).to(self.device, dtype=torch.float32).unsqueeze(0)

        if self.train:
            img, kp2d = random_roi_transform(img, kp2d, roi, self.roi_aug, self.crop_size)
            img = self.appearance_aug(img)
        else:
            img, kp2d = apply_roi_transform(img, kp2d, roi, "test", self.crop_size)

        # Normalize with ImageNet stats
        img = self.normalize(img)

        # Normalize 2D landmark coordinates to [0, 1]
        kp2d = kp2d / img.shape[-1]

        target = {
            "landmarks": kp2d.squeeze(0).cpu(),
            "pose": torch.from_numpy(pose).float(),
            "shape": torch.from_numpy(shape).float(),
            "translation": torch.from_numpy(translation).float(),
        }

        return img.squeeze(0).cpu(), target, uid


# Test dataset and augmentation
if __name__ == "__main__":
    body = False
    if body:
        dataset = SynDataset(
            img_dir=Path("data/raw/synth_body"),
            metadata="data/annotations/body_meta.pkl",
            aug={
                "crop_size": 256.0,
                "roi": {
                    "angle": 25.0,
                    "scale": [0.05, 0.10],
                    "trans": 0.06,
                },
                "appearance": {
                    "probs": {
                        "motion_blur": 0.2,
                        "brightness": 0.4,
                        "contrast": 0.4,
                        "hue_saturation": 0.3,
                        "grayscale": 0.1,
                        "jpeg": 0.2,
                        "iso_noise": 0.2,
                        "cutout": 0.0,
                    }
                },
            },
            mode="train",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        dataset = SynDataset(
            img_dir=Path("data/raw/synth_hand"),
            metadata="data/annotations/hand_meta.pkl",
            aug={
                "crop_size": 128.0,
                "roi": {
                    "angle": 15.0,
                    "scale": [0.04, 0.08],
                    "trans": 0.04,
                },
                "appearance": {
                    "probs": {
                        "motion_blur": 0.2,
                        "brightness": 0.4,
                        "contrast": 0.4,
                        "hue_saturation": 0.3,
                        "grayscale": 0.1,
                        "jpeg": 0.2,
                        "iso_noise": 0.0,
                        "cutout": 0.2,
                    }
                },
            },
            mode="train",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    dataloader = DataLoader(dataset, batch_size=18, shuffle=False)

    print(f"Number of samples: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    images, labels, uids = next(iter(dataloader))
    print(f"Image shape: {images.shape}")

    import cv2

    images = denormalize(images)  # [B, C, H, W]
    kp2d = (labels["landmarks"] * images.shape[-1]).cpu().numpy()

    # Visualize a batch of images
    grid = make_grid(images, nrow=6, padding=2)  # [C, H', W']
    grid_np = grid.permute(1, 2, 0).cpu().numpy()  # [H', W', C]

    grid_bgr = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)

    cv2.imshow("Batch Grid", grid_bgr)
    cv2.waitKey(0)

    # Visualize a single image with 2d landmarks
    idx = 4

    img = images[idx].permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Draw keypoints
    for x, y in kp2d[idx]:
        x, y = int(x.item()), int(y.item())
        cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.imshow("Image with Landmarks", img)
    cv2.waitKey(0)
