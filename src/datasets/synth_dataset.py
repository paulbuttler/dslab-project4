import os
import torch
from pathlib import Path
import kornia.augmentation as K
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from datasets.transforms import denormalize, random_roi_transform, apply_roi_transform, AppearanceAugmentation

class SynDataset(Dataset):

    def __init__(
        self,
        img_dir: str,
        body_meta,  # can be a path or a dict
        indices=None,
        mode="train",
        device="cuda",
    ):
        self.img_dir = img_dir
        self.train = mode == "train"
        self.device = device
        self.appearance_aug = AppearanceAugmentation().to(device) if self.train else None
        self.normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        ).to(device)

        if isinstance(body_meta, str):
            body_meta = torch.load(body_meta, map_location="cpu")

        all_uids = list(body_meta.keys())

        self.uids = [all_uids[i] for i in indices] if indices is not None else all_uids
        self.body_meta = (
            {uid: body_meta[uid] for uid in self.uids}
            if indices is not None
            else body_meta
        )

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        img_path = os.path.join(self.img_dir, f"img_{uid}.jpg")

        img = (decode_image(img_path).float() / 255.0).unsqueeze(0)
        kp2d = self.body_meta[uid]["ldmks_2d"].unsqueeze(0)
        roi = self.body_meta[uid]["roi"].unsqueeze(0)
        pose = self.body_meta[uid]["pose"]  # Local axis angle representation!!
        shape = self.body_meta[uid]["shape"][:10]  # only first 10 elements

        # Move to device for augmentation
        img = img.to(self.device)
        kp2d = kp2d.to(self.device)
        roi = roi.to(self.device)

        if self.train and self.appearance_aug is not None:
            img, kp2d = random_roi_transform(img, kp2d, roi, "train")
            img = self.appearance_aug(img)
        else:
            img, kp2d = apply_roi_transform(img, kp2d, roi, "test")

        # Normalize with ImageNet stats
        img = self.normalize(img)

        # Normalize 2D landmark coordinates to [0, 1]
        kp2d = kp2d/img.shape[-1]

        target = {
            "landmarks": kp2d.squeeze(0),
            "pose": pose.to(self.device),
            "shape": shape.to(self.device),
        }

        return img.squeeze(0), target, uid

# Test dataset and augmentation
if __name__ == "__main__":
    dataset = SynDataset(
        img_dir=Path("data/raw/synth_body"),
        body_meta="data/annotations/body_meta.pt",
        mode="train",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dataloader = DataLoader(dataset, batch_size=18, shuffle=False)

    print(f"Number of samples: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    images, labels, uids = next(iter(dataloader))
    print(f"Image shape: {images.shape}")

    import cv2

    images = denormalize(images)     # [B, C, H, W]
    kp2d = (labels["landmarks"] * images.shape[-1]).cpu().numpy()

    # Visualize a batch of images
    grid = make_grid(images, nrow=6, padding=2)  # [C, H', W']
    grid_np = grid.permute(1, 2, 0).cpu().numpy()  # [H', W', C]

    grid_bgr = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)

    cv2.imshow('Batch Grid', grid_bgr)
    cv2.waitKey(0)

    # Visualize a single image with 2d landmarks
    idx = 3

    img = images[idx].permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Draw keypoints
    for (x, y) in kp2d[idx]:
        x, y = int(x.item()), int(y.item())
        cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow("Image with Landmarks", img)
    cv2.waitKey(0)
