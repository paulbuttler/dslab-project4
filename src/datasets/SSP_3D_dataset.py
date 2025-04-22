import os
import torch
import gzip
import pickle
from pathlib import Path
import numpy as np
import kornia.augmentation as K
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from datasets.transforms import denormalize, random_roi_transform, apply_roi_transform, AppearanceAugmentation
from utils.rottrans import aa2rot, matrix_to_rotation_6d

class SSP_3D_Dataset(Dataset):

    def __init__(self, 
                 img_dir:str, 
                 labels_dir:str, 
                 device="cuda"):
        self.img_dir = img_dir

        # Data
        data = np.load(os.path.join(labels_dir, 'labels.npz'))

        self.image_fnames = data['fnames']
        self.body_shapes = data['shapes']
        self.body_poses = data['poses']

        self.device = device
        

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        if torch.is_tensor(index):
            index = index.tolist()

        fname = self.image_fnames[index]
        img_path = os.path.join(self.images_dir, fname)
        img = (decode_image(img_path).float().to(self.device) / 255.0).unsqueeze(0)

        shape = torch.from_numpy(self.body_shapes[index]).to(device=self.device, dtype=torch.float32)
        pose = torch.from_numpy(self.body_poses[index]).to(device=self.device, dtype=torch.float32)

        target = {
            'pose': pose,
            'shape': shape,
        }

        return img.squeeze(0), target


# Test the dataset
if __name__ == "__main__":
    dataset = SSP_3D_Dataset(
        img_dir=Path("data/raw/SSP_3D/images"),
        labels_dir="data/raw/SSP_3D/labels.npz",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    print(f"Number of samples: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    images, labels = next(iter(dataloader))
    print(f"Image shape: {images.shape}")

