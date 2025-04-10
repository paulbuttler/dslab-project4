import torch
from torch.utils.data import Dataset

class BodyDataset(Dataset):
    def __init__(self, data, labels, transforms=None, mode="train"):
        self.data = data
        self.labels = labels
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.mode == "train" and self.transforms:
            x,y = self.transforms(x, y)
        return x, y