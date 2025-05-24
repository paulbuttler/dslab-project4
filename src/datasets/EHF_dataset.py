import os
import torch
import pickle
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision.io import decode_image
from torch.utils.data import Dataset
from utils.rottrans import rot2aa
from utils.roi import compute_roi

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EHF_Dataset(Dataset):
    """
    EHF Dataset for 3D human pose and shape estimation.
    Data folder structure:
    EHF
    ├── XX_2Djnt.json (OpenPose 2D joints)
    ├── XX_img.jpg
    ├── smplh
    │   ├── XX_align.pkl (SMPLH model parameters converted from SMPL-X)
    └── ...

    Where XX is the index of the sample (01, 02, ..., 99).
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.roi_extractor = YOLO("yolov8m.pt")

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        idx = idx + 1
        img_file = self.data_dir / f"{idx:02d}_img.jpg"
        joints_file = self.data_dir / f"{idx:02d}_2Djnt.json"
        meta_file = self.data_dir / "smplh" / f"{idx:02d}_align.pkl"

        with open(meta_file, "rb") as f:
            metadata = pickle.load(f)

        # Extract pose and shape
        shape = metadata["betas"][0].cpu().detach()
        pose = metadata["full_pose"][0].cpu().detach()
        pose = rot2aa(pose)
        transl = metadata["transl"].cpu().detach() if metadata["transl"] is not None else torch.zeros(3)

        img = decode_image(img_file).float() / 255.0

        # Get the bounding box from the YOLO model
        results = self.roi_extractor.predict(img_file, classes=[0], verbose=False)[0]

        # Extract highest confidence bbox corners (xyxy: [x1, y1, x2, y2])
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            max_idx = np.argmax(confs)
            bbox_corners = boxes[max_idx]

            # Add margin and make square
            x1, y1, x2, y2 = bbox_corners
            w = x2 - x1
            h = y2 - y1
            margin = 0.05  # 5% margin

            # Expand bbox by margin
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            side = max(w, h) * (1 + margin)
            new_x1 = cx - side / 2
            new_y1 = cy - side / 2
            new_x2 = cx + side / 2
            new_y2 = cy + side / 2

            roi = np.array([new_x1, new_y1, new_x2, new_y2])

        else:
            # If no bounding box is detected, use OpenPose joints for ROI computation
            with open(joints_file, "r") as f:
                joint2d_data = json.load(f)
            joints2d = np.asarray(joint2d_data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)
            joints2d = joints2d[~np.all(joints2d == 0, axis=1)]
            joints2d = joints2d[:, :2]
            roi = compute_roi(joints2d, None, margin=0.3, input_size=1600)

        target = {
            "pose": pose,
            "shape": shape,
            "translation": transl,
            "roi": torch.from_numpy(roi).float(),
            "cam_int": torch.tensor(
                [[1498.22426237, 0.0, 790.263706], [0.0, 1498.22426237, 578.90334], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
        }

        return img, target, idx
