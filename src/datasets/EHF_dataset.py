import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.initialization import initial_pose_estimation
from utils.roi import compute_roi
import os
import torch
import gzip
import pickle
import numpy as np
from utils.rottrans import rot2aa
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from aitviewer.models.smpl import SMPLLayer  # type: ignore
from utils.evaluation import get_similarity
from torchvision.transforms import functional as F
from scipy.spatial import procrustes
from models.load import load_model
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class EHF_Dataset(Dataset):
    """
    EHF Dataset
    Data folder structure:
    ├── EHF
    │   ├── XX_2Djnt.json
    │   ├── XX_img.jpg
    │   ├── smplh
    │   │   ├── XX_align.pkl

    Where smplh contains the converted smplx data
    """


    def __init__(self, 
                 data_dir:str, 
                 device="cuda"):
        self.img_dir = data_dir

        # Data
        self.img_paths = [os.path.join(self.img_dir, img_fn)
                    for img_fn in os.listdir(self.img_dir)
                    if img_fn.endswith('.jpg') and not img_fn.startswith('.')
        ]
        print(len(self.img_paths))
        self.device = device
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = self.img_paths[index]
        print("Load", img_path)
        # img = (decode_image(img_path).float().to(self.device) / 255.0).unsqueeze(0)
        # Load image as tensor
        img = decode_image(img_path).float().to(self.device) / 255.0

        

        base_path = os.path.split(img_path)[0]

        img_fn = os.path.split(img_path)[1]
        img_fn, _ = os.path.splitext(os.path.split(img_path)[1])
        txt_index = img_fn.split("_")[0]

        frame_path = os.path.join(base_path, "smplh", f"{txt_index}_align.pkl")
        frame_data = np.load(frame_path, allow_pickle=True)
        
        joint2d_path = os.path.join(base_path, f"{txt_index}_2Djnt.json")
        with open(joint2d_path, "r") as f:
            joint2d_data = json.load(f)

        roi = None
        for idx, person_data in enumerate(joint2d_data['people']):
            body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
            joint2d = body_keypoints.reshape(-1, 3)
            roi = np.array(compute_roi(joint2d[:, :2], None, margin=0.12, input_size=1600), dtype=np.float32)
            print("roi?????????????????????????????????????", roi)
            


        shape = frame_data['betas'].reshape(-1)
        body_pose = frame_data['body_pose'].reshape(1, -1)
        global_orient = frame_data['global_orient'].reshape(1, -1)

        body_pose = torch.cat([global_orient, body_pose], dim=1)
        body_pose = body_pose.reshape(-1, 3, 3)
        body_pose = rot2aa(body_pose).reshape(-1)
        pose = torch.cat([body_pose, torch.zeros(52*3 - body_pose.numel())])
        pose = pose.view(-1, 3)

        vertices = frame_data['vertices'].reshape(1, -1, 3).to(device=self.device)
        print("\n---------------------------------------------------------------------------")
        print("Info", body_pose.shape, shape.shape, global_orient.shape, vertices.shape, roi)
        target = {
            'pose': pose,
            'shape': shape,
            'vertices': vertices,
            # 'global_orient': global_orient,
            "landmarks": [],
            "translation": (0,0,0),
            "joints2D": joint2d,
        }

        return img.squeeze(0), roi, target, index


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = EHF_Dataset(data_dir="data/EHF")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=10)
    body_model, body_config = load_model("body")
    hand_model, hand_config = load_model("hand")

    MPVPE_ls = []
    PA_MPVPE_ls = []
    PA_MPJPE_ls = []
    for i, batch in enumerate(dataloader):
        images, roi, targets, uids = batch
        print("REACHED", device)
        ldmks, std, pred_pose, pred_shape = initial_pose_estimation(images, roi, body_model, hand_model, device)
        MPVPE, PA_MPVPE, PA_MPJPE, params = get_similarity(smpl_layer, pred_shape, targets['shape'], pred_pose, targets['pose'], measure_type="body", num_joints=22)
        #print("MPVPE", MPVPE)
        #print("PA_MPVPE", PA_MPVPE)
        #print("PA_MPJPE", PA_MPJPE)
        MPVPE_ls.append(MPVPE)
        PA_MPVPE_ls.append(PA_MPVPE)
        PA_MPJPE_ls.append(PA_MPJPE)
    print("MPVPE", np.mean(MPVPE_ls))
    print("PA_MPVPE", np.mean(PA_MPVPE_ls))
    print("PA_MPJPE", np.mean(PA_MPJPE_ls))
        

