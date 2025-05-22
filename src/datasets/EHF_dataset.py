import sys
from pathlib import Path

from utils.roi import compute_roi


sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import torch
import gzip
import pickle
import numpy as np
from utils.rottrans import rot2aa
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from aitviewer.models.smpl import SMPLLayer  # type: ignore
from utils.evaluation import evaluate_model, get_full_shape_and_pose, get_mesh_vertices, get_obj_file_vertices
from torchvision.transforms import functional as F
from scipy.spatial import procrustes
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class EHF_Dataset(Dataset):

    def __init__(self, 
                 img_dir:str, 
                 device="cuda"):
        self.img_dir = img_dir

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

            # Center crop to square
            # h, w = img.shape[1], img.shape[2]
            # min_dim = min(h, w)
            # top = (h - min_dim) // 2
            # left = (w - min_dim) // 2
            # cropped_img = img[:, top:top+min_dim, left:left+min_dim]

            # # Resize to 512x512
            # cropped_img = F.resize(cropped_img, [512, 512])
            # roi = compute_roi(joint2d[:, :2], cropped_img.cpu().numpy().transpose(1, 2, 0).copy())
            roi = np.array(compute_roi(joint2d[:, :2], None), dtype=np.float32)


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
        }

        return img.squeeze(0), roi, target, index


if __name__ == "__main__":
    dataset = EHF_Dataset(img_dir="data/EHF", labels_dir="data/labels")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral")

    model, config = load_model("0503-2012_Run_2_cont_74664")

    pve_t_sc = []
    for i, batch in enumerate(dataloader):

        (
            images,
            uids,
            translation,
            ldmks,
            gt_shape,
            gt_pose,
            pred_ldmks,
            pred_std,
            pred_shape,
            pred_pose,
        ) = get_predictions(model, batch, config.device)

        pred_full_shape, pred_full_pose = get_full_shape_and_pose(pred_shape.to(config.device), pred_pose.to(config.device), gt_shape.to(config.device), gt_pose.to(config.device))

        print("bbb",pred_pose.shape, gt_pose.shape, pred_shape.shape, gt_shape.shape, pred_full_shape.shape, pred_full_pose.shape)
        pred_v = get_mesh_vertices(smpl_layer, pred_full_shape, pred_full_pose.cpu().detach())
        gt_v = torch.tensor(batch[1]['vertices'].reshape(-1, 3))

        print("\n---------------------------------------------------------------------------")
        print("pred_v min:", np.min(pred_v.cpu().numpy(), axis=0), "max:", np.max(pred_v.cpu().numpy(), axis=0))
        print("gt_v min:", np.min(gt_v.cpu().numpy(), axis=0), "max:", np.max(gt_v.cpu().numpy(), axis=0))
        pred_v_pa, gt_v_pa, disparity = procrustes(pred_v.cpu().numpy(), gt_v.cpu().numpy())
        
        print("\n---------------------------------------------------------------------------")
        print("pred_v_pa min:", np.min(pred_v_pa, axis=0), "max:", np.max(pred_v_pa, axis=0))
        print("gt_v_pa min:", np.min(gt_v_pa, axis=0), "max:", np.max(gt_v_pa, axis=0))
        print(pred_v_pa.shape, gt_v_pa.shape, disparity)

        print("\n---------------------------------------------------------------------------")
        print("PA-MPVPE", np.mean(np.linalg.norm(pred_v_pa - gt_v_pa, axis=-1)))
        print("---------------------------------------------------------------------------")
        break
        
        # print(pred_full_shape.shape, gt_shape.shape)
        # mse_shape = torch.mean((pred_full_shape.to(config.device)[:10] - gt_shape.to(config.device)) ** 2).item()
        # print("MSE between gt_shape and pred_shape:", mse_shape)

        # mse_pose = torch.mean((pred_p - gt_p) ** 2).item()
        # print("MSE between gt_pose and pred_pose:", mse_pose)

        # similarity = compute_similarity_transform(pred_p.numpy(), gt_p.numpy(), num_joints=24)
        # pred_p_transformed = similarity[0]
        # gt_p_transformed = similarity[1]

        # mse_similarity = torch.mean((pred_p_transformed - gt_p_transformed) ** 2).item()
        # print("MSE between gt_pose and pred_pose after similarity transform:", mse_similarity)


        # mse_pose = 

