import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))


import os
import torch
import gzip
import pickle
import numpy as np
import kornia.augmentation as K
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from aitviewer.renderables.smpl import SMPLSequence  # type: ignore
from utils.evaluation import compute_pve_neutral_pose_scale_corrected
from models.load import load_model
from utils.rottrans import rot2aa
from utils.initialization import initial_pose_estimation
from aitviewer.models.smpl import SMPLLayer  # type: ignore

class SSP_3D_Dataset(Dataset):

    def __init__(self, 
                 img_dir:str, 
                 labels_dir:str, 
                 device="cuda"):
        self.img_dir = img_dir

        # Data
        data = np.load(os.path.join(labels_dir, 'labels.npz'))

        print("Available keys in labels.npz:", list(data.keys()))
        self.image_fnames = data['fnames']
        self.body_shapes = data['shapes']
        self.body_poses = data['poses']
        self.bbox_centres = data['bbox_centres']
        self.bbox_whs = data['bbox_whs']
        self.joints_2d = data['joints2D']
        self.gender = data['genders']
        self.cam_trans = data['cam_trans']

        self.device = device
        

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        fname = self.image_fnames[index]
        img_path = os.path.join(self.img_dir, fname)
        img = (decode_image(img_path).float().to(self.device) / 255.0).unsqueeze(0)

        shape_params = torch.from_numpy(self.body_shapes[index]).to(device=self.device, dtype=torch.float32)
        pose_param = torch.from_numpy(self.body_poses[index]).to(device=self.device, dtype=torch.float32)
        pose_param = pose_param[:-6]
        print("#######",  pose_param.shape)

        # Calculate the four corners of the bounding box (roi)
        cx, cy = self.bbox_centres[index]
        print(self.bbox_centres[index], self.bbox_whs[index])
        size = self.bbox_whs[index]
        x1 = cx - size / 2
        y1 = cy - size / 2
        x2 = cx + size / 2
        y2 = cy + size / 2
        
        roi = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=self.device)
        joints_2d = torch.from_numpy(self.joints_2d[index]).to(device=self.device, dtype=torch.float32)


        # Convert pose
        cam_trans = self.cam_trans[index]
        # Create a 4x4 transformation matrix with identity rotation and cam_trans as translation
        transformation_matrix = torch.eye(4, dtype=torch.float32, device=self.device)
        transformation_matrix[:3, 3] = torch.from_numpy(cam_trans).to(self.device, dtype=torch.float32)

        root_pose = torch.zeros(3, device=self.device)# pose_param[0:3]
        print("------------------------", transformation_matrix)
        

        #pose = torch.cat([root_pose, pose_param[3:], torch.zeros(52*3 - pose_param.numel(), device=self.device)])
        pose = torch.cat([root_pose, pose_param[3:], torch.zeros(52*3 - pose_param.numel(), device=self.device)])
        pose = pose.view(-1, 3)
        print("£££££££££££££££££££££££££", self.gender[index])
        target = {
            'pose': pose.to(device=self.device),
            'shape': shape_params.to(device=self.device),
            
            "landmarks": joints_2d.to(device=self.device),
            "translation": (0,0,0),
            "transform_matrix": transformation_matrix,
        }

        print(img_path)
        return img.squeeze(0), roi, target, index
    




# Test the dataset
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    dataset = SSP_3D_Dataset(
        img_dir=Path("data/SSP_3D/images"),
        labels_dir="data/SSP_3D",
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral")
    body_model, body_config = load_model("body")
    hand_model, hand_config = load_model("hand")

    num_samples = len(dataset)
    pve_t_sc = []
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        # (
        #     images,
        #     uids,
        #     translation,
        #     ldmks,
        #     gt_shape,
        #     gt_pose,
        #     pred_ldmks,
        #     pred_std,
        #     pred_shape,
        #     pred_pose,
        # ) = get_predictions(model, batch, config.device)
        images, roi, targets, uids = batch
        gt_pose = targets["pose"]
        gt_shape = targets["shape"]
        ldmks, std, pred_pose, pred_shape = initial_pose_estimation(images, roi, body_model, hand_model, device)

        pve_t_sc.append(compute_pve_neutral_pose_scale_corrected(smpl_layer, pred_shape, gt_shape.to(device)))

        #pred_p = get_3D_positions(smpl_layer, pred_full_pose)
        #gt_p = get_3D_positions(smpl_layer, gt_pose)
        
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

    
    print("SSP-3D PVE-T-SC:", np.mean(pve_t_sc) * 1000 ) # in mm

