import torch
import torch.nn as nn
import cv2
import numpy as np
import warnings
import json
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models.smplx import SMPLHLayer
from models.load import load_model, get_val_dataloader
from inference.initialization import initial_pose_estimation
from human_body_prior.models.vposer_model import VPoser  # type: ignore
from utils.rottrans import aa2rot, matrix_to_rotation_6d, rotation_6d_to_matrix, rot2aa
from utils.config import ConfigManager
from utils.visualize.visualize_data import draw_mesh


def full_model_inference(images, rois, cam_int, body_model, hand_model, device):
    """
    Perform pose and shape estimation including optimization from single-view images given a bounding box ROI and camera intrinsics.
    Args:
        images (Tensor): Batch of unnormalized, uncropped input images, shape [B, 3, H, W].
        rois (Tensor): Region of interest that tightly contains the whole body, shape [B, 4].
        cam_int (Tensor): Camera intrinsics for each image, shape [B, 3, 3].
        body_model (nn.Module): Trained body model.
        hand_model (nn.Module): Trained hand model.
        device (str): Device to run the models on.
    Returns:
        poses (Tensor): Optimized pose parameters, shape [B, 52, 3].
        shapes (Tensor): Optimized shape parameters, shape [B, 16].
        Rs (Tensor): Rotation matrices of extrinsics, shape [B, 3, 3].
        ts (Tensor): Translation vectors of extrinsics, shape [B, 1, 3].
        ---
        Note: Global orientation and translation are absorbed into the camera extrinsics.
    """

    opt_config = ConfigManager("src/inference/optimization_config.yaml").config
    ldmks, std, pose, shape = initial_pose_estimation(images, rois, body_model, hand_model, device)

    pose_list = []
    shape_list = []
    R_list = []
    t_list = []
    for i in range(images.shape[0]):
        pose_optimised, shape_optimized, R, t = optimize_pose_shape(
            images[i], cam_int[i], ldmks[i], std[i], pose[i], shape[i], device, opt_config
        )

        pose_list.append(pose_optimised)
        shape_list.append(shape_optimized)
        R_list.append(R)
        t_list.append(t)

    poses = torch.stack(pose_list, dim=0)
    shapes = torch.stack(shape_list, dim=0)
    Rs = torch.stack(R_list, dim=0)
    ts = torch.stack(t_list, dim=0)

    return poses, shapes, Rs, ts


def optimize_pose_shape(
    image: torch.Tensor,
    K: torch.Tensor,
    ldmks: torch.Tensor,
    std: torch.Tensor,
    pose: torch.Tensor,
    shape: torch.Tensor,
    device: str,
    opt_config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimize shape and pose parameters with 2D landmarks as references.
    Notice: Optimization is case by case, thus this function does NOT support batch input.
    If you hope to speed up, you can call this function for different samples in parallel using multi process.
    ----
    Args:
        image: (3, H, W)
        K: intrinsics, (3, 3)
        ldmks: (n_ldmk, 2)
            Note that ldmk locations are rescaled to original space (e.g., (0, 511))
        std: (n_ldmk,)
            Note that std are still in normalized space (i.e., (0, 1))
        pose: (51, 3)
        shape: (10,)
        device:
        opt_config: Configuration for optimization. Load from optimization_config.yaml
    ----
    Outputs:
        pose_optimized: (52, 3)
            Note that pelvis pose is set to ZERO.
        shape_optimized: (16,)
        R: rotation of extrinsics, (3, 3)
        t: translation of extrinsics, (1, 3)
    """
    ## ldmk indices
    ldmk_indices = torch.tensor(np.load(opt_config.ldmk_indices_file), dtype=torch.long).to(device)
    ## Body Model
    smplh_layer = SMPLHLayer(
        model_path=opt_config.smplh_model_path,
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=16,
        dtype=torch.float32,
    ).to(device)
    ## pose and shape convert
    pose_rot = aa2rot(pose).to(device)
    full_shape = torch.zeros((16), dtype=torch.float32, device=device)
    full_shape[:10] = shape
    ## Current 3D ldmks (Initialization)
    smplh_init = smplh_layer(
        betas=full_shape.unsqueeze(0),
        global_orient=None,
        body_pose=pose_rot[0:21].unsqueeze(0),
        left_hand_pose=pose_rot[21:36].unsqueeze(0),
        right_hand_pose=pose_rot[36:].unsqueeze(0),
        transl=None,
    )
    points_3d_init = smplh_init.vertices[0, ldmk_indices].detach()  # shape: (N, 3)
    ## 2D ldmk reference
    points_2d = ldmks.detach()  # shape: (N, 2)

    ## Using PnP to initialize extrinsics
    ### cv2.solvePnP requires ndarray, but we keep data flow in tensor
    ### so we wrap PnP inside a sub function
    def cal_extrinsics_using_pnp(
        kp3d: torch.Tensor, kp2d: torch.Tensor, intrinsics: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=kp3d.cpu().numpy().astype(np.float32),
            imagePoints=kp2d.cpu().numpy().astype(np.float32),
            cameraMatrix=intrinsics.cpu().numpy().astype(np.float32),
            distCoeffs=np.zeros((4, 1)),  # assume no lens distortion
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not success:
            warnings.warn("solvePnP failure!")
        R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to matrix
        tvec = tvec.reshape(1, 3)
        return torch.tensor(R, dtype=torch.float32, device=device), torch.tensor(
            tvec, dtype=torch.float32, device=device
        )

    R, t = cal_extrinsics_using_pnp(points_3d_init, points_2d, K)

    img_h, img_w = image.shape[-2], image.shape[-1]
    img_shape = torch.tensor([[img_h, img_w]], dtype=torch.float32, device=device)

    pose_6d = matrix_to_rotation_6d(pose_rot)

    # Initialization
    opt_pose = nn.Parameter(pose_6d.clone(), requires_grad=True)
    opt_shape = nn.Parameter(full_shape.clone(), requires_grad=True)
    std = std.to(dtype=torch.float32, device=device).view(1100, 1)

    # Vposer
    vposer_cfg_path = opt_config.vposer_cfg_path
    vposer_param_path = opt_config.vposer_param_path
    vposer_cfg = OmegaConf.load(vposer_cfg_path)
    vposer = VPoser(vposer_cfg)
    state_dict = torch.load(vposer_param_path)["state_dict"]
    state_dict = {k.replace("vp_model.", "") if k.startswith("vp_model.") else k: v for k, v in state_dict.items()}
    vposer_keys = list(vposer.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in vposer_keys}
    vposer.load_state_dict(state_dict, strict=False)
    vposer = vposer.to(device).eval()

    # Optimisation
    optimizer = torch.optim.LBFGS([opt_pose, opt_shape], lr=1.0, max_iter=20)

    def closure():
        optimizer.zero_grad()
        opt_pose_rotmat = rotation_6d_to_matrix(opt_pose.view(-1, 6)).view(51, 3, 3)
        # 3D ldmks given by current pose and shape
        smpl_out = smplh_layer(
            betas=opt_shape.unsqueeze(0),
            global_orient=None,
            body_pose=opt_pose_rotmat[0:21].unsqueeze(0),
            left_hand_pose=opt_pose_rotmat[21:36].unsqueeze(0),
            right_hand_pose=opt_pose_rotmat[36:].unsqueeze(0),
            transl=None,
        )
        pred_3d = smpl_out.vertices[0, ldmk_indices]
        # project to 2D
        pred_3d_cam = (R @ pred_3d.T).T + t
        proj = (K @ pred_3d_cam.T).T
        pred_2d = proj[:, :2] / proj[:, 2:3]
        # 2D reference
        gt_2d = ldmks.to(dtype=torch.float32, device=device)
        # reprojection loss
        loss_reproj = nn.functional.mse_loss(
            pred_2d, gt_2d, weight=torch.concatenate([1 / (std**2), 1 / (std**2)], dim=1) / (img_shape**2)
        )
        # shape loss
        shape_reg = torch.norm(opt_shape, p=2)
        # pose loss regularized by vposer
        opt_pose_aa = rot2aa(opt_pose_rotmat)  # Use only the body part (first 21 joints)
        pose_body_aa = opt_pose_aa[:21]  # shape [21, 3]
        pose_latent = vposer.encode(pose_body_aa.view(1, -1)).mean
        pose_reg = torch.norm(pose_latent, p=2)
        # total loss
        loss = 1.0 * loss_reproj + 0.1 * shape_reg + 0.1 * pose_reg
        # backward
        loss.backward()
        return loss

    # Run optimization
    optimizer.step(closure)
    # make output
    shape_optimized = opt_shape.detach().cpu()
    opt_pose_rotmat = rotation_6d_to_matrix(opt_pose.view(51, 6)).view(51, 3, 3)
    opt_pose_aa = rot2aa(opt_pose_rotmat)
    zero_global = rot2aa(
        torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
    )  # global orientation is set to 0
    pose_optimised = torch.concatenate([zero_global, opt_pose_aa], dim=0).detach().cpu()

    return pose_optimised, shape_optimized, R, t


def visualize_optimization(img, shape, pose, R, tvec, K, id, dir="experiments/ehf/"):
    world_to_cam_pnp = np.eye(4, dtype=np.float32)
    world_to_cam_pnp[:3, :3] = R.astype(np.float32)
    world_to_cam_pnp[:3, 3] = tvec.reshape(3).astype(np.float32)
    image_np = img.permute(1, 2, 0).cpu().numpy()
    image_mesh = (image_np * 255).astype(np.uint8)
    if len(shape) < 16:
        shape = np.pad(shape, (0, 16 - len(shape)), "constant")
    if pose.shape[0] == 51:
        pose = np.concatenate([np.zeros((1, 3)), pose], axis=0)

    optimized_mesh_image = draw_mesh(
        image_mesh.copy(),
        identity=shape,  # shape: (16,)
        pose=pose,  # shape: (52, 3)
        translation=np.zeros(3, dtype=np.float32),  # no translation
        world_to_cam=world_to_cam_pnp,
        cam_to_img=K,
    )

    # # === Visualize result ===
    plt.figure(figsize=(6, 6))
    plt.imshow(optimized_mesh_image)
    plt.axis("off")
    plt.savefig(dir + f"mesh_id_{id}.pdf", bbox_inches="tight")


# Example usage
if __name__ == "__main__":

    data_root = f"data/synth_body"
    meta_file = f"data/annot/body_meta.pkl"

    body_model, config = load_model("body")
    hand_model, _ = load_model("hand")

    val_loader = get_val_dataloader(config, data_root, meta_file, 3, "test")

    images, targets, uids = next(iter(val_loader))
    images = images.to(config.device)
    roi = targets["roi"].to(config.device)
    cam_int = targets["cam_int"].to(config.device)

    print(images.shape, roi.shape, cam_int.shape)

    full_model_inference(images, roi, cam_int, body_model, hand_model, config.device)
