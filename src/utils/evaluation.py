import numpy as np
import torch
from datasets.transforms import denormalize
from models.smplx.body_models import SMPL
from aitviewer.renderables.smpl import SMPLSequence
from scipy.spatial.transform import Rotation as R
from utils.rottrans import rot2aa, rotation_6d_to_matrix 
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence

def compute_pve_neutral_pose_scale_corrected(smpl_layer, predicted_smpl_shape, target_smpl_shape):
    """
    Given predicted and target SMPL shape parameters, computes neutral-pose per-vertex error
    after scale-correction (to account for scale vs camera depth ambiguity).
    :param predicted_smpl_parameters: predicted SMPL shape parameters tensor with shape (1, 10)
    :param target_smpl_parameters: target SMPL shape parameters tensor with shape (1, 10)
    """
    poses = np.zeros([1, smpl_layer.bm.NUM_BODY_JOINTS * 3])
    pred_smpl_seq = SMPLSequence(poses, smpl_layer, betas=predicted_smpl_shape)
    pred_smpl_neutral_pose_vertices = pred_smpl_seq.fk()[0]

    target_smpl_seq = SMPLSequence(poses, smpl_layer, betas=target_smpl_shape)
    target_smpl_neutral_pose_vertices = target_smpl_seq.fk()[0]

    pred_smpl_neutral_pose_vertices_rescale = scale_and_translation_transform_batch(pred_smpl_neutral_pose_vertices,
                                                                                    target_smpl_neutral_pose_vertices)

    # Compute PVE-T-SC
    pve_neutral_pose_scale_corrected = np.linalg.norm(pred_smpl_neutral_pose_vertices_rescale
                                                      - target_smpl_neutral_pose_vertices,
                                                      axis=-1)  # (1, 6890)

    out = np.mean(pve_neutral_pose_scale_corrected)

    return out
    
def get_full_shape_and_pose(pred_shape, pred_pose, gt_shape, gt_pose):
    """Use ground-truth values for any missing pose and shape parameters"""

    if(gt_shape.shape[1] == 10):
        full_shape = pred_shape
    else:
        full_shape = torch.cat(
            [
                pred_shape,
                gt_shape[:, -6:],
            ],
            dim=1,
        )

    # print(gt_pose.shape, pred_pose.shape, gt_pose.shape)
    # print(gt_pose[:, 0].unsqueeze(1).shape, pred_pose.shape, gt_pose[:, 22:].shape)
    full_pose = torch.cat(
        [gt_pose[:, 0].unsqueeze(1), pred_pose, gt_pose[:, 22:]], dim=1
    )

    return full_shape, full_pose

def get_3D_positions(smpl_layer, pred_pose):
    poses = pred_pose[:, 0:21].contiguous().view(1, smpl_layer.bm.NUM_BODY_JOINTS * 3).cpu().numpy()
    smpl_seq = SMPLSequence(poses, smpl_layer)
    fk = smpl_seq.fk()[1]
    return torch.tensor(fk).view(-1, 3)

def get_obj_file_vertices(file_path):
    """
    Get vertices from an obj file.
    :param file_path: path to the obj file
    :return: vertices as a numpy array of shape (N, 3)
    """
    print(file_path)
    with open(file_path[0], 'r') as f:
        lines = f.readlines()

    vertices = []
    for line in lines:
        if line.startswith('v '):
            vertex = list(map(float, line.strip().split()[1:]))
            vertices.append(vertex)

    return np.array(vertices)

# def body_pose_to_axis_angle(body_pose):
#     # body_pose: (1, 9*n) or (9*n,)
#     pose_flat = body_pose.reshape(-1)
#     if hasattr(pose_flat, "detach"):  # If it's a tensor
#         pose_flat = pose_flat.detach().cpu().numpy()
#     n = pose_flat.shape[0] // 9
#     axis_angles = []
#     for i in range(n):
#         rotmat = pose_flat[i*9:(i+1)*9].reshape(3, 3)
#         r = R.from_matrix(rotmat)
#         axis_angles.append(r.as_rotvec())
#     axis_angles = np.stack(axis_angles, axis=1)  # shape (3, n)
#     return axis_angles.reshape(-1,3)

def get_mesh_vertices(smpl_layer, pred_shape, pred_pose):
    poses = pred_pose[:, 0:21].contiguous().view(1, smpl_layer.bm.NUM_BODY_JOINTS * 3).cpu().numpy()
    smpl_seq = SMPLSequence(poses, smpl_layer, betas=pred_shape)
    fk = smpl_seq.fk()[0]

    return torch.tensor(fk).view(-1, 3)

def get_similarity(smpl_layer, pred_shape, gt_shape, pred_pose, gt_pose, measure_type='body', num_joints=22):
    """
    Get similarity transform between predicted and ground truth SMPL meshes.
    :param pred_shape: predicted SMPL shape parameters
    :param gt_shape: ground truth SMPL shape parameters
    :param pred_pose: predicted SMPL pose parameters
    :param gt_pose: ground truth SMPL pose parameters
    :return: loss values (MPVPE, PA-MPVPE, PA-MPJPE, PA params)
    """
    pred_full_pose = torch.cat([gt_pose[:, 0].unsqueeze(1), pred_pose], dim=1)
    
    # 0: root 
    # 1:22 (21): body, 
    # 22:37 (15): left hand, 
    # 37:52 (15): right hand
    required_pose_len = 52  # 1 root + 21 body + 15 left hand + 15 right hand
    if pred_full_pose.shape[1] < required_pose_len:
        pad_len = required_pose_len - pred_full_pose.shape[1]
        pred_full_pose = torch.cat([pred_full_pose, torch.zeros(pred_full_pose.shape[0], pad_len, device=pred_full_pose.device)], dim=1)
    if gt_pose.shape[1] < required_pose_len:
        pad_len = required_pose_len - gt_pose.shape[1]
        gt_pose = torch.cat([gt_pose, torch.zeros(gt_pose.shape[0], pad_len, device=gt_pose.device)], dim=1)


    #if(gt_shape.shape[1] == 10):
     #   pred_full_shape = pred_shape
    #else:
        #pred_full_shape = torch.cat(
         #   [
        #        pred_shape,
       #         gt_shape[:, -6:],
      #      ],
     #       dim=1,
    #    )

    
    pred_smpl_seq = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_root=pred_full_pose[:, 0].reshape(1, -1),
        poses_body=pred_full_pose[:, 1:22].reshape(1, -1),
        poses_left_hand=pred_full_pose[:, 22:37].reshape(1, -1),
        poses_right_hand=pred_full_pose[:, 37:].reshape(1, -1),
        betas=pred_shape,
    )
    pred_fk = pred_smpl_seq.fk()

    gt_smpl_seq = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_root=gt_pose[:, 0].reshape(1, -1),
        poses_body=gt_pose[:, 1:22].reshape(1, -1),
        poses_left_hand=gt_pose[:, 22:37].reshape(1, -1),
        poses_right_hand=gt_pose[:, 37:].reshape(1, -1),
        betas=gt_shape,
    )
    gt_fk = gt_smpl_seq.fk()
    print(pred_fk[1].shape, gt_fk[1].shape, pred_fk[0].shape, gt_fk[0].shape)
    pred_hat, vertex_hat, params = compute_similarity_transform(
        pred_fk[1].squeeze(),
        gt_fk[1].squeeze(),
        num_joints=num_joints,
        verts=gt_fk[0].squeeze(),
    )

    MPVPE = np.mean(np.linalg.norm(pred_fk[0] - gt_fk[0], axis=-1)) * 1000 # in mm
    PA_MPVPE = np.mean(np.linalg.norm(vertex_hat - gt_fk[0], axis=-1)) * 1000 # in mm
    PA_MPJPE = np.mean(np.linalg.norm(pred_hat - gt_fk[1], axis=-1)) * 1000 # in mm
    return MPVPE, PA_MPVPE, PA_MPJPE, params





def compute_similarity_transform(S1, S2, num_joints, verts=None):
    """
    Computes a similarity transform (sR, t) that takes a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale. I.e., solves the orthogonal Procrutes problem.
    Borrowed from https://github.com/aymenmir1/3dpw-eval/blob/master/evaluate.py
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        if verts is not None:
            verts = verts.T
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # Use only body joints for procrustes
    S1_p = S1[:, :num_joints]
    S2_p = S2[:, :num_joints]
    # 1. Remove mean.
    mu1 = S1_p.mean(axis=1, keepdims=True)
    mu2 = S2_p.mean(axis=1, keepdims=True)
    X1 = S1_p - mu1
    X2 = S2_p - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    verts_hat = None
    if verts is not None:
        verts_hat = scale * R.dot(verts) + t
        if transposed:
            verts_hat = verts_hat.T

    if transposed:
        S1_hat = S1_hat.T

    procrustes_params = {"scale": scale, "R": R, "trans": t}

    if verts_hat is not None:
        return S1_hat, verts_hat, procrustes_params
    else:
        return S1_hat, procrustes_params
    
def scale_and_translation_transform_batch(P, T):
    """
    First normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of N 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of N reference 3D meshes.
    :return: P transformed
    """
    P_mean = np.mean(P, axis=1, keepdims=True)
    P_trans = P - P_mean
    P_scale = np.sqrt(np.sum(P_trans ** 2, axis=(1, 2), keepdims=True) / P.shape[1])
    P_normalised = P_trans / P_scale

    T_mean = np.mean(T, axis=1, keepdims=True)
    T_scale = np.sqrt(np.sum((T - T_mean) ** 2, axis=(1, 2), keepdims=True) / T.shape[1])

    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed

def evaluate_model(model, batch, device):
    """
    Get ground truth data and model predictions from a dataset batch.
    """
    images, targets, uids = batch

    h = images.shape[-1]
    images = images.to(device)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        pred_ldmks = (outputs["landmarks"][..., :2] * h).cpu()
        pred_std = (torch.exp(0.5 * outputs["landmarks"][..., 2]) * h).cpu()
        pred_pose = outputs["pose"].cpu()
        pred_shape = outputs["shape"].cpu()

        pred_pose = rotation_6d_to_matrix(pred_pose)  # [B, 21, 3, 3]
        pred_pose = rot2aa(pred_pose)  # [B, 21, 3]

    images = denormalize(images).cpu()  # [B, 3, H, W]

    return (
        images,
        uids,
        pred_ldmks,
        pred_std,
        pred_shape,
        pred_pose,
    )