import numpy as np
import torch
import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.EHF_dataset import EHF_Dataset
from aitviewer.models.smpl import SMPLLayer  # type: ignore
from models.load import load_model, get_val_dataloader
from inference.initialization import initial_pose_estimation
from inference.optimization import full_model_inference, visualize_optimization
from utils.visualization import visualize_pose_and_shape


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


def get_vertices_and_joints(pose, shape):
    """
    Get vertices and joints from SMPL layer given pose and shape. Global orientation and translation are set to zero.
    """
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=shape.shape[1])
    B = pose.shape[0]
    vertices, joints = smpl_layer.fk(
        betas=shape.reshape(B, -1),
        poses_body=pose[:, 1:22].reshape(B, -1),
        poses_left_hand=pose[:, 22:37].reshape(B, -1),
        poses_right_hand=pose[:, 37:].reshape(B, -1),
    )
    return vertices, joints[:, :52]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synth", "ehf"], default="ehf")
    parser.add_argument("--optimize", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()

    device = "cuda"

    body_model, config = load_model("body")
    hand_model, _ = load_model("hand")

    if args.dataset == "ehf":
        dataset = EHF_Dataset(data_dir=Path("data/EHF"))
        test_loader = DataLoader(dataset, batch_size=20, shuffle=False)

    elif args.dataset == "synth":
        data_root = "data/synth_body"
        meta_file = "data/annot/body_meta.pkl"
        test_loader = get_val_dataloader(config, data_root, meta_file, 100, "test")

    with open("src/evaluation/smpl_vert_segmentation.json", "r") as f:
        smpl_vert_segmentation = json.load(f)

    left_hand_seg = smpl_vert_segmentation["leftHand"]
    right_hand_seg = smpl_vert_segmentation["rightHand"]
    head_seg = smpl_vert_segmentation["head"]
    body_seg = np.setdiff1d(np.arange(6890), np.concatenate([left_hand_seg, right_hand_seg, head_seg]))

    print("\nBody segmentation", len(body_seg))
    print("Left hand segmentation", len(left_hand_seg))
    print("Right hand segmentation", len(right_hand_seg))
    print("Head segmentation", len(head_seg))
    print("Total vertices", len(body_seg) + len(left_hand_seg) + len(right_hand_seg) + len(head_seg))

    MPVPE = {"Full-body": 0.0, "Hands": 0.0, "Face": 0.0}
    PA_MPVPE = {"Full-body": 0.0, "Body": 0.0, "Hands": 0.0, "Face": 0.0}
    PA_MPJPE = {"Body": 0.0, "Hands": 0.0}

    for images, targets, uids in test_loader:
        images = images.to(device)
        rois = targets["roi"].to(device)
        cam_int = targets["cam_int"].to(device)

        gt_pose = targets["pose"]
        gt_shape = targets["shape"]

        if args.optimize:
            pred_pose, pred_shape, R, t = full_model_inference(images, rois, cam_int, body_model, hand_model, device)
        else:
            ldmks, std, pred_pose, pred_shape = initial_pose_estimation(images, rois, body_model, hand_model, device)
            pred_pose = torch.cat([torch.zeros((pred_pose.shape[0], 1, 3), device=pred_pose.device), pred_pose], dim=1)

        if args.visualize:
            # Visualize 3d pose and shape in aitviewer
            gt_pose[:, 0] = torch.zeros_like(gt_pose[:, 0], device=gt_pose.device)  # Set global orientation to zero

            visualize_pose_and_shape(
                uids,
                gt_pose,
                gt_shape,
                pred_pose,
                pred_shape,
                torch.zeros((pred_pose.shape[0], 3), device=pred_pose.device),  # Set translation to zero
                "full",
                args.dataset,
                billboard=False,
            )

        # Get vertices and joints
        gt_verts, gt_joints = get_vertices_and_joints(gt_pose, gt_shape)
        pred_verts, pred_joints = get_vertices_and_joints(pred_pose, pred_shape)

        gt_verts, gt_joints = gt_verts.cpu().numpy(), gt_joints.cpu().numpy()
        pred_verts, pred_joints = pred_verts.cpu().numpy(), pred_joints.cpu().numpy()

        # Align pelvis of ground truth and predicted vertices
        gt_verts_pelv_aligned = gt_verts - gt_joints[:, 0][:, np.newaxis, :]
        pred_verts_pelv_aligned = pred_verts - pred_joints[:, 0][:, np.newaxis, :]

        # Per-part root-joint alignment for MPVPE of hands and face
        gt_left_root_aligned = gt_verts[:, left_hand_seg] - gt_joints[:, 20][:, np.newaxis, :]
        gt_right_root_aligned = gt_verts[:, right_hand_seg] - gt_joints[:, 21][:, np.newaxis, :]
        gt_face_root_aligned = gt_verts[:, head_seg] - gt_joints[:, 15][:, np.newaxis, :]
        pred_left_root_aligned = pred_verts[:, left_hand_seg] - pred_joints[:, 20][:, np.newaxis, :]
        pred_right_root_aligned = pred_verts[:, right_hand_seg] - pred_joints[:, 21][:, np.newaxis, :]
        pred_face_root_aligned = pred_verts[:, head_seg] - pred_joints[:, 15][:, np.newaxis, :]

        for idx in range(len(uids)):

            # Peform procrustes alignment for full body and body segments separately
            verts_aligned, _ = compute_similarity_transform(pred_verts[idx], gt_verts[idx], num_joints=6890)
            body_verts_aligned, _ = compute_similarity_transform(
                pred_verts[idx][body_seg], gt_verts[idx][body_seg], num_joints=len(body_seg)
            )
            left_verts_aligned, _ = compute_similarity_transform(
                pred_verts[idx][left_hand_seg],
                gt_verts[idx][left_hand_seg],
                num_joints=len(left_hand_seg),
            )
            right_verts_aligned, _ = compute_similarity_transform(
                pred_verts[idx][right_hand_seg],
                gt_verts[idx][right_hand_seg],
                num_joints=len(right_hand_seg),
            )
            head_verts_aligned, _ = compute_similarity_transform(
                pred_verts[idx][head_seg], gt_verts[idx][head_seg], num_joints=len(head_seg)
            )

            body_joints_aligned, _ = compute_similarity_transform(
                pred_joints[idx][:22], gt_joints[idx][:22], num_joints=22
            )
            left_joints_aligned, _ = compute_similarity_transform(
                pred_joints[idx][22:37], gt_joints[idx][22:37], num_joints=15
            )
            right_joints_aligned, _ = compute_similarity_transform(
                pred_joints[idx][37:], gt_joints[idx][37:], num_joints=15
            )

            # Compute MPVPE
            MPVPE["Full-body"] += np.mean(
                np.linalg.norm(pred_verts_pelv_aligned[idx] - gt_verts_pelv_aligned[idx], axis=1)
            )
            MPVPE_hands = np.concatenate(
                [
                    np.linalg.norm(
                        pred_left_root_aligned[idx] - gt_left_root_aligned[idx],
                        axis=1,
                    ),
                    np.linalg.norm(
                        pred_right_root_aligned[idx] - gt_right_root_aligned[idx],
                        axis=1,
                    ),
                ],
                axis=0,
            )
            MPVPE["Hands"] += np.mean(MPVPE_hands)
            MPVPE["Face"] += np.mean(
                np.linalg.norm(
                    pred_face_root_aligned[idx] - gt_face_root_aligned[idx],
                    axis=1,
                )
            )

            # Compute PA_MPVPE
            PA_MPVPE["Full-body"] += np.mean(np.linalg.norm(verts_aligned - gt_verts[idx], axis=1))
            PA_MPVPE["Body"] += np.mean(np.linalg.norm(body_verts_aligned - gt_verts[idx][body_seg], axis=1))
            PA_MPVPE_hands = np.concatenate(
                [
                    np.linalg.norm(left_verts_aligned - gt_verts[idx][left_hand_seg], axis=1),
                    np.linalg.norm(right_verts_aligned - gt_verts[idx][right_hand_seg], axis=1),
                ],
                axis=0,
            )
            PA_MPVPE["Hands"] += np.mean(PA_MPVPE_hands)
            PA_MPVPE["Face"] += np.mean(np.linalg.norm(head_verts_aligned - gt_verts[idx][head_seg], axis=1))

            # Compute PA_MPJPE
            PA_MPJPE["Body"] += np.mean(np.linalg.norm(body_joints_aligned - gt_joints[idx][:22], axis=1))
            PA_MPJPE_Hands = np.concatenate(
                [
                    np.linalg.norm(left_joints_aligned - gt_joints[idx][22:37], axis=1),
                    np.linalg.norm(right_joints_aligned - gt_joints[idx][37:], axis=1),
                ],
                axis=0,
            )
            PA_MPJPE["Hands"] += np.mean(PA_MPJPE_Hands)

    N = len(test_loader.dataset)
    MPVPE = {key: val / N * 1000 for key, val in MPVPE.items()}
    PA_MPVPE = {key: val / N * 1000 for key, val in PA_MPVPE.items()}
    PA_MPJPE = {key: val / N * 1000 for key, val in PA_MPJPE.items()}

    print("\nEvaluation results with" + "out" * (not args.optimize) + " optimization:\n")
    print("MPVPE", MPVPE)
    print("PA_MPVPE", PA_MPVPE)
    print("PA_MPJPE", PA_MPJPE)
