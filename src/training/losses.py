import torch
import torch.nn as nn
from utils.rottrans import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    local_to_global,
    aa2rot,
)


class RotationLoss(nn.Module):
    """Rotation loss in rad computed from world space rotation matrices"""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, R_hat, R_gt):
        """
        Args:
            R_hat: (B, J, 3, 3) rot mat
            R_gt: (B, J, 3, 3) rot mat
        """
        # relative rot mat
        R_rel = torch.matmul(R_hat, R_gt.transpose(-1, -2))

        # angle diff in rad
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        clipped_trace = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
        angle_diffs = torch.acos(clipped_trace)

        if self.reduction == "mean":
            return angle_diffs.mean()
        elif self.reduction == "sum":
            return angle_diffs.sum()
        return angle_diffs


class JointPositionLoss(nn.Module):
    """L1 loss of joint loc diff"""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_joints, gt_joints):
        """
        Args:
            pred_joints: (B, J, 3) predicted loc of joints
            gt_joints: (B, J, 3) ground truth joint loc
        """
        l1_loss = torch.abs(pred_joints - gt_joints).sum(dim=-1)

        if self.reduction == "mean":
            return l1_loss.mean()
        elif self.reduction == "sum":
            return l1_loss.sum()
        return l1_loss


class ProbabilisticLandmarkLoss(nn.Module):
    """2D landmark loss with estimated variance"""

    def forward(self, pred, gt):
        """
        Args:
            pred: (B, L, 3) predicted 2D landmarks [mu_x, mu_y, logvar]
            gt: (B, L, 2) ground truth 2D landmark loc [x, y]
        """
        pred_mu = pred[..., :2]
        log_var = pred[..., 2]

        sq_diff = (pred_mu - gt).pow(2).sum(dim=-1)
        loss = log_var + 0.5 * sq_diff * torch.exp(-log_var)
        return loss.mean()


class DNNMultiTaskLoss(nn.Module):
    """DNN synthetic loss"""

    def __init__(self, config, smplh_layer):
        super().__init__()
        self.rot_loss = RotationLoss()
        self.joint_loss = JointPositionLoss()
        self.landmark_loss = ProbabilisticLandmarkLoss()
        self.pose_loss = nn.L1Loss()
        self.shape_loss = nn.L1Loss()
        self.landmark_mse_loss = nn.MSELoss()

        self.weights = {
            "rotation": config.rot_weight,
            "translation": config.trans_weight,
            "landmark": config.landmark_weight,
            "pose": config.pose_weight,
            "shape": config.shape_weight,
        }

        self.smplh_layer = smplh_layer

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict containing at least the following key-value pairs
                - pose: (B, J, 6) SMPLH pose params in 6D rotations
                - landmarks: (B, L, 3) landmarks prediction
                - (OPTIONAL) shape: (B, 10) SMPLH shape params

            targets: dict containing the following key-value pairs
                - pose: (B, 52, 3) SMPLH pose params in local axis-angle representation
                - shape: (B, 16) SMPLH shape params
                - translation: (B, 3) SMPLH translation
                - landmarks: (B, L, 2) landmarks gt
        Return:
            loss: dict containing different terms of loss
        """
        body = "shape" in outputs  # else using hand model

        # 2D landmark loss
        landmark_loss = self.landmark_loss(outputs["landmarks"], targets["landmarks"])

        # shape loss
        if body:
            shape_loss = self.shape_loss(outputs["shape"], targets["shape"][:, :10])

        gt_pose_aa = targets["pose"]  # (B, 52, 3)
        pred_pose_6d = outputs["pose"]  # (B, J, 6)

        with torch.no_grad():
            gt_pose_rotmat = aa2rot(gt_pose_aa)  # (B, 52, 3, 3)
            gt_global_rot = local_to_global(
                gt_pose_rotmat.reshape(-1, 52 * 9),
                part="body",
                output_format="rotmat",
                input_format="rotmat",
            ).reshape(-1, 52, 3, 3)

            if body:
                gt_pose_6d = matrix_to_rotation_6d(gt_pose_rotmat[:, 1:22, ...])  # (B, 21, 6)
            else:
                gt_pose_6d = matrix_to_rotation_6d(gt_pose_rotmat[:, 22:37, ...])  # (B, 15, 6)

        # pose loss 6d rotations
        pose_loss = self.pose_loss(pred_pose_6d, gt_pose_6d)

        pred_pose_rotmat = rotation_6d_to_matrix(pred_pose_6d)  # (B, J, 3, 3)
        full_shape, full_pose = self.get_full_shape_and_pose(
            outputs["shape"] if body else None,
            pred_pose_rotmat,
            targets["shape"],
            gt_pose_rotmat,
        )  # (B, 16), (B, 52, 3, 3)

        pred_global_rot = local_to_global(
            full_pose.reshape(-1, 52 * 9),
            part="body",
            output_format="rotmat",
            input_format="rotmat",
        ).reshape(-1, 52, 3, 3)

        translation = targets["translation"]  # (B, 3)
        # Joint translation loss
        gt_joints = self._get_predicted_joints(
            shape=targets["shape"],
            pose=gt_pose_rotmat,
            translation=translation,
            require_grad=False,
        )
        pred_joints = self._get_predicted_joints(
            shape=full_shape,
            pose=full_pose,
            translation=translation,
            require_grad=True,
        )
        if not body:
            gt_global_rot = gt_global_rot[:, 22:37]
            pred_global_rot = pred_global_rot[:, 22:37]
            gt_joints = gt_joints[:, 22:37]
            pred_joints = pred_joints[:, 22:37]

        # Joint rotation and translation loss
        rot_loss = self.rot_loss(pred_global_rot, gt_global_rot)
        joint_loss = self.joint_loss(pred_joints, gt_joints)

        mse_loss = self.landmark_mse_loss(outputs["landmarks"][..., :2], targets["landmarks"])
        mean_var = torch.exp(outputs["landmarks"][..., 2]).mean()

        # total_loss
        total_loss = (
            self.weights["rotation"] * rot_loss
            + self.weights["translation"] * joint_loss
            + self.weights["landmark"] * landmark_loss
            + self.weights["pose"] * pose_loss
            + (self.weights["shape"] * shape_loss if body else 0.0)
        )

        loss_dict =  {"total": total_loss,
            "rotation": rot_loss,
            "translation": joint_loss,
            "landmark": landmark_loss,
            "pose": pose_loss,
            "landmark_mse": mse_loss,
            "mean_var": mean_var
            }
        
        if body:
            loss_dict["shape"] = shape_loss

        return loss_dict

    def _get_predicted_joints(self, shape, pose, translation, require_grad=True):
        """
        get joint locs using smplh layer
        Params:
            shape: (batch_size, 16)
            pose: (batch_size, 52, 3, 3) in rot mat
            translation: (batch_size, 3)
            require_grad: Bool. Set True for prediction, set False for ground truth.
        Return:
            joints: (batch_size, 52, 3)
        """
        if require_grad:
            smpl_output = self.smplh_layer(
                betas=shape,
                global_orient=pose[:, 0],
                body_pose=pose[:, 1:22],
                left_hand_pose=pose[:, 22:37],
                right_hand_pose=pose[:, 37:],
                transl=translation,
                return_verts=False,
                return_full_pose=False,
            )
        else:
            with torch.no_grad():
                smpl_output = self.smplh_layer(
                    betas=shape,
                    global_orient=pose[:, 0],
                    body_pose=pose[:, 1:22],
                    left_hand_pose=pose[:, 22:37],
                    right_hand_pose=pose[:, 37:],
                    transl=translation,
                    return_verts=False,
                    return_full_pose=False,
                )
        # Optionally include extra key points on face, feet and hands by using all 73 components
        return smpl_output.joints[:, :52]

    def get_full_shape_and_pose(self, pred_shape, pred_pose, gt_shape, gt_pose):
        """Use ground-truth values for any missing pose and shape parameters"""

        if pred_shape is None:  # Hand model
            full_shape = gt_shape
            full_pose = torch.cat([gt_pose[:, :22], pred_pose, gt_pose[:, 37:]], dim=1)
        else:
            full_shape = torch.cat([pred_shape, gt_shape[:, -6:]], dim=1)
            full_pose = torch.cat([gt_pose[:, 0].unsqueeze(1), pred_pose, gt_pose[:, 22:]], dim=1)

        return full_shape, full_pose


# Test loss function implementation
if __name__ == "__main__":

    from models.smplx import SMPLHLayer
    from types import SimpleNamespace

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = SimpleNamespace(
        rot_weight=1.0,  # α_r
        trans_weight=2.0,  # α_t   (joint‑translation)
        landmark_weight=10.0,  # α_l
        pose_weight=1.0,  # α_p
        shape_weight=1.0,  # α_s
    )

    smplh_layer = SMPLHLayer(
        model_path="src/models/smplx/params/smplh",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=16,
        dtype=torch.float32,
    ).to(device)

    criterion = DNNMultiTaskLoss(config, smplh_layer).to(device)

    B, L = 2, 1100
    dummy_out = dict(
        pose=torch.randn(B, 21, 6, requires_grad=True, device=device),
        shape=torch.randn(B, 10, requires_grad=True, device=device),
        landmarks=torch.randn(B, L, 3, requires_grad=True, device=device),
    )
    dummy_tgt = dict(
        pose=torch.randn(B, 52, 3, device=device),
        shape=torch.randn(B, 16, device=device),
        translation=torch.randn(B, 3, device=device),
        landmarks=torch.randn(B, L, 2, device=device),
    )
    loss = criterion(dummy_out, dummy_tgt)["total"]
    loss.backward()

    assert dummy_out["pose"].grad is not None
    assert dummy_out["shape"].grad is not None
    assert dummy_out["landmarks"].grad is not None
    print("forward / backward OK")
