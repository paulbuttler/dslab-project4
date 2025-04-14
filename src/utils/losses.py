# loss.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import torch.nn as nn
from utils.rottrans import rotation_6d_to_matrix, local_to_global

class RotationLoss(nn.Module):
    """rotation loss in rad computed from rotation matrix"""
    def __init__(self, reduction='mean'):
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
        
        if self.reduction == 'mean':
            return angle_diffs.mean()
        elif self.reduction == 'sum':
            return angle_diffs.sum()
        return angle_diffs

class JointPositionLoss(nn.Module):
    """l1 loss of joint loc diff"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred_joints, gt_joints):
        """
        Args:
            pred_joints: (B, J, 3) predicted loc of joints
            gt_joints: (B, J, 3) ground truth joint loc
        """
        l1_loss = torch.abs(pred_joints - gt_joints).sum(dim=-1)
        
        if self.reduction == 'mean':
            return l1_loss.mean()
        elif self.reduction == 'sum':
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
        pred_var = torch.exp(pred[..., 2])
        return (0.5 * (pred_mu - gt).pow(2) / pred_var[..., torch.newaxis]).mean() + 0.5 * pred[..., 2].mean()

class DNNMultiTaskLoss(nn.Module):
    """DNN synthetic loss"""
    def __init__(self, config, smplh_layer):
        super().__init__()
        self.rot_loss = RotationLoss()
        self.joint_loss = JointPositionLoss()
        self.landmark_loss = ProbabilisticLandmarkLoss()
        self.pose_loss = nn.L1Loss()
        self.shape_loss = nn.L1Loss()
        
        self.weights = {
            'rotation': config.rot_weight,
            'translation': config.trans_weight,
            'landmark': config.landmark_weight,
            'pose': config.pose_weight,
            'shape': config.shape_weight
        }
        
        self.smplh_layer = smplh_layer

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict containing at least the following key-value pairs
                - pose: (B, 52, 6) SMPLH pose params in 6D rotations
                - shape: (B, 10) SMPLH shape params
                - landmarks: (B, L, 3) landmarks prediction
                
            targets: dict containin at least the following key-value pairs
                - pose: (B, 52, 6) SMPLH pose params in 6D rotations
                - shape: (B, 10) SMPLH shape params
                - landmarks: (B, L, 2) landmarks gt
        Return:
            loss: dict containing different terms of loss
        """
        # axis angle to rot mat
        pred_pose_rotmat = rotation_6d_to_matrix(outputs['pose']) # (B, 52, 3, 3)
        gt_pose_rotmat = rotation_6d_to_matrix(targets['pose']) # (B, 52, 3, 3)
        
        # rot loss
        rot_loss = self.rot_loss(pred_pose_rotmat, gt_pose_rotmat)
        
        # translation loss
        pred_joints = self._get_predicted_joints(shape=outputs['shape'], pose=pred_pose_rotmat, require_grad=True) # (B, 52, 3)
        gt_joints = self._get_predicted_joints(shape=targets['shape'], pose=gt_pose_rotmat, require_grad=False) # (B, 52, 3)
        joint_loss = self.joint_loss(pred_joints, gt_joints)
        
        # 2D landmark loss
        landmark_loss = self.landmark_loss(outputs['landmarks'], targets['landmarks'])
        
        # pose loss
        pose_loss = self.pose_loss(outputs['pose'], targets['pose'])
        
        # shape loss
        shape_loss = self.shape_loss(outputs['shape'], targets['shape'])

        # total_loss
        total_loss = (
            self.weights['rotation'] * rot_loss +
            self.weights['translation'] * joint_loss +
            self.weights['landmark'] * landmark_loss +
            self.weights['pose'] * pose_loss +
            self.weights['shape'] * shape_loss
        )
        
        return {
            'total': total_loss,
            'rotation': rot_loss,
            'translation': joint_loss,
            'landmark': landmark_loss,
            'pose': pose_loss,
            'shape': shape_loss
        }
    
    def _get_predicted_joints(self, shape, pose, require_grad=True):
        '''
            get joint locs using smplh layer
            Params:
                shape: (batch_size, 10)
                pose: (batch_size, 52, 3, 3) in rot mat
                require_grad: Bool. Set True for prediction, set False for ground truth.
            Return:
                joints: (batch_size, 52, 3)
        '''
        if require_grad:
            smpl_output = self.smplh_layer(
                betas=shape[:, :10],
                global_orient=pose[:, 0, ...],
                body_pose=pose[:, 1:22, ...],
                left_hand_pose=pose[:, 22:37, ...],
                right_hand_pose=pose[:, 37:, ...],
                transl=None,
                return_verts=False,
                return_full_pose=False
            )
        else:
            with torch.no_grad():
                smpl_output = self.smplh_layer(
                betas=shape,
                global_orient=pose[:, 0, ...],
                body_pose=pose[:, 1:22, ...],
                left_hand_pose=pose[:, 22:37, ...],
                right_hand_pose=pose[:, 37:, ...],
                transl=None,
                return_verts=False,
                return_full_pose=False
            )
        return smpl_output.joints