# This file contains useful toolkits from aitviewer: https://github.com/eth-ait/aitviewer
# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import roma
import torch
import torch.nn.functional as F

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def rot2aa(rotation_matrices):
    """
    Convert rotation matrices to rotation vectors (angle-axis representation).
    :param rotation_matrices: A torch tensor of shape (..., 3, 3).
    :return: A torch tensor of shape (..., 3).
    """
    assert isinstance(rotation_matrices, torch.Tensor)
    return roma.rotmat_to_rotvec(rotation_matrices)

def aa2rot(rotation_vectors):
    """
    Convert rotation vectors (angle-axis representation) to rotation matrices.
    :param rotation_vectors: A torch tensor of shape (..., 3).
    :return: A torch tensor of shape (..., 3, 3).
    """
    assert isinstance(rotation_vectors, torch.Tensor)
    return roma.rotvec_to_rotmat(rotation_vectors)

JOINT_PARENTS = {
    "body": [
        [0, -1],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 1],
        [5, 2],
        [6, 3],
        [7, 4],
        [8, 5],
        [9, 6],
        [10, 7],
        [11, 8],
        [12, 9],
        [13, 9],
        [14, 9],
        [15, 12],
        [16, 13],
        [17, 14],
        [18, 16],
        [19, 17],
        [20, 18],
        [21, 19],
        [22, 20],
        [23, 22],
        [24, 23],
        [25, 20],
        [26, 25],
        [27, 26],
        [28, 20],
        [29, 28],
        [30, 29],
        [31, 20],
        [32, 31],
        [33, 32],
        [34, 20],
        [35, 34],
        [36, 35],
        [37, 21],
        [38, 37],
        [39, 38],
        [40, 21],
        [41, 40],
        [42, 41],
        [43, 21],
        [44, 43],
        [45, 44],
        [46, 21],
        [47, 46],
        [48, 47],
        [49, 21],
        [50, 49],
        [51, 50],
    ], "hand": [
        [0, -1],
        [1, 0],
        [2, 1],
        [3, 2],
        [4, 0],
        [5, 4],
        [6, 5],
        [7, 0],
        [8, 7],
        [9, 8],
        [10, 0],
        [11, 10],
        [12, 11],
        [13, 0],
        [14, 13],
        [15, 14],
        [16, 3],
        [17, 6],
        [18, 9],
        [19, 12],
        [20, 15],
    ],
}

def local_to_global(poses, part:str, output_format="aa", input_format="aa"):
    """
    Convert relative joint angles to global ones by unrolling the kinematic chain.
    :param poses: A tensor of shape (batch_size, N_JOINTS*3) defining the relative poses in angle-axis format.
    :param parts: indicating 'body', 'hand', and 'face'
    :param output_format: 'aa' or 'rotmat'.
    :param input_format: 'aa' or 'rotmat'
    :return: The global joint angles as a tensor of shape (batch_size, N_JOINTS*DOF).
    """
    assert part in ['body', 'face', 'hand']
    parents = JOINT_PARENTS[part] #A list of parents for each joint j, i.e. parent[j][1] is the parent of joint j.

    assert output_format in ["aa", "rotmat"]
    assert input_format in ["aa", "rotmat"]
    dof = 3 if input_format == "aa" else 9
    n_joints = poses.shape[-1] // dof
    if input_format == "aa":
        local_oris = aa2rot(poses.reshape((-1, 3)))
    else:
        local_oris = poses
    local_oris = local_oris.reshape((-1, n_joints, 3, 3))
    global_oris = torch.zeros_like(local_oris)

    # Initialize global_oris as a list to avoid in-place modifications
    global_oris = [None] * n_joints

    for j in range(n_joints):
        if parents[j][1] < 0:
            # Root rotation
            global_oris[j] = local_oris[..., j, :, :]
        else:
            parent_rot = global_oris[parents[j][1]]
            local_rot = local_oris[..., j, :, :]
            global_oris[j] = torch.matmul(parent_rot, local_rot)

    # Stack the list into a single tensor
    global_oris = torch.stack(global_oris, dim=-3)

    if output_format == "aa":
        global_oris = rot2aa(global_oris.reshape((-1, 3, 3)))
        res = global_oris.reshape((-1, n_joints * 3))
    else:
        res = global_oris.reshape((-1, n_joints * 3 * 3))
    return res
