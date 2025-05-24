"""Numpy implementation of the SMPL body model.

See https://smpl.is.tue.mpg.de/ for information about the model.

This python file is licensed under the MIT license (see below).
The datasets are licensed under the Research Use of Data Agreement v1.0 (see LICENSE.md).

Copyright (c) 2024 Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pathlib import Path

import numpy as np


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Turns an axis-angle rotation into a 3x3 rotation matrix.

    See https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis%E2%80%93angle.
    """
    assert isinstance(axis_angle, np.ndarray)

    angle = np.linalg.norm(axis_angle)
    if angle < np.finfo(np.float32).tiny:
        return np.identity(3)

    axis = axis_angle / angle
    u_x, u_y, u_z = axis
    R = np.cos(angle) * np.identity(3)
    R += np.sin(angle) * np.array([0, -u_z, u_y, u_z, 0, -u_x, -u_y, u_x, 0]).reshape(3, 3)
    R += +(1.0 - np.cos(angle)) * (axis * axis[:, None])

    return R


class SMPL:
    """A NumPy implementation of SMPL."""

    def __init__(self, model_path: str | Path):
        """A NumPy implementation of SMPL.

        Arguments:
            model_path: A path to a SMPL model file (.npz).
        """
        model_path = Path(model_path)
        assert model_path.is_file(), f"{model_path} does not exist."
        assert model_path.suffix == ".npz", "Expecting a pickle file."

        params = np.load(model_path)

        self._vertex_template = params["v_template"]
        self._vertex_shape_basis = params["shapedirs"]
        self._vertex_pose_basis = params["posedirs"]

        self._joint_parent_idxs = params["kintree_table"][0]
        self._joint_regressor = params["J_regressor"]

        self._skinning_weights = params["weights"]

        self._triangles = params["f"]

        self._n_vertices = len(self._vertex_template)
        self._n_joints = len(self._joint_regressor)

        # Used to calculate pose-dependent blendshapes coefficients
        self._identity_cube = np.identity(3)[np.newaxis, ...].repeat(self._n_joints - 1, axis=0)

        # The vertex and joint positions in the bind-pose
        self._vertices_bind_pose = self._vertex_template.copy()
        self._joints_bind_pose = self._joint_regressor.dot(self._vertices_bind_pose)

        self._shape_dim = self._vertex_shape_basis.shape[-1]
        self._theta_shape = (self._n_joints, 3)

        self.beta = np.zeros(self._shape_dim, dtype=float)
        self.theta = np.zeros(self._theta_shape, dtype=float)
        self._translation = np.zeros(3, dtype=float)

        # Joint transforms in local and world space
        self._j_transforms_local = np.identity(4)[np.newaxis, ...].repeat(self._n_joints, axis=0)
        self._j_transforms_global = self._j_transforms_local.copy()

        # The vertices of the posed model
        self._vertices = self._vertices_bind_pose.copy()

        # The normals of the posed model, to be calculated
        self._normals = np.zeros_like(self._vertices)

    @property
    def beta(self) -> np.ndarray:
        """SMPL's linear shape basis parameters."""
        return self._beta

    @beta.setter
    def beta(self, value: np.ndarray) -> None:
        value = np.asarray(value)
        assert value.shape == (self._shape_dim,), f"Expecting beta to have shape ({self._shape_dim},)."
        self._beta = value
        self._update_shape()

    @property
    def joint_parent_indices(self) -> np.ndarray:
        """For each joint, the index of its parent in the skeleton hierarchy."""
        return self._readonly_view(self._joint_parent_idxs)

    @property
    def theta(self) -> np.ndarray:
        """SMPL's pose parameters - per-joint rotations in axis-angle representation."""
        return self._theta

    @theta.setter
    def theta(self, value: np.ndarray) -> None:
        value = np.asarray(value)
        assert value.shape == self._theta_shape, f"Expecting theta to have shape ({self._theta_shape},)."
        self._theta = value

    @property
    def translation(self) -> np.ndarray:
        """The 3D translation of the SMPL model."""
        return self._translation

    @translation.setter
    def translation(self, value: np.ndarray) -> None:
        value = np.asarray(value)
        assert value.shape == (3,), "Translation should be 3D."
        self._translation = value

    @staticmethod
    def _readonly_view(array: np.ndarray) -> np.ndarray:
        view = array.view()
        view.setflags(write=False)
        return view

    @property
    def joint_transforms(self) -> np.ndarray:
        """The global transforms of each joint in SMPL (read-only)."""
        self._update_joint_transforms()
        return self._readonly_view(self._j_transforms_global)

    @property
    def vertices(self) -> np.ndarray:
        """The posed vertex positions of the SMPL model (read-only)."""
        self._update_joint_transforms()
        self._update_vertices()
        return self._readonly_view(self._vertices)

    @property
    def normals(self) -> np.ndarray:
        """Normals to the surface of the posed SMPL model (read-only)."""
        self._update_joint_transforms()
        self._update_vertices()
        self._update_normals()
        return self._readonly_view(self._normals)

    @property
    def joint_positions(self) -> np.ndarray:
        """The posed joint positions of the SMPL model (read-only)."""
        self._update_joint_transforms()
        return self._readonly_view(self._j_transforms_global[:, :3, 3])

    @property
    def shape_dim(self) -> int:
        """The number of elements in SMPL's shape vector (read-only)."""
        return self._shape_dim

    @property
    def n_vertices(self) -> int:
        """The number of vertices in SMPL (read-only)."""
        return self._n_vertices

    @property
    def n_joints(self) -> int:
        """The number of joints in SMPL (read-only)."""
        return self._n_joints

    @property
    def triangles(self) -> np.ndarray:
        """The vertex triangle indices (read-only)."""
        return self._triangles

    def _update_shape(self) -> None:
        self._vertices_bind_pose = self._vertex_template + self._vertex_shape_basis.dot(self._beta)
        self._joints_bind_pose = self._joint_regressor.dot(self._vertices_bind_pose)

    def _update_joint_transforms(self) -> None:
        # Initialize joint-local pose transforms to the identity
        self._j_transforms_local = np.identity(4)[np.newaxis, ...].repeat(self._n_joints, axis=0)

        # Set the root joint translation
        self._j_transforms_local[0, :3, 3] = self._translation + self._joints_bind_pose[0]

        # Set the translational offset between each joint and its parent, excluding the root
        p_offsets = self._joints_bind_pose[1:] - self._joints_bind_pose[self._joint_parent_idxs[1:]]
        self._j_transforms_local[1:, :3, 3] = p_offsets

        # Set local rotations of each joint
        for j_idx in range(self._n_joints):
            self._j_transforms_local[j_idx, :3, :3] = axis_angle_to_rotation_matrix(self._theta[j_idx])

        # Calculate transforms of each joint in global space
        self._j_transforms_global = np.zeros_like(self._j_transforms_local)
        self._j_transforms_global[0] = self._j_transforms_local[0]
        for j_idx in range(1, self._n_joints):
            parent_idx = self._joint_parent_idxs[j_idx]
            self._j_transforms_global[j_idx] = self._j_transforms_global[parent_idx] @ self._j_transforms_local[j_idx]

    def _update_vertices(self) -> None:
        # Apply the SMPL vertex pose basis
        pose_basis_coeffs = (self._j_transforms_local[1:, :3, :3] - self._identity_cube).ravel()
        vertices = self._vertices_bind_pose + self._vertex_pose_basis.dot(pose_basis_coeffs)

        # Skinning transforms are relative to the bind pose.
        # This is the equivalent to pre-applying the inverse bind pose transform of each joint.
        skinning_transforms = self._j_transforms_global.copy()
        deltas = np.einsum("nij,nj->ni", self._j_transforms_global[:, :3, :3], self._joints_bind_pose)
        skinning_transforms[:, :3, 3] -= deltas

        # Get weighted per-vertex skinning transforms
        skinning_transforms = np.einsum("nj,jkl->nkl", self._skinning_weights, skinning_transforms)

        # Homogenize vertices
        vertices = np.hstack([vertices, np.ones((self._n_vertices, 1))])

        # Apply skinning transforms to vertices
        vertices = np.matmul(skinning_transforms, vertices[..., np.newaxis])

        # Dehomogenize, and remove additional dimension
        self._vertices = vertices[:, :3, 0]

    def _update_normals(self) -> None:
        # Calculate per-face normals for the mesh
        vs_ts = self._vertices[self._triangles]
        per_face_normals = np.cross(vs_ts[::, 1] - vs_ts[::, 0], vs_ts[::, 2] - vs_ts[::, 0])

        # For each triangle, add that triangle's normal to each vertex in the triangle
        normals = np.zeros_like(self._vertices)
        np.add.at(normals, self._triangles.ravel(), np.repeat(per_face_normals, 3, axis=0))

        # Normalize normals
        normals /= np.linalg.norm(normals, axis=1).reshape(-1, 1)

        self._normals = normals
