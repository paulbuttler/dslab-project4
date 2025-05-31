import os
import cv2
import numpy as np
from contextlib import redirect_stdout
from aitviewer.models.smpl import SMPLLayer  # type: ignore
from aitviewer.renderables.smpl import SMPLSequence  # type: ignore

# Copied from the repository: https://github.com/microsoft/SynthMoCap and adapted for our use
# Copyright (c) 2024 Microsoft Corporation
def draw_landmarks(
    img: np.ndarray,
    ldmks_2d: np.ndarray,
    connectivity: list[list[int]],
    thickness: int = 1,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draws 2D landmarks on an image with connections."""
    if img.dtype != np.uint8:
        raise ValueError("Image must be uint8")
    if np.any(np.isnan(ldmks_2d)):
        raise ValueError("NaNs in landmarks")

    img_size = (img.shape[1], img.shape[0])

    if connectivity is not None:
        ldmk_connection_pairs = ldmks_2d[np.asarray(connectivity).astype(int)].astype(int)

        for p_0, p_1 in ldmk_connection_pairs:
            cv2.line(img, tuple(p_0 + 1), tuple(p_1 + 1), (0, 0, 0), thickness, cv2.LINE_AA)
        for p_0, p_1 in ldmk_connection_pairs:
            cv2.line(img, tuple(p_0), tuple(p_1), color, thickness, cv2.LINE_AA)

    for ldmk in ldmks_2d.astype(int):
        if np.all(ldmk > 0) and np.all(ldmk < img_size):
            cv2.circle(
                img,
                tuple(ldmk + 1),
                thickness + (1 if connectivity is not None else 0),
                (0, 0, 0),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                img,
                tuple(ldmk),
                thickness + (1 if connectivity is not None else 0),
                color,
                -1,
                cv2.LINE_AA,
            )


def project_3d_to_2d(
    points_3d: np.ndarray, camera_intrinsics: np.ndarray, camera_extrinsics: np.ndarray
) -> np.ndarray:
    """Projects 3D points to 2D using camera extrinsics and intrinsics."""
    assert points_3d.shape[1] == 3, "Points must be in 3D space."
    assert camera_intrinsics.shape == (3, 3), "Camera intrinsics must be a 3x3 matrix."
    assert camera_extrinsics.shape == (3, 4), "Camera extrinsics must be a 3x4 matrix."

    X_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    X_cam = X_hom @ camera_extrinsics.T
    x_proj = X_cam @ camera_intrinsics.T
    x_proj /= x_proj[:, 2][:, np.newaxis]
    return x_proj[:, :2]


def get_3d_landmarks(metadata, vertex_indices):
    """Generate 3D landmarks from metafile given vertex indices."""

    # Extract metadata.
    body_identity = np.asarray(metadata["body_identity"])
    pose = np.asarray(metadata["pose"])
    translation = np.asarray(metadata["translation"])

    global_orient = pose[0].reshape(1, -1)
    body_pose = pose[1:22].reshape(1, -1)
    left_hand_pose = pose[22:37].reshape(1, -1)
    right_hand_pose = pose[37:].reshape(1, -1)

    # Create a SMPL sequence.
    with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
        smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=16)
        smpl_seq = SMPLSequence(
            smpl_layer=smpl_layer,
            poses_body=body_pose,
            betas=body_identity.reshape(1, -1),
            poses_root=global_orient,
            poses_left_hand=left_hand_pose,
            poses_right_hand=right_hand_pose,
            trans=translation.reshape(1, -1),
        )

    return smpl_seq.vertices[:, vertex_indices] + smpl_seq.position[np.newaxis]
