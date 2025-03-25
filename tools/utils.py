import numpy as np


def set_camera_from_extrinsics(camera, transform):
    # Extract rotation (R) and translation (T)
    R = transform[:3, :3]  # 3x3 rotation matrix
    T = transform[:3, 3]   # 3x1 translation vector

    # Compute camera position in world space
    camera_position = -np.dot(R.T, T)

    # Extract forward (-Z) and up (+Y) vectors
    forward = -R.T[:, 2]  # Third column of R, negated
    up = -R.T[:, 1]        # Second column of R

    # Print results
    print("Camera Position:", camera_position)
    print("Forward Direction:", forward)
    print("Up Vector:", up)

    camera.position = camera_position
    camera.up = up
    camera.target = forward