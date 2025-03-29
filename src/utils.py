import cv2
import numpy as np

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
    ldmk_connection_pairs = ldmks_2d[np.asarray(connectivity).astype(int)].astype(int)

    for p_0, p_1 in ldmk_connection_pairs:
        cv2.line(img, tuple(p_0 + 1), tuple(p_1 + 1), (0, 0, 0), thickness, cv2.LINE_AA)
    for p_0, p_1 in ldmk_connection_pairs:
        cv2.line(img, tuple(p_0), tuple(p_1), color, thickness, cv2.LINE_AA)

    for ldmk in ldmks_2d.astype(int):
        if np.all(ldmk > 0) and np.all(ldmk < img_size):
            cv2.circle(img, tuple(ldmk + 1), thickness + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(ldmk), thickness + 1, color, -1, cv2.LINE_AA)