import argparse
import numpy as np
import cv2
from pathlib import Path
import os


def crop_hand_roi(image, landmarks, roi_size=256):
    """Crop a square ROI that tightly contains all landmarks with padding."""
    h, w = image.shape[:2]

    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    box_size = max(x_max - x_min, y_max - y_min)
    box_size *= 1.1  # margin

    x1 = int(np.floor(center_x - box_size / 2))
    y1 = int(np.floor(center_y - box_size / 2))
    x2 = int(np.ceil(center_x + box_size / 2))
    y2 = int(np.ceil(center_y + box_size / 2))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    image_padded = cv2.copyMakeBorder(
        image,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    cropped = image_padded[y1:y2, x1:x2]
    roi_resized = cv2.resize(cropped, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)

    landmarks_padded = landmarks + np.array([pad_left, pad_top])
    landmarks_cropped = landmarks_padded - np.array([x1, y1])
    scale = roi_size / (x2 - x1)
    landmarks_transformed = landmarks_cropped * scale

    return roi_resized, landmarks_transformed


def draw_landmarks(image, landmarks, radius=2, color=(0, 255, 0)):
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), radius, color, -1)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark_path", type=Path, default="tools/results/projected_2d_landmarks.npy")
    parser.add_argument("--image_path", type=Path, default="tools/results/projected_landmark_overlay.jpg")
    parser.add_argument("--output_dir", type=Path, default="tools/results")
    args = parser.parse_args()

    # === Load 2D projected landmarks and image ===
    ldmks_2d = np.load(str(args.landmark_path))
    image = cv2.imread(str(args.image_path))
    assert image is not None, f"Image not found at {args.image_path}"

    # === Crop ROI and transform landmarks ===
    roi_image, ldmks_transformed = crop_hand_roi(image, ldmks_2d)

    # === Visualize landmarks ===
    original_with_landmarks = draw_landmarks(image.copy(), ldmks_2d)
    vis_img = draw_landmarks(roi_image.copy(), ldmks_transformed)

    # === Save outputs ===
    args.output_dir.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(args.output_dir / "original_with_landmarks.png"), original_with_landmarks)
    cv2.imwrite(str(args.output_dir / "roi_with_landmarks.png"), vis_img)
    np.save(str(args.output_dir / "landmarks_transformed.npy"), ldmks_transformed)

    print("Saved ROI and landmark overlays to:", args.output_dir)
