import argparse
import joblib
import numpy as np
import cv2
from pathlib import Path
import torch


def compute_roi(landmarks, image=None, margin=0.08, input_size=512):
    """
    Compute a square ROI containing all landmarks, with optional margin.
    Handles both single numpy array and batched torch tensor inputs.
    """

    # PyTorch batched case: (B, N, 2)
    if isinstance(landmarks, torch.Tensor) and landmarks.ndim == 3:
        x_min = landmarks[..., 0].min(dim=1).values
        y_min = landmarks[..., 1].min(dim=1).values
        x_max = landmarks[..., 0].max(dim=1).values
        y_max = landmarks[..., 1].max(dim=1).values

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        box_size = (x_max - x_min).maximum(y_max - y_min) * (1.0 + margin)
        box_size = box_size.minimum(landmarks.new_tensor(input_size))

        x_min = (center_x - box_size / 2).floor().to(torch.int32)
        y_min = (center_y - box_size / 2).floor().to(torch.int32)
        x_max = (center_x + box_size / 2).ceil().to(torch.int32)
        y_max = (center_y + box_size / 2).ceil().to(torch.int32)

        rois = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        return rois
    
    # NumPy single sample: (N, 2)
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    box_size = max(x_max - x_min, y_max - y_min) * (1.0 + margin)
    box_size = min(input_size, box_size)

    x_min = int(np.floor(center_x - box_size / 2))
    y_min = int(np.floor(center_y - box_size / 2))
    x_max = int(np.ceil(center_x + box_size / 2))
    y_max = int(np.ceil(center_y + box_size / 2))

    if image is not None:
        # Visualize ROI and landmarks
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        for x, y in landmarks:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.imshow("Square ROI", image)
        cv2.waitKey(0)

    return np.array([x_min, y_min, x_max, y_max])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", action="store_true", default=False)
    parser.add_argument("--sidx", type=int, default=None)
    parser.add_argument("--fidx", type=int, default=None)
    args = parser.parse_args()

    args.data_dir = (Path("data/raw/synth_body") if not args.hand else Path("data/raw/synth_hand"))

    if args.sidx is None:
        args.sidx = np.random.randint(0, 20000)
    if args.fidx is None:
        args.fidx = np.random.randint(0, 5)

    uid = f"{args.sidx:07d}_{args.fidx:03d}"
    img_file = args.data_dir / f"img_{uid}.jpg"

    if not img_file.exists():
        print(f"File not found: {img_file}")
        exit(1)

    if not args.hand:
        landmarks_dict = joblib.load("data/annotations/body_ldmks_roi.pkl")
        landmarks = landmarks_dict[uid]
        margin = 0.08
    else:
        hand_meta = joblib.load("data/annotations/hand_meta.pkl")
        landmarks = hand_meta[uid]["ldmks_2d"]
        margin = 0.10

    img = cv2.imread(str(img_file))
    _ = compute_roi(landmarks, img, margin, img.shape[-2])
