import argparse
import pickle
import gzip
import numpy as np
import cv2
from pathlib import Path


def compute_roi(landmarks, image=None, size=512):
    """Crop a square ROI that tightly contains all landmarks."""
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    # 8% padding seems to be enough to contain all bodyparts even in weird poses
    box_size = max(x_max - x_min, y_max - y_min) * 1.08
    box_size = min(size, box_size)

    x_min = int(np.floor(center_x - box_size / 2))
    y_min = int(np.floor(center_y - box_size / 2))
    x_max = int(np.ceil(center_x + box_size / 2))
    y_max = int(np.ceil(center_y + box_size / 2))

    if image is not None:
        # Visualize ROI and landmarks on the image
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

    args.data_dir = Path("data/raw/synth_body")

    if args.sidx is None:
        args.sidx = np.random.randint(0, 1000)
    if args.fidx is None:
        args.fidx = np.random.randint(0, 5)

    uid = f"{args.sidx:07d}_{args.fidx:03d}"
    img_file = args.data_dir / f"img_{uid}.jpg"

    if not img_file.exists():
        print(f"File not found: {img_file}")
        exit(1)

    with gzip.open("data/annotations/body_ldmks_roi.pkl.gz", "rb") as f:
        landmarks_dict = pickle.load(f)
    landmarks = landmarks_dict[uid]

    # with gzip.open("data/annotations/body_meta.pkl.gz", "rb") as f:
    #     body_dict = pickle.load(f)
    # landmarks = body_dict[uid]["ldmks_2d"]
    img = cv2.imread(str(img_file))

    _ = compute_roi(landmarks, img)
