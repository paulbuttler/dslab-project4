import argparse
import json
import pickle
import gzip
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import ldmks, extract_roi


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", action="store_true", default=False)
    parser.add_argument("--n-ids", type=int, default=20000)
    args = parser.parse_args()
    args.data_dir = Path("data/raw/synth_body")

    vertex_indices = np.int64(
        np.load("src/visualization/vertices/complete_vertices.npy")
    )
    vertex_indices_roi = np.int64(
        np.load("src/visualization/vertices/body36_vertices.npy")
    )

    # landmarks_3d = {}
    landmarks_2d = {}
    landmarks_roi = {}
    roi = {}

    for sidx in tqdm(range(args.n_ids), desc="Processing IDs"):
        for fidx in range(5):
            uid = f"{sidx:07d}_{fidx:03d}"
            meta_file = args.data_dir / f"metadata_{uid}.json"
            if not meta_file.exists():
                print(f"File not found: {meta_file}")
                continue

            with open(meta_file, "r") as f:
                metadata = json.load(f)
            # Get camera parameters
            world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
            camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])

            ldmks_3d = ldmks.get_3d_landmarks(
                metadata, np.concatenate((vertex_indices, vertex_indices_roi))
            )
            ldmks_2d = ldmks.project_3d_to_2d(
                ldmks_3d.squeeze(), camera_to_image, world_to_camera[:3]
            )

            # landmarks_3d[uid] = ldmks_3d.squeeze()[:-36]
            landmarks_2d[uid] = ldmks_2d[:-36]
            landmarks_roi[uid] = ldmks_2d[-36:]

            # Compute ROI from sparse landmarks
            roi[uid] = extract_roi.compute_roi(ldmks_2d[-36:])

    # Save landmarks to a compressed pickle file.
    with gzip.open("data/annotations/2d_landmarks.pkl.gz", "wb") as f:
        pickle.dump(landmarks_2d, f, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open("data/annotations/2d_roi_landmarks.pkl.gz", "wb") as f:
        pickle.dump(landmarks_roi, f, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open("data/annotations/roi.pkl.gz", "wb") as f:
        pickle.dump(roi, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with gzip.open("data/annotations/3d_landmarks.pkl.gz", "wb") as f:
    #     pickle.dump(landmarks_3d, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
