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

    body_meta = {}
    body_ldmks_roi = {}
    body_ldmks_3d = {}

    for sidx in tqdm(range(args.n_ids), desc="Processing IDs"):
        for fidx in range(5):
            uid = f"{sidx:07d}_{fidx:03d}"
            meta_file = args.data_dir / f"metadata_{uid}.json"
            if not meta_file.exists():
                print(f"File not found: {meta_file}")
                continue

            with open(meta_file, "r") as f:
                metadata = json.load(f)

            world_to_camera = np.asarray(metadata["camera"]["world_to_camera"])
            camera_to_image = np.asarray(metadata["camera"]["camera_to_image"])
            shape = np.array(metadata["body_identity"])
            pose = np.array(metadata["pose"])

            ldmks_3d = ldmks.get_3d_landmarks(
                metadata, np.concatenate((vertex_indices, vertex_indices_roi))
            )
            ldmks_2d = ldmks.project_3d_to_2d(
                ldmks_3d.squeeze(), camera_to_image, world_to_camera[:3]
            )

            body_meta[uid] = {
                "shape": shape,
                "pose": pose,
                "roi": extract_roi.compute_roi(ldmks_2d[-36:]),
                "ldmks_2d": ldmks_2d[:-36],
            }
            body_ldmks_roi[uid] = ldmks_2d[-36:]
            body_ldmks_3d[uid] = ldmks_3d.squeeze()[:-36]

    # Save landmarks and relevant metadata to compressed pkl files
    with gzip.open("data/annotations/meta_body.pkl.gz", "wb") as f:
        pickle.dump(body_meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open("data/annotations/body_ldmks_roi.pkl.gz", "wb") as f:
        pickle.dump(body_ldmks_roi, f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open("data/annotations/body_ldmks_3d.pkl.gz", "wb") as f:
        pickle.dump(body_ldmks_3d, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
