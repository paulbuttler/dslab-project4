import argparse
import json
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import ldmks, roi

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", action="store_true", default=False)
    parser.add_argument("--n-ids", type=int, default=20000)
    args = parser.parse_args()
    args.data_dir = (Path("data/raw/synth_body") if not args.hand else Path("data/raw/synth_hand"))

    body_indices = np.int64(np.load("src/visualization/vertices/complete_vertices.npy"))
    body_indices_roi = np.int64(np.load("src/visualization/vertices/body36_vertices.npy"))
    hand_indices = np.int64(np.load("src/visualization/vertices/left_hand_vertices.npy"))

    body_meta = {}
    body_ldmks_roi = {}
    hand_meta = {}

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
            shape = np.asarray(metadata["body_identity"])
            pose = np.asarray(metadata["pose"])
            translation = np.asarray(metadata["translation"])

            if not args.hand:

                ldmks_3d = ldmks.get_3d_landmarks(metadata, np.concatenate((body_indices, body_indices_roi)))
                ldmks_2d = ldmks.project_3d_to_2d(ldmks_3d.squeeze(), camera_to_image, world_to_camera[:3])

                body_meta[uid] = {
                    "shape": shape,
                    "pose": pose,
                    "translation": translation,
                    "roi": roi.compute_roi(landmarks=ldmks_2d[-36:], margin=0.08),
                    "ldmks_2d": ldmks_2d[:-36],
                }
                body_ldmks_roi[uid] = ldmks_2d[-36:]

            else:

                ldmks_3d = ldmks.get_3d_landmarks(metadata, hand_indices)
                ldmks_2d = ldmks.project_3d_to_2d(ldmks_3d.squeeze(), camera_to_image, world_to_camera[:3])

                hand_meta[uid] = {
                    "shape": shape,
                    "pose": pose,
                    "translation": translation,
                    "roi": roi.compute_roi(landmarks=ldmks_2d, margin=0.10),
                    "ldmks_2d": ldmks_2d,
                }

    # Save landmarks and relevant metadata to compressed files
    if not args.hand:
        joblib.dump(body_meta, "data/annot/body_meta.pkl")
        joblib.dump(body_ldmks_roi, "data/annot/body_ldmks_roi.pkl")
    else:
        joblib.dump(hand_meta, "data/annot/hand_meta.pkl")

if __name__ == "__main__":
    main()
