"""Helper script to download the SynthMoCap datasets.

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

import argparse
import json
import subprocess
import sys
import tarfile
from getpass import getpass
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
from tqdm import tqdm

MANO_N_J = 15
SMPL_H_N_J = 22
LEFT_HAND = SMPL_H_N_J
RIGHT_HAND = SMPL_H_N_J + MANO_N_J

MANO_FILENAME = "manoposesv10"
MOSH_FILENAME = "MoSh"
POSELIM_FILENAME = "PosePrior"

MANO_LEFT_DATA = None
MANO_RIGHT_DATA = None

N_PARTS = 20


def _download_mpii_file(username: str, password: str, domain: str, file: str, out_path: Path) -> None:
    out_path.parent.mkdir(exist_ok=True, parents=True)
    url = f"https://download.is.tue.mpg.de/download.php?domain={domain}&resume=1&sfile={file}"
    try:
        subprocess.check_call(
            [
                "wget",
                "--post-data",
                f"username={username}&password={password}",
                url,
                "-O",
                out_path.as_posix(),
                "--no-check-certificate",
                "--continue",
            ]
        )
    except FileNotFoundError as exc:
        raise RuntimeError("wget not found, please install it") from exc
    except subprocess.CalledProcessError as exc:
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError("Download failed, check your login details") from exc


def get_mano(out_dir: Path) -> None:
    """Download MANO data."""
    print("Downloading MANO...")
    username = input("Username for https://mano.is.tue.mpg.de/: ")
    password = getpass("Password for https://mano.is.tue.mpg.de/: ")
    _download_mpii_file(
        username,
        password,
        "mano",
        f"{MANO_FILENAME}.zip",
        out_dir / f"{MANO_FILENAME}.zip",
    )


def get_amass(out_dir: Path) -> None:
    """Download AMASS data."""
    print("Downloading AMASS...")
    username = input("Username for https://amass.is.tue.mpg.de/: ")
    password = getpass("Password for https://amass.is.tue.mpg.de/: ")
    _download_mpii_file(
        username,
        password,
        "amass",
        f"amass_per_dataset/smplh/gender_specific/mosh_results/{MOSH_FILENAME}.tar.bz2",
        out_dir / f"{MOSH_FILENAME}.tar.bz2",
    )
    _download_mpii_file(
        username,
        password,
        "amass",
        f"amass_per_dataset/smplh/gender_specific/mosh_results/{POSELIM_FILENAME}.tar.bz2",
        out_dir / f"{POSELIM_FILENAME}.tar.bz2",
    )


def extract(data_path: Path, out_path: Optional[Path] = None) -> None:
    """Extract the data from the given path."""
    print(f"Extracting {data_path.name}...")
    if data_path.suffix == ".zip":
        out_path = out_path or data_path.parent / data_path.stem
        with ZipFile(data_path) as f:
            f.extractall(out_path)
    elif data_path.suffix == ".bz2":
        out_path = out_path or data_path.parent / data_path.name.replace(".tar.bz2", "")
        with tarfile.open(data_path, "r:bz2") as f:
            f.extractall(out_path)
    else:
        raise ValueError(f"Unknown file type {data_path.suffix}")


def _mano_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the MANO data."""
    global MANO_LEFT_DATA, MANO_RIGHT_DATA
    if MANO_LEFT_DATA is None:
        MANO_LEFT_DATA = np.load(
            data_dir / f"{MANO_FILENAME}/mano_poses_v1_0/handsOnly_REGISTRATIONS_r_lm___POSES___L.npy"
        )
    if MANO_RIGHT_DATA is None:
        MANO_RIGHT_DATA = np.load(
            data_dir / f"{MANO_FILENAME}/mano_poses_v1_0/handsOnly_REGISTRATIONS_r_lm___POSES___R.npy"
        )
    return MANO_LEFT_DATA, MANO_RIGHT_DATA


def _process_meta(args: tuple[Path, Path]) -> None:
    metadata_fn, data_dir = args
    mano_left, mano_right = _mano_data(data_dir)
    with open(metadata_fn, "r") as f:
        metadata = json.load(f)
        if isinstance(metadata["pose"][1], str):
            # body pose comes from AMASS
            seq_name: str = metadata["pose"][1]
            frame = int(seq_name.split("_")[-2])
            assert int(seq_name.split("_")[-1]) == 0
            seq_path = Path("/".join(seq_name.split("/")[1:])).with_suffix(".npz").as_posix()
            if seq_name.startswith("MoSh_MPI_MoSh"):
                # fix paths to match downloaded data
                seq_path = seq_path.replace("Data/moshpp_fits_SMPL", "MPI_mosh")
                seq_path = seq_path.replace(".npz", "_poses.npz")
                if not (data_dir / MOSH_FILENAME / seq_path).exists():
                    # there is a sequence incorrectly named with _poses_poses
                    seq_path = seq_path.replace(".npz", "_poses.npz")
                seq_data = np.load(data_dir / MOSH_FILENAME / seq_path)
            elif seq_name.startswith("MoSh_MPI_PoseLimits"):
                # fix paths to match downloaded data
                seq_path = seq_path.replace("Data/moshpp_fits_SMPL", "MPI_Limits")
                seq_path = seq_path.replace(".npz", "_poses.npz")
                seq_data = np.load(data_dir / POSELIM_FILENAME / seq_path)
            else:
                raise RuntimeError(f"Unknown sequence name {seq_name}")
            # we resampled to ~30 fps so have to adjust the frame number
            frame_step = int(np.floor(seq_data["mocap_framerate"] / 30))
            seq = seq_data["poses"][::frame_step]
            # exclude root joint
            metadata["pose"][1:SMPL_H_N_J] = seq[frame].reshape((-1, 3))[1:SMPL_H_N_J].tolist()
        if isinstance(metadata["pose"][LEFT_HAND], str):
            # left hand comes from MANO
            idx = int(metadata["pose"][LEFT_HAND].split("_")[1])
            metadata["pose"][LEFT_HAND:RIGHT_HAND] = mano_left[idx].reshape((MANO_N_J, 3)).tolist()
        if isinstance(metadata["pose"][RIGHT_HAND], str):
            # right hand comes from MANO
            idx = int(metadata["pose"][RIGHT_HAND].split("_")[1])
            metadata["pose"][RIGHT_HAND:] = mano_right[idx].reshape((MANO_N_J, 3)).tolist()
    with open(metadata_fn, "w") as f:
        json.dump(metadata, f, indent=4)


def download_synthmocap_data(data_dir: Path, dataset: str, zip_dir: Path, single_id: bool, single_chunck: bool) -> None:
    """Download one of the SynthMoCap datasets."""
    data_dir.mkdir(exist_ok=True, parents=True)
    zip_dir.mkdir(exist_ok=True, parents=True)
    parts = (
        [f"{dataset}_sample.zip"]
        if single_id
        else [f"{dataset}_{i:02d}.zip" for i in range(1, 2 if single_chunck else N_PARTS + 1)]
    )
    for part in parts:
        out_path = zip_dir / part
        print(f"Downloading {part}...")
        url = f"https://facesyntheticspubwedata.z6.web.core.windows.net/sga-2024-synthmocap/{part}"
        try:
            subprocess.check_call(
                [
                    "wget",
                    url,
                    "-O",
                    str(out_path),
                    "--no-check-certificate",
                    "--continue",
                    "--secure-protocol=TLSv1_2",
                ]
            )
        except FileNotFoundError as exc:
            raise RuntimeError("wget not found, please install it") from exc
        except subprocess.CalledProcessError:
            print("Download failed")
            if out_path.exists():
                out_path.unlink()
            sys.exit(1)
        extract(out_path, data_dir / dataset)
        out_path.unlink()


def process_metadata(data_dir: Path, dataset_name: str) -> None:
    """Process the metadata to include the correct pose data."""
    metadata_files = list((data_dir / dataset_name).glob("*.json"))
    with Pool() as p:
        list(
            tqdm(
                p.imap(
                    _process_meta,
                    [(metadata_fn, data_dir) for metadata_fn in metadata_files],
                ),
                total=len(metadata_files),
                desc="Processing metadata",
            )
        )


def main() -> None:
    """Download and process the dataset."""
    parser = argparse.ArgumentParser(description="Download SynthMoCap datasets")
    parser.add_argument("--output-dir", type=Path, help="Output directory", required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to download",
        choices=["face", "body", "hand"],
        required=True,
    )
    parser.add_argument(
        "--single-id",
        action="store_true",
        help="Only download one subject from the dataset",
    )
    parser.add_argument(
        "--single-chunk",
        action="store_true",
        help="Only download one chunk from the dataset",
    )
    args = parser.parse_args()
    assert not (args.single_id and args.single_chunk), "Cannot specify both single-id and single-chunk"
    dataset_name = f"synth_{args.dataset}"
    data_dir = Path(args.output_dir)
    if args.dataset != "face":
        # download data from MPII sources
        if not (data_dir / MOSH_FILENAME).exists() or not (data_dir / POSELIM_FILENAME).exists():
            get_amass(data_dir)
        if not (data_dir / MANO_FILENAME).exists():
            get_mano(data_dir)
        # extract the data
        for path in list(data_dir.glob("*.zip")) + list(data_dir.glob("*.bz2")):
            extract(path)
            path.unlink()
    # download the SynthMoCap dataset
    zip_dir = data_dir / f"{dataset_name}_zip"
    download_synthmocap_data(data_dir, dataset_name, zip_dir, args.single_id, args.single_chunk)
    zip_dir.rmdir()
    if args.dataset != "face":
        # process the metadata
        process_metadata(data_dir, dataset_name)


if __name__ == "__main__":
    main()
