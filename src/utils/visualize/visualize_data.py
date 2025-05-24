"""Helper script to visualize the SynthMoCap datasets.

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
import lzma
import subprocess
from getpass import getpass
from pathlib import Path
from tarfile import TarFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
from transformations import rotation_matrix

from .smpl_numpy import SMPL

SEMSEG_LUT = (plt.get_cmap("tab20")(np.arange(255 + 1)) * 255).astype(np.uint8)[..., :3][..., ::-1]
LDMK_CONN = {
    "face": [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [17, 18],
        [18, 19],
        [19, 20],
        [20, 21],
        [22, 23],
        [23, 24],
        [24, 25],
        [25, 26],
        [27, 28],
        [28, 29],
        [29, 30],
        [31, 32],
        [32, 33],
        [33, 34],
        [34, 35],
        [36, 37],
        [37, 38],
        [38, 39],
        [39, 40],
        [40, 41],
        [41, 36],
        [42, 43],
        [43, 44],
        [44, 45],
        [45, 46],
        [46, 47],
        [47, 42],
        [48, 49],
        [49, 50],
        [50, 51],
        [51, 52],
        [52, 53],
        [53, 54],
        [54, 55],
        [55, 56],
        [56, 57],
        [57, 58],
        [58, 59],
        [59, 48],
        [60, 61],
        [61, 62],
        [62, 63],
        [63, 64],
        [64, 65],
        [65, 66],
        [66, 67],
        [67, 60],
    ],
    "body": [
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 1],
        [5, 2],
        [6, 3],
        [7, 4],
        [8, 5],
        [9, 6],
        [10, 7],
        [11, 8],
        [12, 9],
        [13, 9],
        [14, 9],
        [15, 12],
        [16, 13],
        [17, 14],
        [18, 16],
        [19, 17],
        [20, 18],
        [21, 19],
        [22, 20],
        [23, 22],
        [24, 23],
        [25, 20],
        [26, 25],
        [27, 26],
        [28, 20],
        [29, 28],
        [30, 29],
        [31, 20],
        [32, 31],
        [33, 32],
        [34, 20],
        [35, 34],
        [36, 35],
        [37, 21],
        [38, 37],
        [39, 38],
        [40, 21],
        [41, 40],
        [42, 41],
        [43, 21],
        [44, 43],
        [45, 44],
        [46, 21],
        [47, 46],
        [48, 47],
        [49, 21],
        [50, 49],
        [51, 50],
    ],
    "hand": [
        [1, 0],
        [2, 1],
        [3, 2],
        [4, 0],
        [5, 4],
        [6, 5],
        [7, 0],
        [8, 7],
        [9, 8],
        [10, 0],
        [11, 10],
        [12, 11],
        [13, 0],
        [14, 13],
        [15, 14],
        [16, 3],
        [17, 6],
        [18, 9],
        [19, 12],
        [20, 15],
    ],
}
SMPLH_MODEL = None


def draw_transformed_3d_axes(
    image: np.ndarray,
    transform: np.ndarray,
    loc: np.ndarray,
    scale: float,
    projection_matrix: np.ndarray,
) -> None:
    """Draw a transformed set of coordinate axes, in color."""
    trsf_4x4 = np.eye(4)
    trsf_4x4[:3, :3] = transform
    axes_edges = np.array([[0, 1], [0, 2], [0, 3]])
    axes_verts = np.vstack([np.zeros((1, 3)), np.eye(3)]) * 3.0
    axes_verts = np.hstack([axes_verts, np.ones((len(axes_verts), 1))])
    axes_verts = np.array([0, 0, 10]) + axes_verts.dot(trsf_4x4.T)[:, :-1]
    projected = axes_verts.dot(projection_matrix.T)
    projected = projected[:, :2] / projected[:, 2:]

    center = np.array([image.shape[0] // 2, image.shape[1] // 2])
    projected = ((projected - center) * scale + loc).astype(int)

    ldmk_connection_pairs = projected[axes_edges].astype(int)
    for p_0, p_1 in ldmk_connection_pairs:
        cv2.line(image, tuple(p_0 + 1), tuple(p_1 + 1), (0, 0, 0), 2, cv2.LINE_AA)

    colors = np.fliplr(np.eye(3) * 255)
    for i, (p_0, p_1) in enumerate(ldmk_connection_pairs):
        cv2.line(image, tuple(p_0), tuple(p_1), colors[i], 2, cv2.LINE_AA)


def draw_landmarks(
    img: np.ndarray,
    ldmks_2d: np.ndarray,
    connectivity: list[list[int]],
    thickness: int = 1,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Drawing dots on an image."""
    if img.dtype != np.uint8:
        raise ValueError("Image must be uint8")
    if np.any(np.isnan(ldmks_2d)):
        raise ValueError("NaNs in landmarks")

    img_size = (img.shape[1], img.shape[0])

    ldmk_connection_pairs = ldmks_2d[np.asarray(connectivity).astype(int)].astype(int)
    for p_0, p_1 in ldmk_connection_pairs:
        cv2.line(img, tuple(p_0 + 1), tuple(p_1 + 1), (0, 0, 0), thickness, cv2.LINE_AA)
    for i, (p_0, p_1) in enumerate(ldmk_connection_pairs):
        cv2.line(
            img,
            tuple(p_0),
            tuple(p_1),
            (int(color[0]), int(color[1]), int(color[2])),
            thickness,
            cv2.LINE_AA,
        )

    for ldmk in ldmks_2d.astype(int):
        if np.all(ldmk > 0) and np.all(ldmk < img_size):
            cv2.circle(img, tuple(ldmk + 1), thickness + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(
                img,
                tuple(ldmk),
                thickness + 1,
                (int(color[0]), int(color[1]), int(color[2])),
                -1,
                cv2.LINE_AA,
            )


def _download_smplh() -> None:
    print("Downloading SMPL-H...")
    username = input("Username for https://mano.is.tue.mpg.de/: ")
    password = getpass("Password for https://mano.is.tue.mpg.de/: ")
    out_path = Path(__file__).parent / "smplh" / "smplh.tar.xz"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    url = "https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz"
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
    with lzma.open(out_path) as fd:
        with TarFile(fileobj=fd) as f:
            f.extractall(out_path.parent)
    out_path.unlink()


def _get_smplh() -> SMPL:
    global SMPLH_MODEL

    if SMPLH_MODEL is None:
        model_path = Path(__file__).parent / "smplh" / "neutral" / "model.npz"
        if not model_path.exists():
            _download_smplh()

        SMPLH_MODEL = SMPL(model_path)

    return SMPLH_MODEL


def _render_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    world_to_cam: np.ndarray,
    cam_to_img: np.ndarray,
    resolution: tuple[int, int],
) -> np.ndarray:
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])

    camera_pr = pyrender.IntrinsicsCamera(
        cx=cam_to_img[0, 2],
        cy=cam_to_img[1, 2],
        fx=cam_to_img[0, 0],
        fy=cam_to_img[1, 1],
        zfar=5000.0,
        name="cam",
    )
    scene = pyrender.Scene(ambient_light=[100, 100, 100], bg_color=[0, 0, 0, 0])

    # OpenCV to OpenGL convention
    world_to_cam_gl = np.linalg.inv(world_to_cam).dot(rotation_matrix(np.pi, [1, 0, 0]))
    camera_node = pyrender.Node(camera=camera_pr, matrix=world_to_cam_gl)
    scene.add_node(camera_node)

    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    R1 = rotation_matrix(np.radians(25), [0, 1, 0])
    R2 = rotation_matrix(np.radians(-30), [1, 0, 0])
    key_pose = world_to_cam_gl.dot(R1.dot(R2))
    scene.add(key_light, pose=key_pose)

    back_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    R1 = rotation_matrix(np.radians(-150), [0, 1, 0])
    back_pose = world_to_cam_gl.dot(R1)
    scene.add(back_light, pose=back_pose)

    mesh_trimesh = trimesh.Trimesh(vertices, triangles, process=False)
    colors = np.repeat([[255, 61, 13]], len(vertices), axis=0)
    mesh_trimesh.visual.vertex_colors = colors
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
    mesh_pyrender.primitives[0].material.roughnessFactor = 0.6
    mesh_pyrender.primitives[0].material.alphaMode = "OPAQUE"
    scene.add(mesh_pyrender)

    rendered_img, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.ALL_SOLID)
    renderer.delete()

    return rendered_img.astype(float) / 255


def draw_mesh(
    image: np.ndarray,
    identity: np.ndarray,
    pose: np.ndarray,
    translation: np.ndarray,
    world_to_cam: np.ndarray,
    cam_to_img: np.ndarray,
) -> None:
    """Draw a mesh from identity, pose, and translation parameters."""
    smplh = _get_smplh()
    smplh.beta = identity[: smplh.shape_dim]
    smplh.theta = pose
    smplh.translation = translation
    render = _render_mesh(smplh.vertices, smplh.triangles, world_to_cam, cam_to_img, image.shape[:2][::-1])
    # alpha blend
    return (
        ((image.astype(np.float64) / 255) * (1 - 0.75 * render[..., -1:]) + render[..., :3] * 0.75 * render[..., -1:])
        * 255
    ).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--n-ids", type=int, default=25)
    parser.add_argument("--n-frames", type=int, default=2)
    args = parser.parse_args()

    dataset_type = args.data_dir.stem.split("_")[-1]

    for sidx in range(args.n_ids):
        for fidx in range(args.n_frames):
            meta_file = args.data_dir / f"metadata_{sidx:07d}_{fidx:03d}.json"
            img_file = args.data_dir / f"img_{sidx:07d}_{fidx:03d}.jpg"

            if not meta_file.exists() or not img_file.exists():
                continue

            with open(meta_file, "r") as f:
                metadata = json.load(f)

            frame = cv2.imread(str(img_file))
            vis_imgs = [frame]
            ldmks_2d = np.asarray(metadata["landmarks"]["2D"])
            ldmk_vis = frame.copy()
            draw_landmarks(
                ldmk_vis,
                ldmks_2d,
                LDMK_CONN[dataset_type],
            )
            vis_imgs.append(ldmk_vis)

            if dataset_type != "hand":
                seg_file = args.data_dir / f"segm_parts_{sidx:07d}_{fidx:03d}.png"
                seg_parts = cv2.imread(str(seg_file), cv2.IMREAD_GRAYSCALE)
                parts = SEMSEG_LUT[seg_parts]
                for idx, mask_name in enumerate(
                    [
                        "hair",
                        "beard",
                        "eyebrows",
                        "eyelashes",
                        "glasses",
                        "headwear",
                        "facewear",
                        "clothing",
                    ]
                ):
                    mask_path = args.data_dir / f"segm_{mask_name}_{sidx:07d}_{fidx:03d}.png"
                    if not mask_path.exists():
                        continue
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 100
                    parts[mask] = SEMSEG_LUT[idx + (13 if dataset_type == "face" else 17)]
                vis_imgs.append(parts)

            if dataset_type in {"hand", "body"}:
                vis_imgs.append(
                    draw_mesh(
                        frame.copy(),
                        np.asarray(metadata["body_identity"]),
                        np.asarray(metadata["pose"]),
                        np.asarray(metadata["translation"]),
                        np.asarray(metadata["camera"]["world_to_camera"]),
                        np.asarray(metadata["camera"]["camera_to_image"]),
                    )
                )
            else:
                pose_vis = frame.copy()
                cam_to_img = np.asarray(metadata["camera"]["camera_to_image"])
                head_loc = np.mean(metadata["landmarks"]["2D"], axis=0)
                left_eye_loc = metadata["landmarks"]["2D"][-2]
                right_eye_loc = metadata["landmarks"]["2D"][-1]
                draw_transformed_3d_axes(
                    pose_vis,
                    np.asarray(metadata["head_pose"]),
                    head_loc,
                    0.5,
                    cam_to_img,
                )
                draw_transformed_3d_axes(
                    pose_vis,
                    np.asarray(metadata["left_eye_pose"]),
                    left_eye_loc,
                    0.1,
                    cam_to_img,
                )
                draw_transformed_3d_axes(
                    pose_vis,
                    np.asarray(metadata["right_eye_pose"]),
                    right_eye_loc,
                    0.1,
                    cam_to_img,
                )
                vis_imgs.append(pose_vis)

            cv2.imshow(args.data_dir.stem, np.hstack(vis_imgs))
            k = cv2.waitKey(0)
            if k in {ord("q"), 27}:
                cv2.destroyAllWindows()
                exit()


if __name__ == "__main__":
    main()
