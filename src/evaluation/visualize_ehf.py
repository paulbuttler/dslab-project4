import pickle
import cv2
import numpy as np
from pathlib import Path
from aitviewer.models.smpl import SMPLLayer  # type: ignore
from aitviewer.renderables.smpl import SMPLSequence  # type: ignore
from aitviewer.renderables.billboard import Billboard  # type: ignore
from aitviewer.scene.camera import OpenCVCamera  # type: ignore
from aitviewer.renderables.spheres import Spheres  # type: ignore
from aitviewer.viewer import Viewer  # type: ignore
from utils.rottrans import rot2aa


if __name__ == "__main__":

    data_dir = Path("data/raw/EHF")

    idx = np.random.randint(1, 2)

    img_file = data_dir / f"{idx:02d}_img.jpg"
    meta_file = data_dir / "smplh" / f"{idx:02d}_align.pkl"

    if not img_file.exists() or not meta_file.exists():
        print(f"Image or metadata file not found: {img_file} or {meta_file}")
        exit(1)

    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)

    # Extract pose and shape
    shape = metadata["betas"][0].cpu().detach().numpy()
    pose = metadata["full_pose"][0].cpu().detach()
    pose = rot2aa(pose).numpy()
    transl = metadata["transl"].cpu().detach().numpy() if metadata["transl"] is not None else np.zeros(3)

    # Camera intrinsics
    K = np.array([[1498.22426237, 0.0, 790.263706], [0.0, 1498.22426237, 578.90334], [0.0, 0.0, 1.0]])

    # Camera extrinsics
    rvec = np.array([[-2.98747896], [0.01172457], [-0.05704687]])
    tvec = np.array([[-0.03609917], [0.43416458], [2.37101226]])
    R, _ = cv2.Rodrigues(rvec)
    E = np.hstack((R, tvec))

    # Create a SMPL sequence.
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=16)
    smpl_seq = SMPLSequence(
        smpl_layer=smpl_layer,
        trans=transl.reshape(1, -1),  # TODO: Translation is not included in conversion but needed!!
        poses_root=pose[0].reshape(1, -1),
        poses_body=pose[1:22].reshape(1, -1),
        poses_left_hand=pose[22:37].reshape(1, -1),
        poses_right_hand=pose[37:].reshape(1, -1),
        betas=shape.reshape(1, -1),
    )

    # Load the input image.
    input_img = cv2.imread(img_file)
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    cols, rows = input_img.shape[1], input_img.shape[0]

    # Create Viewer and OpenCV Camera
    v = Viewer(size=(cols, rows))
    camera = OpenCVCamera(K, E, cols, rows, viewer=v)

    # Display the set of manually created vertices.
    vertex_indices = np.int64(np.load("src/visualization/vertices/complete_vertices.npy"))
    dense_ldmks_3d = smpl_seq.vertices[:, vertex_indices] + smpl_seq.position[np.newaxis]

    print("Number of Vertices:", dense_ldmks_3d.shape[1])
    spheres = Spheres(
        dense_ldmks_3d,
        name="Body_Vertices",
        radius=0.005,
        color=(0.0, 0.0, 1.0, 1.0),
    )

    billboard = Billboard.from_camera_and_distance(
        camera,
        5.0,
        cols,
        rows,
        [img_rgb],
    )

    v.scene.add(billboard, spheres, smpl_seq, camera)
    v.set_temp_camera(camera)

    # Viewer settings.
    v.scene.floor.enabled = False
    v.scene.origin.enabled = False
    v.shadows_enabled = False

    v.run()
