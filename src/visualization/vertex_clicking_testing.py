# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import sys
from pathlib import Path

import torch


sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import numpy as np
from datasets.synth_dataset import SynDataset
from datasets.SSP_3D_dataset import SSP_3D_Dataset

from aitviewer.models.smpl import SMPLLayer  # type: ignore
from aitviewer.renderables.smpl import SMPLSequence  # type: ignore
from aitviewer.renderables.spheres import Spheres  # type: ignore
from aitviewer.viewer import Viewer
from torch.utils.data import Dataset, DataLoader
from models.load import load_model
from utils.initialization import initial_pose_estimation

from datasets.EHF_dataset import EHF_Dataset
from utils.evaluation import get_full_shape_and_pose  # type: ignore


class ClickingViewer(Viewer):
    """
    This viewer just allows to place spheres onto vertices that we clicked on with the mouse.
    This only works if the viewer is in "inspection" mode (Click I).
    """

    title = "Clicking Viewer"

    def __init__(self, body_part, **kwargs):
        super().__init__(**kwargs)
        # Initialize an empty list to store the vertices we clicked on.
        self.clicked_vertices = []
        self.body_part = body_part
        if self.body_part == "hand":
            self.file_path = "src/visualization/vertices/hand_vertices.npy"
        elif self.body_part == "body":
            self.file_path = "src/visualization/vertices/body_vertices.npy"
        elif self.body_part == "body36":
            self.file_path = "src/visualization/vertices/body36_vertices.npy"
        elif self.body_part == "head":
            self.file_path = "src/visualization/vertices/head_vertices.npy"

    def save_clicked_vertices(self):
        """
        Save the clicked vertices to the .npy file. If the file already exists,
        append the new data to the existing data.
        """
        if os.path.exists(self.file_path):
            # Load existing data and append new data
            existing_data = np.load(self.file_path)
            updated_data = np.unique(np.concatenate((existing_data, np.array(self.clicked_vertices))))
        else:
            # No existing file, just save the new data
            updated_data = np.array(self.clicked_vertices)

        # Save the updated data back to the file
        np.save(self.file_path, updated_data)
        print(f"Clicked vertices saved to {self.file_path}")

    def add_virtual_marker(self, intersection):
        # Create a marker sequence for the entire sequence at once.
        # First get the positions.
        seq = intersection.node
        positions = seq.vertices[:, intersection.vert_id : intersection.vert_id + 1] + seq.position[np.newaxis]

        # Append the clicked vertex to the list.
        self.clicked_vertices.append(intersection.vert_id)

        if self.body_part == "hand":
            radius = 0.0015
        elif self.body_part == "body":
            radius = 0.005
        elif self.body_part == "body36":
            radius = 0.005
        elif self.body_part == "head":
            radius = 0.002

        ms = Spheres(positions, name="{}".format(intersection.vert_id), radius=radius)

        ms.current_frame_id = seq.current_frame_id
        self.scene.add(ms)

    def mouse_press_event(self, x: int, y: int, button: int):
        if not self.imgui_user_interacting and self.selected_mode == "inspect":
            result = self.mesh_mouse_intersection(x, y)
            if result is not None:
                self.interact_with_sequence(result, button)
        else:
            # Pass the event to the viewer if we didn't handle it.
            super().mouse_press_event(x, y, button)

    def interact_with_sequence(self, intersection, button):
        """
        Called when the user clicked on a mesh while holding ctrl.
        :param intersection: The result of intersecting the user click with the scene.
        :param button: The mouse button the user clicked.
        """
        if button == 1:  # left mouse
            self.add_virtual_marker(intersection)


if __name__ == "__main__":
    # To enable clicking, put the viewer into "inspection" mode by hitting
    # the `I` key. In this mode, a new window pops up that displays the face
    # and nearest vertex IDs for the current mouse position.
    #
    # To place spheres onto vertices, it might be easier to show the edges
    # by hitting the `E` key.

    file_to_edit = "body"

    v = ClickingViewer(file_to_edit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SSP_3D_Dataset(        
        img_dir=Path("data/SSP_3D/images"),
        labels_dir="data/SSP_3D",
        device=device
    )
    dataset = EHF_Dataset(
        data_dir=Path("data/EHF"),
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    ter = iter(dataloader)
    batch = next(ter)


    # 0: root 
    # 1:22 (21): body, 
    # 22:37 (15): left hand, 
    # 37:52 (15): right hand

    body_model, body_config = load_model("body")
    hand_model, hand_config = load_model("hand")
    images, roi, targets, uids = batch
    gt_pose = targets["pose"]
    ldmks, std, pred_pose, pred_shape = initial_pose_estimation(images, roi, body_model, hand_model, device)
    full_pose = torch.cat([gt_pose[:, 0].unsqueeze(1).to(device), pred_pose.to(device)], dim=1)
    #full_pose = gt_pose
    smpl_layer = SMPLLayer(model_type="smplh", gender="neutral", num_betas=10)
    print("!!!!!!!!!!!!!!!!!!", full_pose[:, 0])
    smpl_seq = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_root=full_pose[:, 0].reshape(1, -1),
        poses_body=full_pose[:, 1:22].reshape(1, -1),
        poses_left_hand=full_pose[:, 22:37].reshape(1, -1),
        poses_right_hand=full_pose[:, 37:].reshape(1, -1),
        betas=pred_shape,
    )

    # # Delete specific vertex indices from the file. Uncomment if needed.
    # vertex_indices = np.load(v.file_path)
    # vertex_indices = np.delete(
    #     vertex_indices,
    #     np.where(
    #         np.isin(vertex_indices, [0])
    #     )[0],
    # )
    # np.save(v.file_path, vertex_indices)

    # Display generated vertices for the SMPL-H model.
    '''vertex_positions = np.int64(
        targets['joints2D']
    )
    # Extract the positions of the specified vertices and display them.
    
    print(smpl_seq.position[np.newaxis].shape)
    vertex_positions = vertex_positions.reshape(-1, 3)
    vertices = Spheres(
        vertex_positions,
        name=f"Vertices",
        radius=0.004,
        color=(0.0, 0.0, 1.0, 1.0),
    )'''
    #v.scene.add(vertices)

    # Display in viewer.
    v.scene.add(smpl_seq)
    v.scene.floor.enabled = False
    v.scene.origin.enabled = False
    v.shadows_enabled = False

    v.run()
    
