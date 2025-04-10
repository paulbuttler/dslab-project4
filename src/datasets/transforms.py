import random
import torch
import gzip
import pickle
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
from io import BytesIO
import kornia.augmentation as K
from torchvision.io import decode_image
from kornia.geometry.transform import (
    get_rotation_matrix2d,
    warp_affine,
)


def random_roi_transform(
    img: torch.Tensor, kp2d: torch.Tensor, roi: torch.Tensor, crop_size: float = 256.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly apply a rotation, scaling, and translation to the image and keypoints.
    """
    device = img.device

    B = img.shape[0]

    # random perturbation of roi
    angle = torch.rand(B, device=device) * 50.0 - 25.0
    scale = (torch.rand(B, device=device) * 0.15 + 1.05).unsqueeze(1).expand(-1, 2)
    roi_w = roi[:, 2] - roi[:, 0]
    tx = (torch.rand(B, device=device) * 2 - 1) * 0.05 * roi_w
    ty = (torch.rand(B, device=device) * 2 - 1) * 0.05 * roi_w
    translation = torch.stack([tx, ty], dim=1)

    return apply_roi_transform(img, kp2d, roi, crop_size, angle, scale, translation)


def apply_roi_transform(
    img: torch.Tensor,
    kp2d: torch.Tensor,
    roi: torch.Tensor,
    crop_size: float = 256.0,
    angle: torch.Tensor = None,
    scale: torch.Tensor = None,
    translation: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotation, scaling, and translation around the center of a bounding box,
    then crop and warp the image and 2D keypoints.
    """
    device = img.device
    B = img.shape[0]

    if angle is None and scale is None and translation is None:
        angle = torch.zeros(B, dtype=torch.float32, device=device)
        scale = torch.full((B, 2), 1.1, dtype=torch.float32, device=device)
        translation = torch.zeros(B, 2, dtype=torch.float32, device=device)

    corners = torch.stack(
        [roi[:, :2], torch.stack([roi[:, 2], roi[:, 1]], dim=-1), roi[:, 2:]], dim=1
    ).to(
        device
    )  # [B, 3, 2]
    center = (corners[:, 0] + corners[:, 2]) / 2.0

    # Apply rotation + scaling + translation to the bounding box corners
    M1 = get_rotation_matrix2d(center=center, angle=angle, scale=scale).to(device)
    M1[:, :, 2] += translation.to(device)

    corners_h = torch.cat([corners, torch.ones_like(corners[..., :1])], dim=2)
    src = torch.bmm(M1, corners_h.transpose(1, 2)).transpose(1, 2)
    src_h = torch.cat([src, torch.ones_like(src[..., :1])], dim=2)

    # Compute affine matrix to map transformed box to canonical crop
    target = (
        torch.tensor(
            [[0.0, 0.0], [crop_size, 0.0], [crop_size, crop_size]], device=device
        )
        .unsqueeze(0)
        .expand(img.shape[0], -1, -1)
    )

    M2 = torch.linalg.lstsq(src_h, target).solution.permute(0, 2, 1)  # [B, 2, 3]

    # Warp image and keypoints
    img_warped = warp_affine(img, M2, dsize=(int(crop_size), int(crop_size)))
    kp2d_h = torch.cat([kp2d, torch.ones_like(kp2d[..., :1])], dim=2)
    kp2d_warped = torch.bmm(M2, kp2d_h.transpose(1, 2)).transpose(1, 2)  # [B, N, 2]

    return img_warped, kp2d_warped


class AppearanceAugmentation:
    def __init__(self, is_hand=False, occluder_library=None):
        self.is_hand = is_hand
        self.occluder_library = occluder_library if occluder_library is not None else []

    def __call__(self, img, landmarks):
        # Loc/Rot/Scale augmentation
        if random.random() < 0.5:
            img, landmarks = self.apply_random_transformation(img, landmarks)

        # Random motion blur
        if random.random() < 0.5:
            kernel_size = int(min(img.size) * random.uniform(0.01, 0.05))
            if kernel_size % 2 == 0:
                kernel_size += 1
            angle = random.uniform(0, 360)
            img = K.RandomMotionBlur(kernel_size=kernel_size, angle=angle, direction=random.uniform(-1, 1))(np.array(img))
            img = Image.fromarray(img.astype(np.uint8))

        # Brightness
        if random.random() < 0.5:
            brightness_offset = random.uniform(-0.2, 0.2)
            img = np.clip(np.array(img) + brightness_offset * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)

        # Adjust contrast
        if random.random() < 0.5:
            contrast = random.uniform(-0.5, 0.5)
            img = np.clip((np.array(img) / 255.0 - 0.5) * (1 + contrast) + 0.5, 0, 1) * 255
            img = Image.fromarray(img.astype(np.uint8))

        # Adjust hue and saturation
        if random.random() < 0.5:
            img = K.ColorJitter(hue=random.uniform(-0.1, 0.1), saturation=random.uniform(0.8, 1.2))(np.array(img))
            img = Image.fromarray(img.astype(np.uint8))

        # Apply JPEG compression
        if random.random() < 0.5:
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=random.randint(30, 90))
            buffer.seek(0)
            img = Image.open(buffer)

        # Convert to grayscale
        if random.random() < 0.5:
            img = img.convert("L").convert("RGB")

        # Add random occlusions (hand only)
        if self.is_hand and random.random() < 0.5 and self.occluder_library:
            occluder = random.choice(self.occluder_library)
            occluder = occluder.resize((random.randint(20, 100), random.randint(20, 100)))
            occluder = occluder.rotate(random.uniform(0, 360), expand=True)
            x_offset = random.randint(0, img.size[0] - occluder.size[0])
            y_offset = random.randint(0, img.size[1] - occluder.size[1])
            img.paste(occluder, (x_offset, y_offset), occluder)

        # Add random ISO noise
        if random.random() < 0.5:
            img_array = np.array(img) / 255.0
            poisson_noise = np.random.poisson(img_array * 255) / 255.0
            gaussian_noise = np.random.normal(0, 0.02, img_array.shape)
            img_array = np.clip(img_array + poisson_noise + gaussian_noise, 0, 1) * 255
            img = Image.fromarray(img_array.astype(np.uint8))

        return img, landmarks

    def apply_random_transformation(self, img, landmarks):
        # Image
        scale = random.uniform(0.8, 1.2)

        angle = random.uniform(-30, 30)

        tx = random.uniform(-20, 20)
        ty = random.uniform(-20, 20)

        width, height = img.size
        img = img.resize((int(width * scale), int(height * scale)), Image.BICUBIC)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

        img = img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, tx, 0, 1, ty),
            resample=Image.BICUBIC,
        )

        # Landmarks
        landmarks = np.array(landmarks)
        center_x, center_y = width / 2, height / 2

        landmarks = (landmarks - [center_x, center_y]) * scale + [center_x, center_y]

        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        landmarks = np.dot(landmarks - [center_x, center_y], rotation_matrix.T) + [center_x, center_y]

        # Translate landmarks
        landmarks += [tx, ty]

        return img, landmarks.tolist()
