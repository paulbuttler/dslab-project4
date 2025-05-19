import torch
import torch.nn as nn
import kornia.filters as KF
import kornia.augmentation as K
from kornia.geometry.transform import (
    get_rotation_matrix2d,
    warp_affine,
)


def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize an image tensor with ImageNet mean and std."""
    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    return (img - mean) / std


def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize an image tensor with ImageNet mean and std."""
    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    return img * std + mean


def sample_triangular(low, high, mode, size, device=None):
    """Sample from a triangular distribution using inverse transform sampling."""
    u = torch.rand(size, device=device)
    c = (mode - low) / (high - low)
    diff = high - low

    return torch.where(
        u < c,
        low + torch.sqrt(u * diff * (mode - low)),
        high - torch.sqrt((1 - u) * diff * (high - mode)),
    )


def random_roi_transform(
    img: torch.Tensor,
    kp2d: torch.Tensor,
    roi: torch.Tensor,
    roi_aug: dict,
    crop_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly apply a rotation, scaling, and translation to the image and keypoints,
    simulating imperfections from sliding window ROI extraction.
    """
    device = img.device
    B = img.shape[0]

    angle = roi_aug["angle"]
    scale = roi_aug["scale"]
    trans = roi_aug["trans"]

    # Use a triangular distribution for simple sampling around the ideal ROI
    angle = sample_triangular(-angle, angle, 0.0, (B,), device=device)

    scale_offset = sample_triangular(scale[0], scale[1], 0.0, (B,), device=device)
    scale = (1.0 + scale_offset).unsqueeze(1).expand(-1, 2)

    roi_w = roi[:, 2] - roi[:, 0]
    tx = sample_triangular(-trans, trans, 0.0, (B,), device=device) * roi_w
    ty = sample_triangular(-trans, trans, 0.0, (B,), device=device) * roi_w
    translation = torch.stack([tx, ty], dim=1)

    return apply_roi_transform(
        img, kp2d, roi, "train", crop_size, angle, scale, translation
    )


def apply_roi_transform(
    img: torch.Tensor,
    kp2d: torch.Tensor,
    roi: torch.Tensor,
    mode: str = "train",
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

    corners = torch.stack(
        [roi[:, :2], torch.stack([roi[:, 2], roi[:, 1]], dim=-1), roi[:, 2:]], dim=1
    ).to(
        device
    )  # [B, 3, 2]
    center = (corners[:, 0] + corners[:, 2]) / 2.0

    if mode == "train":

        # Apply rotation + scaling + translation to the bounding box corners
        M1 = get_rotation_matrix2d(center=center, angle=angle, scale=scale).to(device)
        M1[:, :, 2] += translation.to(device)

        corners_h = torch.cat([corners, torch.ones_like(corners[..., :1])], dim=2)
        src = torch.bmm(M1, corners_h.transpose(1, 2)).transpose(1, 2)

    else:
        src = corners

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

    if mode == "test":
        return img_warped, M2

    kp2d_h = torch.cat([kp2d, torch.ones_like(kp2d[..., :1])], dim=2)
    kp2d_warped = torch.bmm(M2, kp2d_h.transpose(1, 2)).transpose(1, 2)  # [B, N, 2]

    # assert torch.allclose(warp_back(kp2d_warped, M2), kp2d.cpu(), atol=1e-4)
    return img_warped, kp2d_warped


def warp_back(kp2d, M):
    """
    Given a set of 2D keypoints and an affine transformation matrix,
    compute the original 2D keypoints before the transformation.
    """
    B, N, _ = kp2d.shape
    ldmks_crop_h = torch.cat([kp2d, torch.ones_like(kp2d[..., :1])], dim=-1)  # [B, N, 3]
    eye_row = torch.tensor([0.0, 0.0, 1.0], device=M.device).view(1, 1, 3).expand(B, 1, 3)
    M_h = torch.cat([M, eye_row], dim=1)  # [B, 3, 3]

    # We want to solve for Y: (M_h @ Y^T = X^T), so Y^T = torch.linalg.solve(M_h, X^T)
    ldmks_orig_h_t = torch.linalg.solve(M_h, ldmks_crop_h.transpose(1, 2))  # [B, 3, N]
    ldmks_orig_h = ldmks_orig_h_t.transpose(1, 2)  # [B, N, 3]
    return ldmks_orig_h[..., :2].cpu()


class AppearanceAugmentation(nn.Module):

    def __init__(self, appearance_aug: dict):
        super().__init__()

        self.probs = appearance_aug["probs"]

        # Built-in Kornia augmentations
        self.colorjitter = K.ColorJitter(
            hue=0.05, saturation=0.15, p=self.probs["hue_saturation"]
        )
        self.grayscale = K.RandomGrayscale(p=self.probs["grayscale"])
        self.jpeg = K.RandomJPEG(jpeg_quality=(50, 95), p=self.probs["jpeg"])

        self.random_erasing = K.RandomErasing(
            p=self.probs["cutout"],
            scale=(0.02, 0.12),
            ratio=(0.5, 2.0),
        )

    def forward(self, img: torch.Tensor):

        device = img.device

        # Motion blur with kernel size propotional to image size
        if torch.rand(1) < self.probs["motion_blur"]:
            kernel_size = max(3, min(12, int(img.shape[-1] * 0.02) | 1))  # Force odd
            angle = torch.rand(1, device=device) * 360
            direction = torch.rand(1, device=device) * 2 - 1
            img = KF.motion_blur(
                img, kernel_size=kernel_size, angle=angle, direction=direction
            )

        # Brightness (constant shift)
        if torch.rand(1) < self.probs["brightness"]:
            offset = torch.rand(1, device=device) * 0.3 - 0.15
            img = (img + offset).clamp(0, 1)

        # Contrast adjustment
        if torch.rand(1) < self.probs["contrast"]:
            contrast = torch.rand(1, device=device) * 0.5 - 0.25
            img = ((img - 0.5) * (1 + contrast) + 0.5).clamp(0, 1)

        # Random erasing (as a replacement for overlaying occluders) for hand images
        img = self.random_erasing(img)

        # Hue and saturation, grayscale and JPEG compression
        img = self.colorjitter(img)
        img = self.grayscale(img)
        # Fix error in kornia package
        img = self.jpeg(img.to("cpu")).to(device)

        # ISO noise
        if torch.rand(1) < self.probs["iso_noise"]:
            img = (
                torch.poisson(img * 512) / 512 + torch.randn_like(img) * 0.002
            ).clamp(0, 1)

        return img
