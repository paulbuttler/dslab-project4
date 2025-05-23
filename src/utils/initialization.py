import torch
import cv2
from utils.rottrans import aa2rot, rot2aa, rotation_6d_to_matrix
from utils.roi import compute_roi
from models.load import load_model, get_val_dataloader
from datasets.transforms import apply_roi_transform, normalize, denormalize, warp_back
from torchvision.utils import make_grid

def initial_pose_estimation(images, roi, body_model, hand_model, device):
    """
    Perform initial pose, shape and refined landmark estimation using trained body and hand models on full images given a bounding box ROI.
    Args:
        images (Tensor): Batch of unnormalized, uncropped input images, shape [B, 3, H, W].
        roi (Tensor): Region of interest that tightly contains the whole body, shape [B, 4].
        body_model (nn.Module): Trained body model.
        hand_model (nn.Module): Trained hand model.
        device (str): Device to run the models on.
    Returns:
        ldmks (Tensor): Refined landmarks, shape [B, N, 2].
        std (Tensor): Standard deviations of landmarks, shape [B, N].
        pose (Tensor): Pose parameters, excluding the global orientation, shape [B, 51, 3].
        shape (Tensor): Shape parameters, shape [B, 10].
    """

    # Crop and normalize images for the body model
    cropped_images, M = apply_roi_transform(images, None, roi, "test", 256.0)
    norm_images = normalize(cropped_images)

    body_model.eval()
    with torch.no_grad():
        outputs = body_model(norm_images)
        ldmks = outputs["landmarks"][..., :2] * 256.0
        std = (torch.exp(0.5 * outputs["landmarks"][..., 2])).cpu()
        body_pose = rotation_6d_to_matrix(outputs["pose"]).cpu()
        shape = outputs["shape"].cpu()

    # Warp back the landmarks to the original image
    ldmks = warp_back(ldmks, M)

    # Compute the hand ROIs in original image space based on predicted hand landmarks
    left_hand_roi = compute_roi(ldmks[:, 665:802], None, 0.25, images.shape[-2]).to(device, dtype=torch.float32)
    right_hand_roi = compute_roi(ldmks[:, 802:939], None, 0.25, images.shape[-2]).to(device, dtype=torch.float32)

    # Crop and normalize images for the hand model
    left_hand_img, M1 = apply_roi_transform(images, None, left_hand_roi, "test", 128.0)
    right_hand_img, M2 = apply_roi_transform(images, None, right_hand_roi, "test", 128.0)
    right_hand_img_flipped = torch.flip(right_hand_img, [-1])

    # visualize_batch_of_images(cropped_images)
    # visualize_batch_of_images(left_hand_img)
    # visualize_batch_of_images(right_hand_img_flipped)

    left_hand_img = normalize(left_hand_img)
    right_hand_flipped = normalize(right_hand_img_flipped)

    hand_model.eval()
    with torch.no_grad():
        left_out = hand_model(left_hand_img)
        right_out = hand_model(right_hand_flipped)

        left_hand_ldmks_crop = left_out["landmarks"][..., :2] * 128.0
        left_hand_std = (torch.exp(0.5 * left_out["landmarks"][..., 2])).cpu()  # [B, 137]
        left_hand_pose = rotation_6d_to_matrix(left_out["pose"]).cpu()

        right_hand_ldmks_crop = right_out["landmarks"][..., :2] * 128.0
        right_hand_std = (torch.exp(0.5 * right_out["landmarks"][..., 2])).cpu()
        right_hand_pose_flipped = rotation_6d_to_matrix(right_out["pose"]).cpu()

        # Flip predicted landmarks
        right_hand_ldmks_crop[..., 0] = 128.0 - right_hand_ldmks_crop[..., 0]

    # Warp back the hand landmarks to the original image
    left_hand_ldmks_orig = warp_back(left_hand_ldmks_crop, M1)
    right_hand_ldmks_orig = warp_back(right_hand_ldmks_crop, M2)

    # Replace the original hand landmarks with the refined ones
    ldmks[:, 665:802] = left_hand_ldmks_orig
    ldmks[:, 802:939] = right_hand_ldmks_orig

    # Compute statistics of the landmark std deviations
    body_std = torch.mean(std[:, :665], dim=1)
    left_hand_std = torch.mean(left_hand_std, dim=1)
    right_hand_std = torch.mean(right_hand_std, dim=1)
    print(f"Body std mean, min, max: {body_std.mean(), body_std.min(), body_std.max()}")
    print(f"Left hand std mean, min, max: {left_hand_std.mean(), left_hand_std.min(), left_hand_std.max()}")
    print(f"Right hand std mean, min, max: {right_hand_std.mean(), right_hand_std.min(), right_hand_std.max()}")

    # Concatenate pose parameters and flip the right hand pose
    pose = torch.cat([body_pose, left_hand_pose, right_hand_pose_flipped], dim=1)
    pose = rot2aa(pose)
    pose[:, -15:] *= torch.tensor([1.0, -1.0, -1.0], device=pose.device)
    return ldmks, std, pose, shape

def visualize_batch_of_images(images):
    grid = make_grid(images, nrow=5, padding=2)  # [C, H', W']
    grid_np = grid.permute(1, 2, 0).cpu().numpy()  # [H', W', C]
    grid_bgr = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("Batch Grid", grid_bgr)
    cv2.waitKey(0)

# Example usage
if __name__ == "__main__":

    data_root = f"data/raw/synth_body"
    meta_file = f"data/annotations/body_meta.pkl"

    body_model, config = load_model("body")
    hand_model, _ = load_model("hand")

    val_loader = get_val_dataloader(config, data_root, meta_file, 5, "test")

    images, targets, uids = next(iter(val_loader))
    roi = targets["roi"].to(config.device)

    images = images.to(config.device)
    images = denormalize(images)

    # Perform pose, shape and refined landmark estimation
    ldmks, std, pose, shape = initial_pose_estimation(images, roi, body_model, hand_model, config.device)

    print(f"Landmarks shape: {ldmks.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Pose shape: {pose.shape}")
    print(f"Shape shape: {shape.shape}")
