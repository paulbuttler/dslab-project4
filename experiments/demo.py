import torch
from src.models.load import get_val_dataloader, load_model
from src.inference.optimization import full_model_inference
from src.utils.visualization import visualize_prediction, visualize_optimization

if __name__ == "__main__":

    save = False  # Set to True to save 2D visualizations

    # Load the trained body and hand models
    body_model, config = load_model("body")
    hand_model, _ = load_model("hand")

    # Get a batch of images, rois, and camera intrinsics from synth validation set or EHF dataset
    val_loader = get_val_dataloader(config, "data/synth_body", "data/annot/body_meta.pkl", 3, "test")
    images, targets, uids = next(iter(val_loader))
    images = images.to(config.device)

    # Our method requires a tight square ROI around the body
    # For an example to obtain tight crops using YOLOv8, see the EHF dataset class in `src/datasets/EHF_dataset.py`
    roi = targets["roi"].to(config.device)

    # We also require camera intrinsics
    cam_int = targets["cam_int"].to(config.device)

    # Perform inference using the full model architecture
    pose, shape, R, t = full_model_inference(images, roi, cam_int, body_model, hand_model, config.device)
    cam_ext = torch.cat([R, t.reshape(-1, 3, 1)], axis=2)  # (B, 3, 4)

    # Display the results in 3D using the aitviewer
    visualize_prediction(images, pose, shape, cam_int.cpu(), cam_ext)

    # Save 2D results in experiments directory
    for idx in range(len(uids)):
        visualize_optimization(
            images[idx],
            shape[idx].numpy(),
            pose[idx].numpy(),
            R[idx].numpy(),
            t[idx].numpy(),
            cam_int[idx],
            uids[idx],
            f"experiments/synth/",
        )