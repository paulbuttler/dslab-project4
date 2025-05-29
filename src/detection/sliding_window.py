import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_windows(img, scale_factors=[0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4], stride_ratio=0.1):
    h, w, _ = img.shape
    base_size = min(h, w)
    windows = []

    for scale in scale_factors:
        win_size = int(base_size * scale)
        stride = int(win_size * stride_ratio)

        for y in range(0, h - win_size + 1, stride):
            for x in range(0, w - win_size + 1, stride):
                y1, y2 = y, y + win_size
                x1, x2 = x, x + win_size
                crop = img[y1:y2, x1:x2]
                windows.append(((y1, y2, x1, x2), crop, scale))

    return windows


def extract_confidence(log_vars):
    sigma = torch.exp(log_vars / 2)
    return (1 / sigma).mean().item()


def visualize_all_windows(windows, landmarks_list, confidence_list):
    cols = 4
    rows = (len(windows) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()

    for i, ((_, crop, scale), ax) in enumerate(zip(windows, axs)):
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Conf: {confidence_list[i]:.2f}\nScale: {scale:.2f}")

        if landmarks_list[i] is not None:
            lm = landmarks_list[i]
            ax.scatter(lm[:, 0], lm[:, 1], c='lime', s=10)

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_best_window(crop, landmarks, confidence, scale):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if landmarks is not None:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c='lime', s=10)
    else:
        plt.text(10, 20, "No pose detected", color='red')
    plt.title(f"Best Window\nConf: {confidence:.2f} | Scale: {scale:.2f}")
    plt.axis('off')
    plt.show()


def get_best_body_roi(raw_img, sparse_model, dense_model, config, input_size=224, visualize=False):
    windows = generate_windows(raw_img)

    best_conf = -float("inf")
    best_crop = None
    best_box = None
    best_scale = None
    landmarks_list = []
    confidence_list = []

    sparse_model.eval()
    dense_model.eval()

    for (y1, y2, x1, x2), crop, scale in windows:
        resized_crop = cv2.resize(crop, (input_size, input_size))
        input_tensor = torch.from_numpy(resized_crop.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(config.device)

        with torch.no_grad():
            sparse_out = sparse_model(input_tensor)
            landmarks_sparse = sparse_out["landmarks"][0]  # shape: [N, 3]
            log_vars_sparse = landmarks_sparse[:, 2]
            confidence = extract_confidence(log_vars_sparse)

        # Scale landmarks to original crop size
        lm = landmarks_sparse[:, :2].cpu().numpy()
        lm_scaled = lm * np.array([crop.shape[1], crop.shape[0]])
        landmarks_list.append(lm_scaled)
        confidence_list.append(confidence)

        if confidence > best_conf:
            best_conf = confidence
            best_crop = crop
            best_box = (y1, y2, x1, x2)
            best_scale = scale

    # Apply dense model on best crop
    resized_best_crop = cv2.resize(best_crop, (input_size, input_size))
    input_tensor = torch.from_numpy(resized_best_crop.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(config.device)

    with torch.no_grad():
        dense_out = dense_model(input_tensor)
        landmarks_dense = dense_out["landmarks"][0, :, :2].cpu().numpy()
        landmarks_dense = landmarks_dense * np.array([best_crop.shape[1], best_crop.shape[0]])

    if visualize:
        visualize_all_windows(windows, landmarks_list, confidence_list)
        visualize_best_window(best_crop, landmarks_dense, best_conf, best_scale)

    return best_crop, best_box
