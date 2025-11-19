import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def save_attention_map(attn_map: torch.Tensor, save_path: str, cmap="viridis"):
    """
    保存单张 attention map 为热力图
    attn_map: (H, W) or (1,H,W)
    save_path: 保存路径
    """
    attn_map = attn_map.detach().cpu()
    if attn_map.ndim == 3:
        attn_map = attn_map.squeeze(0)

    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    plt.figure(figsize=(5, 5))
    plt.imshow(attn_map.numpy(), cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def overlay_attention_on_image(attn_map: torch.Tensor, image: np.ndarray, save_path: str, cmap="jet", alpha=0.5):
    """
    将注意力图叠加到原始图像上
    attn_map: (H,W) 张量
    image: (H,W,3) numpy 数组 (RGB)
    save_path: 保存路径
    """
    attn_map = attn_map.detach().cpu()
    if attn_map.ndim == 3:
        attn_map = attn_map.squeeze(0)

    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    plt.figure(figsize=(5, 5))
    plt.imshow(image)  # 原图
    plt.imshow(attn_map.numpy(), cmap=cmap, alpha=alpha)  # 叠加注意力
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_batch_attention_maps(attn_batch: torch.Tensor, save_dir: str, prefix="attn",
                              images: list = None, cmap="viridis", overlay=False, alpha=0.5):
    """
    保存一个 batch 内的所有 attention maps
    attn_batch: (B,1,H,W)
    images: list of numpy arrays (B,H,W,3)，可选，如果提供则支持叠加
    overlay: 是否叠加在原图上
    """
    os.makedirs(save_dir, exist_ok=True)
    B = attn_batch.shape[0]
    for i in range(B):
        attn_map = attn_batch[i, 0]  # (H,W)
        if overlay and images is not None:
            save_path = os.path.join(save_dir, f"{prefix}_overlay_{i}.png")
            overlay_attention_on_image(attn_map, images[i], save_path, cmap=cmap, alpha=alpha)
        else:
            save_path = os.path.join(save_dir, f"{prefix}_{i}.png")
            save_attention_map(attn_map, save_path, cmap=cmap)


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2  # 用于高斯模糊


def save_attention_map_gradcam(attn_map: torch.Tensor, save_path: str,
                               image: np.ndarray = None, alpha=0.5, colormap=cv2.COLORMAP_JET,
                               blur=True, blur_ksize=7):
    """
    Grad-CAM 风格的 attention 可视化
    attn_map: (H,W) 或 (1,H,W) 张量
    image: 可选原图，numpy array (H,W,3)，RGB
    alpha: 叠加透明度
    colormap: cv2 颜色映射
    blur: 是否对注意力图进行高斯模糊
    blur_ksize: 高斯核大小
    """
    attn_map = attn_map.detach().cpu().numpy()
    if attn_map.ndim == 3:
        attn_map = attn_map.squeeze(0)

    # 归一化到0~1
    # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # 高斯模糊，平滑热力图
    if blur:
        attn_map = cv2.GaussianBlur(attn_map, (blur_ksize, blur_ksize), 0)

    # 转成0~255并应用 colormap
    heatmap = np.uint8(255 * attn_map)
    heatmap = cv2.applyColorMap(heatmap, colormap)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转回 RGB

    # 叠加原图
    if image is not None:
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        heatmap = heatmap.astype(np.float32) / 255.0
        cam = (1 - alpha) * image + alpha * heatmap
    else:
        cam = heatmap.astype(np.float32) / 255.0

    cam = np.clip(cam, 0, 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(cam)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_batch_attention_gradcam(attn_batch: torch.Tensor, save_dir: str,
                                 prefix="attn", images=None, alpha=0.5):
    """
    保存 batch Attention Map 为 Grad-CAM 风格
    attn_batch: (B,1,H,W)
    images: list of numpy arrays (B,H,W,3)
    """
    os.makedirs(save_dir, exist_ok=True)
    B = attn_batch.shape[0]
    for i in range(B):
        attn_map = attn_batch[i, 0]
        img = images[i] if images is not None else None
        save_path = os.path.join(save_dir, f"{prefix}_{i}.png")
        save_attention_map_gradcam(attn_map, save_path, image=img, alpha=alpha)

