import torch
# from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
# from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial
import math

import pickle
from tqdm import tqdm
import sys
import torch.nn as nn
import heapq


from typing import List, Tuple



# def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
#     """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
#     Args:
#         category (str): Category of product
#         masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
#         amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
#         num_th (int, optional): Number of thresholds
#     """
#
#     assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
#     assert isinstance(masks, ndarray), "type(masks) must be ndarray"
#     assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
#     assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
#     assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
#     assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
#     assert isinstance(num_th, int), "type(num_th) must be int"
#
#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
#     binary_amaps = np.zeros_like(amaps, dtype=bool)
#
#     min_th = amaps.min()
#     max_th = amaps.max()
#     delta = (max_th - min_th) / num_th
#
#     for th in np.arange(min_th, max_th, delta):
#         binary_amaps[amaps <= th] = 0
#         binary_amaps[amaps > th] = 1
#
#         pros = []
#         for binary_amap, mask in zip(binary_amaps, masks):
#             for region in measure.regionprops(measure.label(mask)):
#                 axes0_ids = region.coords[:, 0]
#                 axes1_ids = region.coords[:, 1]
#                 tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
#                 pros.append(tp_pixels / region.area)
#
#         inverse_masks = 1 - masks
#         fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
#         fpr = fp_pixels / inverse_masks.sum()
#
#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
#
#     # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
#     df = df[df["fpr"] < 0.3]
#     df["fpr"] = df["fpr"] / df["fpr"].max()
#
#     pro_auc = auc(df["fpr"], df["pro"])
#     return pro_auc


def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 50) -> float:
    """
    Optimized: pre-compute regionprops only once per image, reuse for each threshold.
    """
    # expand shape (H,W) -> (1,H,W)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if amaps.ndim == 2:
        amaps = amaps[None, ...]

    assert masks.shape == amaps.shape, "masks and amaps must have the same shape (N,H,W)."

    # binary masks
    masks = (masks > 0.5).astype(np.uint8)
    amaps = amaps.astype(np.float32)

    # pre-compute regionprops once
    regions_per_img: List[List[measure._regionprops.RegionProperties]] = []
    for mask in masks:
        labeled = measure.label(mask)
        regions_per_img.append(measure.regionprops(labeled))

    # thresholds (建议用分位数代替均匀采样，速度更快)
    thresholds = np.linspace(amaps.min(), amaps.max(), num=num_th, endpoint=False)

    inverse_masks = 1 - masks
    total_inverse = inverse_masks.sum() or 1.0  # 避免除零

    pro_list, fpr_list = [], []

    # loop over thresholds
    for th in thresholds:
        binary_amaps = amaps > th  # (N,H,W) bool

        # per-region PRO
        pros = []
        for binary_amap, regions in zip(binary_amaps, regions_per_img):
            for region in regions:
                coords = region.coords
                tp = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pros.append(tp / region.area)
        pro_mean = np.mean(pros) if len(pros) else 0.0

        # false positive rate
        fp = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp / total_inverse

        pro_list.append(pro_mean)
        fpr_list.append(fpr)

    # keep fpr < 0.3
    df = pd.DataFrame({"pro": pro_list, "fpr": fpr_list})
    df = df[df["fpr"] < 0.3]
    if df.empty:
        return 0.0

    max_fpr = df["fpr"].max()
    if max_fpr <= 0:
        return 0.0

    df["fpr"] /= max_fpr
    return float(auc(df["fpr"].values, df["pro"].values))



def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()