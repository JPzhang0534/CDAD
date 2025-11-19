
import random
from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import cv2


class MVTec2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, cls_name=None, transform=None, target_transform=None,
                 mode="test", split="public"):
        """
        mode: "train" | "validation" | "test"
        split: only used if mode == "test"
               "public" | "private" | "private_mixed"
        """
        self.root = root
        self.cls_name = cls_name
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.split = split

        base_dir = os.path.join(root, cls_name)
        if mode == "train":
            self.img_dir = os.path.join(base_dir, "train", "good")
        elif mode == "validation":
            self.img_dir = os.path.join(base_dir, "validation", "good")
        elif mode == "test":
            if split == "public":
                self.img_dir = os.path.join(base_dir, "test_public")
            elif split == "private":
                self.img_dir = os.path.join(base_dir, "test_private")
            elif split == "private_mixed":
                self.img_dir = os.path.join(base_dir, "test_private_mixed")
            else:
                raise ValueError(f"Unknown split {split}")
        else:
            raise ValueError(f"Unknown mode {mode}")

        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.png")) +
                                glob.glob(os.path.join(self.img_dir, "*.jpg")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # TODO: 根据 test_public 的 mask 目录，加载对应的 gt mask
        # mask_path = img_path.replace("test_public", "ground_truth").replace(".png", "_mask.png")
        mask = None

        return {
            "img": img,
            "img_path": img_path,
            "cls_name": self.cls_name,
            "img_mask": mask,
            "anomaly": 1 if "defect" in img_path else 0,
        }
