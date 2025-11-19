# dataset/mvtec2.py
import os
import glob
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class MVTec2Dataset(Dataset):
    def __init__(self, root, transform, target_transform, mode="test", test_type="test_public", cls_only=None):
        """
        root: /path/to/mvtec_ad_2
        mode: "train" / "validation" / "test"
        test_type: "test_public" / "test_private" / "test_private_mixed"
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.test_type = test_type
        self.cls_only = cls_only

        self.img_paths, self.gt_paths, self.labels, self.types, self.cls_names = self.load_dataset()

    def load_dataset(self):
        img_paths, gt_paths, labels, types, cls_names = [], [], [], [], []

        for cls_name in sorted(os.listdir(self.root)):
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            if self.cls_only and cls_name != self.cls_only:
                continue

            if self.mode in ["train", "validation"]:
                good_dir = os.path.join(cls_dir, self.mode, "good")
                files = sorted(glob.glob(os.path.join(good_dir, "*.png")) +
                               glob.glob(os.path.join(good_dir, "*.jpg")))
                img_paths.extend(files)
                gt_paths.extend([0] * len(files))
                labels.extend([0] * len(files))
                types.extend(["good"] * len(files))
                cls_names.extend([cls_name] * len(files))

            elif self.mode == "test":
                test_dir = os.path.join(cls_dir, self.test_type)

                if self.test_type == "test_public":

                    gt_dir = os.path.join(test_dir, "ground_truth")
                    for defect_type in os.listdir(test_dir):
                        if defect_type == "ground_truth":
                            continue
                        defect_dir = os.path.join(test_dir, defect_type)
                        files = sorted(glob.glob(os.path.join(defect_dir, "*.png")) +
                                       glob.glob(os.path.join(defect_dir, "*.jpg")))
                        if defect_type == "good":
                            img_paths.extend(files)
                            gt_paths.extend([0] * len(files))
                            labels.extend([0] * len(files))
                            types.extend(["good"] * len(files))
                            cls_names.extend([cls_name] * len(files))
                        else:
                            gts = sorted(glob.glob(os.path.join(gt_dir, defect_type, "*.png"))) if gt_dir else [0] * len(files)
                            img_paths.extend(files)
                            gt_paths.extend(gts)
                            labels.extend([1] * len(files))
                            types.extend([defect_type] * len(files))
                            cls_names.extend([cls_name] * len(files))

                elif self.test_type == "test_private":

                    files = sorted(glob.glob(os.path.join(test_dir, "*.png")) +
                                   glob.glob(os.path.join(test_dir, "*.jpg")))
                    img_paths.extend(files)
                    gt_paths.extend([0] * len(files))  # 没有 mask
                    labels.extend([-1] * len(files))  # -1 表示未知标签
                    types.extend(["unknown"] * len(files))
                    cls_names.extend([cls_name] * len(files))

                elif self.test_type == "test_private_mixed":

                    files = sorted(glob.glob(os.path.join(test_dir, "*.png")) +
                                   glob.glob(os.path.join(test_dir, "*.jpg")))
                    img_paths.extend(files)
                    gt_paths.extend([0] * len(files))  # 没有 mask
                    labels.extend([-1] * len(files))  # -1 表示未知标签
                    types.extend(["unknown"] * len(files))
                    cls_names.extend([cls_name] * len(files))

        return img_paths, gt_paths, labels, types, cls_names

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img_t = self.transform(img)

        label = self.labels[idx]
        if label <= 0 or self.gt_paths[idx] == 0:  # good 或 unknown
            gt = torch.zeros([1, img_t.shape[1], img_t.shape[2]])
        else:
            gt_img = Image.open(self.gt_paths[idx]).convert("L")
            gt = self.target_transform(gt_img)
            if gt.shape[0] == 3:
                gt = gt[:1]

        img = Image.open(self.img_paths[idx]).convert("RGB")
        img_t = self.transform(img)

        # 如果你希望 img_pil 就是和 img 一样的 tensor：
        img_pil = transforms.ToTensor()(img)

        return {
            "img": img_t,
            "img_pil": img_pil,  # ✅ 这里变成 tensor
            "img_path": self.img_paths[idx],
            "cls_name": self.cls_names[idx],
            "img_mask": gt,
            "anomaly": label,
        }

    def get_cls_names(self):
        return sorted(set(self.cls_names))
