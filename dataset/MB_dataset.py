import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import os
import glob

class MBDataset(data.Dataset):
    """
    适配 MPDD / BTAD 数据集 (仿 VisaDataset 格式输出)
    root: 数据集路径 (MPDD/ 或 BTAD/01/)
    mode: 'train' or 'test'
    """

    def __init__(self, root, transform=None, target_transform=None, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        self.data_all = []
        self.cls_names = []

        # MPDD: root/train/cls_name/*.png
        # BTAD: root/train/*.png  (类别 = 01, 02, 03...)
        if os.path.basename(root).isdigit():
            # BTAD 单个类别 (01, 02, 03)
            self.cls_names = [os.path.basename(root)]
        else:
            # MPDD 多类别 (tile, scratch, bubble...)
            self.cls_names = sorted(os.listdir(os.path.join(root, mode)))

        for cls_name in self.cls_names:
            img_dir = os.path.join(root, mode, cls_name) if os.path.basename(root).isdigit() is False else os.path.join(root, mode)
            img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))

            for img_path in img_files:
                if mode == 'train':
                    # 训练集默认无缺陷
                    anomaly = 0
                    mask_path = None
                    specie_name = 'good'
                else:
                    # 测试集，区分 good / defect
                    if 'good' in img_path:
                        anomaly = 0
                        mask_path = None
                        specie_name = 'good'
                    else:
                        anomaly = 1
                        fname = os.path.splitext(os.path.basename(img_path))[0]
                        mask_path = os.path.join(root, "ground_truth", cls_name, fname + "_mask.png")
                        if not os.path.exists(mask_path):
                            mask_path = None
                        specie_name = 'defect'

                self.data_all.append({
                    'img_path': os.path.relpath(img_path, root),
                    'mask_path': os.path.relpath(mask_path, root) if mask_path else None,
                    'cls_name': cls_name,
                    'specie_name': specie_name,
                    'anomaly': anomaly
                })

        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def get_cls_names(self):
        return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = \
            data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']

        img_pil = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        if anomaly == 0 or mask_path is None:
            img_mask = Image.fromarray(np.zeros((img_pil.size[1], img_pil.size[0])), mode='L')
        else:
            img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        img = self.transform(img_pil) if self.transform is not None else img_pil
        img_mask = self.target_transform(img_mask) if self.target_transform is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        return {
            'img_pil': np.array(img_pil.resize((512, 512))),
            'img': img,
            'img_mask': img_mask,
            'cls_name': cls_name.replace("_", " "),
            'anomaly': anomaly,
            'anomaly_class': specie_name.replace("_", " "),
            'img_path': os.path.join(self.root, img_path)
        }