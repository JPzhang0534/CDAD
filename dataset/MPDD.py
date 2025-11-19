import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import os


class MPDDDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, mode='test', k_shot=0, save_dir=None, obj_name=None):
        """
        root: MPDD 数据集根目录 (包含多个 class 文件夹)
        mode: 'train' or 'test'
        obj_name: 指定 class 名称 (对应 MPDD/class1, class2 ...)
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        self.data_all = []

        # 如果训练，只选择一个 class
        if mode == 'train':
            assert obj_name is not None, "训练时必须指定 obj_name"
            self.cls_names = [obj_name]
            save_path = os.path.join(save_dir, 'k_shot.txt') if save_dir else None
        else:
            self.cls_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            save_path = None

        for cls_name in self.cls_names:
            cls_dir = os.path.join(root, cls_name)
            phase_dir = os.path.join(cls_dir, mode)

            # good 样本
            good_dir = os.path.join(phase_dir, 'good')
            if os.path.isdir(good_dir):
                for img_name in sorted(os.listdir(good_dir)):
                    img_path = os.path.join(cls_name, mode, 'good', img_name)
                    self.data_all.append(dict(
                        img_path=img_path,
                        mask_path='',
                        cls_name=cls_name,
                        specie_name='good',
                        anomaly=0,
                    ))

            # defect 样本 (只在 test 中存在)
            if mode == 'test':
                defect_types = [d for d in os.listdir(phase_dir) if d != 'good' and os.path.isdir(os.path.join(phase_dir, d))]
                for defect in defect_types:
                    defect_dir = os.path.join(phase_dir, defect)
                    mask_dir = os.path.join(cls_dir, 'ground_truth', defect)  # mask 与 defect 对应
                    mask_files = sorted(os.listdir(mask_dir)) if os.path.isdir(mask_dir) else []

                    img_files = sorted(os.listdir(defect_dir))
                    for idx, img_name in enumerate(img_files):
                        img_path = os.path.join(cls_name, mode, defect, img_name)
                        mask_path = os.path.join(cls_name, 'ground_truth', defect, mask_files[idx]) if idx < len(mask_files) else ''
                        self.data_all.append(dict(
                            img_path=img_path,
                            mask_path=mask_path,
                            cls_name=cls_name,
                            specie_name=defect,
                            anomaly=1,
                        ))

        # k-shot 采样
        if mode == 'train' and k_shot > 0:
            indices = torch.randint(0, len(self.data_all), (k_shot,))
            self.data_all = [self.data_all[i] for i in indices]
            if save_path:
                with open(save_path, 'w') as f:
                    for d in self.data_all:
                        f.write(d['img_path'] + '\n')

        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def get_cls_names(self):
        return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = \
            data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']

        # 读图像
        img_pil = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        # 读掩码
        if anomaly == 0 or mask_path == '':
            img_mask = Image.fromarray(np.zeros((img_pil.size[1], img_pil.size[0])), mode='L')
        else:
            mask_pil = Image.open(os.path.join(self.root, mask_path)).convert('L')
            mask_np = np.array(mask_pil) > 0
            img_mask = Image.fromarray(mask_np.astype(np.uint8) * 255, mode='L')

        # transform
        img = self.transform(img_pil) if self.transform else img_pil
        img_mask = self.target_transform(img_mask) if self.target_transform else img_mask
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
