import argparse
import logging
import os
import numpy as np
import torch
import torchvision
import threading
import torchvision.transforms as transforms
from tabulate import tabulate
from sklearn.metrics import roc_auc_score
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import math
from PIL import Image

# from model.CD_AD import CLIP_Diffusion
from model.Prototype_CDAD import CLIP_Diffusion

from sklearn.metrics import average_precision_score
from utils.metric import compute_pro, f1_score_max

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import json
import glob

from dataset.mvtec import MVTecDataset
from dataset.visa import VisaDataset
from dataset.mvtec_loco import MVTecLocoDataset
from dataset.brainmri import BrainMRIDataset
from dataset.his import HISDataset
from dataset.resc import RESCDataset
from dataset.liverct import LiverCTDataset
from dataset.chestxray import ChestXrayDataset
from dataset.oct17 import OCT17Dataset
from dataset.MPDD import MPDDDataset

from torch.utils.data import DistributedSampler, DataLoader
import torch.distributed as dist


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.manual_seed(0)


def load_few_shot_images(cls_name, dataset_name, dataset_dir, k_shot, round_idx,
                         image_transform, device, rank, world_size, image_size):
    """
    加载 few-shot 样本，rank0 从磁盘读，其它 rank 用 broadcast 同步
    """
    if rank == 0:
        if dataset_name == "mvtec":
            good_dir = os.path.join(dataset_dir, cls_name.replace(" ", "_"), "train", "good")
            candidates = sorted(
                glob.glob(os.path.join(good_dir, "*.png")) +
                glob.glob(os.path.join(good_dir, "*.PNG")) +
                glob.glob(os.path.join(good_dir, "*.jpg")) +
                glob.glob(os.path.join(good_dir, "*.JPG"))
            )

        elif dataset_name == "mvtec_loco":
            candidates = [
                f"/home/data/zjp/dataset/mvtec_loco/{cls_name.replace(' ', '_')}/train/good/{str(i).zfill(3)}.png"
                for i in range(round_idx, round_idx + k_shot)
            ]

        elif dataset_name == "visa":
            good_dir = os.path.join(dataset_dir, cls_name.replace(" ", "_"), "train", "good")
            if cls_name.replace(" ", "_") in ["capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]:
                candidates = [
                    os.path.join(good_dir, f"{str(i).zfill(3)}.JPG")
                    for i in range(round_idx, round_idx + k_shot)
                ]
            else:
                candidates = [
                    os.path.join(good_dir, f"{str(i).zfill(4)}.JPG")
                    for i in range(round_idx, round_idx + k_shot)
                ]

        elif dataset_name in ["mvtec2", "mvtec_ad2"]:
            train_dir = os.path.join(dataset_dir, cls_name.replace(" ", "_"), "train", "good")
            val_dir = os.path.join(dataset_dir, cls_name.replace(" ", "_"), "validation", "good")
            candidates = []
            for d in [train_dir, val_dir]:
                if os.path.isdir(d):
                    candidates.extend(glob.glob(os.path.join(d, "*.png")) +
                                      glob.glob(os.path.join(d, "*.jpg")))
            candidates = sorted(candidates)

        elif dataset_name in ["his", "oct17", "chestxray", "brainmri", "liverct", "resc"]:
            good_dir = f"./data/{cls_name.replace(' ', '_')}/train/good"
            files = sorted(os.listdir(good_dir))[:k_shot]
            candidates = [os.path.join(good_dir, file) for file in files]

        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported in few-shot loader")

        if len(candidates) == 0:
            raise FileNotFoundError(f"No images found for {cls_name} in dataset {dataset_name}")

        normal_image_paths = candidates[:k_shot] if dataset_name != "mvtec" else candidates[round_idx:round_idx + k_shot]

        normal_images = []
        for x in normal_image_paths:
            with Image.open(x).convert("RGB") as img:
                normal_images.append(image_transform(img).unsqueeze(0))
        normal_images = torch.cat(normal_images, dim=0).to(device)
    else:
        normal_images = torch.empty((k_shot, 3, image_size, image_size), device=device)
        normal_image_paths = None

    if world_size > 1:
        dist.broadcast(normal_images, src=0)

    return normal_images, normal_image_paths


def init_distributed(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = True
    else:
        local_rank = int(args.gpu)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        is_distributed = False

    return rank, world_size, local_rank, device, is_distributed


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def safe_auc(y_true, y_score):
    try:
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        if y_true.size == 0 or y_score.size == 0:
            return 0.5
        y_true = (y_true > 0.5).astype(np.uint8)
        if len(np.unique(y_true)) < 2:
            return 0.5
        if np.all(y_score == y_score[0]):
            return 0.5
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def setup_logger(log_file, rank):
    logger = logging.getLogger("test")
    for h in list(logger.handlers):
        logger.removeHandler(h)

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
                            datefmt="%y-%m-%d %H:%M:%S")

    if rank == 0:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test", add_help=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--gpu_clip", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--data_path", type=str, default="/home/data/zjp/dataset/mvtec")

    # parser.add_argument(
    #     "--dataset", type=str, default="brainmri", help="train dataset name"
    # )
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="/home/data/zjp/dataset/medical_datasets/BrainMRI",
    #     help="path to test dataset",
    # )
    #
    # parser.add_argument(
    #     "--dataset", type=str, default="mvtec_loco", help="train dataset name"
    # )
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="/home/data/zjp/dataset/mvtec_loco",
    #     help="path to test dataset",
    # )


    # parser.add_argument(
    #     "--dataset", type=str, default="visa", help="train dataset name"
    # )
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="/home/data/zjp/dataset/VisA_pytorch/1cls",
    #     help="path to test dataset",
    # )

    # parser.add_argument(
    #     "--dataset", type=str, default="MPDD", help="train dataset name"
    # )
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="/home/data/zjp/dataset/MPDD",
    #     help="path to test dataset",
    # )



    parser.add_argument("--save_path", type=str, default=f"./results/clip/")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--class_name", type=str, default="None")
    args = parser.parse_args()

    rank, world_size, local_rank, device, is_distributed = init_distributed(args)
    print(f"[Rank {local_rank}] Running on {device}")

    dataset_name = args.dataset
    dataset_dir = args.data_path
    k_shot = args.k_shot
    image_size = args.image_size
    save_path = args.save_path + "/" + dataset_name + "/"
    os.makedirs(save_path, exist_ok=True)
    txt_path = os.path.join(save_path, "log.txt")
    resume_path = os.path.join(save_path, "resume_results.json")

    logger = setup_logger(txt_path, rank)

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    if rank == 0 and os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            saved_results = json.load(f)
        logger.info(f"已加载进度: {list(saved_results.keys())}")
    else:
        saved_results = {}

    if rank == 0:
        with open(resume_path, "w") as f:
            json.dump(saved_results, f)

    UniVAD_model = CLIP_Diffusion(args.image_size, device=device).to(device)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    few_shot_cache = {}

    if dataset_name == "mvtec":
        test_data = MVTecDataset(root=dataset_dir, transform=transform,
                                 target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "visa":
        test_data = VisaDataset(root=dataset_dir, transform=transform,
                                target_transform=transform, mode="test")
    elif dataset_name == "MPDD":
        test_data = MPDDDataset(root=dataset_dir, transform=transform, target_transform=transform, mode="test")

    elif dataset_name == "mvtec_loco":
        test_data = MVTecLocoDataset(root=dataset_dir, transform=transform,
                                     target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name in ["mvtec2", "mvtec_ad2"]:
        from dataset.mvtec2 import MVTec2Dataset
        test_data = MVTec2Dataset(root=dataset_dir, transform=transform,
                                  target_transform=transform, mode="test", test_type=args.test_type, cls_only=None)
    elif dataset_name == "brainmri":
        test_data = BrainMRIDataset(root=dataset_dir, transform=transform,
                                    target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "his":
        test_data = HISDataset(root=dataset_dir, transform=transform,
                               target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "resc":
        test_data = RESCDataset(root=dataset_dir, transform=transform,
                                target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "chestxray":
        test_data = ChestXrayDataset(root=dataset_dir, transform=transform,
                                     target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "oct17":
        test_data = OCT17Dataset(root=dataset_dir, transform=transform,
                                 target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "liverct":
        test_data = LiverCTDataset(root=dataset_dir, transform=transform,
                                   target_transform=transform, aug_rate=-1, mode="test")
    else:
        raise NotImplementedError("Dataset not supported")

    if is_distributed:
        sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)
        test_dataloader = DataLoaderX(test_data, sampler=sampler, batch_size=args.batch_size,
                                      num_workers=2, pin_memory=False)
    else:
        test_dataloader = DataLoaderX(test_data, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4,
                                      pin_memory=True, persistent_workers=False)

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor()
    ])

    results_all = {}

    for items in tqdm(test_dataloader):
        images = items["img"].to(device)
        images_pil = items["img_pil"]
        image_paths = items["img_path"]
        cls_names = items["cls_name"]
        gt_masks = items["img_mask"]
        anomalies = items["anomaly"]

        if args.class_name != "None":
            mask = [args.class_name in p for p in image_paths]
            if not any(mask):
                continue
            images = images[mask]
            images_pil = [images_pil[i] for i in range(len(mask)) if mask[i]]
            image_paths = [image_paths[i] for i in range(len(mask)) if mask[i]]
            cls_names = [cls_names[i] for i in range(len(mask)) if mask[i]]
            gt_masks = gt_masks[mask]
            anomalies = anomalies[mask]

        if k_shot != 0:
            for cls_name in set(cls_names):
                if cls_name not in few_shot_cache:
                    normal_images, normal_image_paths = load_few_shot_images(
                        cls_name, dataset_name, dataset_dir, k_shot,
                        args.round, image_transform, device, rank, world_size, image_size
                    )
                    few_shot_cache[cls_name] = (normal_images, normal_image_paths)

                setup_data = {
                    "few_shot_samples": normal_images,
                    "dataset_category": cls_name.replace(" ", "_"),
                    "image_path": normal_image_paths,
                }
                UniVAD_model.setup(setup_data)

        UniVAD_model.eval()
        with torch.no_grad():
            outputs = UniVAD_model(images, image_paths, images_pil, cls_names)
            if not isinstance(outputs, (list, tuple)):
                outputs = (outputs,)

        for i in range(len(outputs)):
            model_key = f"model{i+1}"
            if model_key not in results_all:
                results_all[model_key] = {
                    "cls_names": [], "imgs_masks": [], "anomaly_maps": [], "gt_sp": [], "pr_sp": []
                }

        for i, pred_value in enumerate(outputs):
            model_key = f"model{i+1}"
            anomaly_scores, anomaly_maps = pred_value["pred_score"], pred_value["pred_mask"]

            for j in range(len(cls_names)):
                cls_name = cls_names[j]
                anomaly_label = anomalies[j].item()
                score = float(anomaly_scores[j].item())

                results_all[model_key]["cls_names"].append(cls_name)
                results_all[model_key]["gt_sp"].append(int(anomaly_label))
                results_all[model_key]["pr_sp"].append(score)

                am_np = anomaly_maps[j].detach().cpu().squeeze().numpy()
                if am_np.ndim == 3:
                    am_np = am_np[0]
                results_all[model_key]["anomaly_maps"].append(am_np)

                gt_np = gt_masks[j].detach().cpu().squeeze().numpy()
                results_all[model_key]["imgs_masks"].append(gt_np)

    def evaluate_results(results, logger, resume_path=None):
        table_ls, auroc_sp_ls, auroc_px_ls = [], [], []
        grouped_results = {}
        for idx, cls_name in enumerate(results["cls_names"]):
            if cls_name not in grouped_results:
                grouped_results[cls_name] = {"gt_px": [], "pr_px": [], "gt_sp": [], "pr_sp": []}
            grouped_results[cls_name]["gt_px"].append(results["imgs_masks"][idx])
            grouped_results[cls_name]["pr_px"].append(results["anomaly_maps"][idx])
            grouped_results[cls_name]["gt_sp"].append(results["gt_sp"][idx])
            grouped_results[cls_name]["pr_sp"].append(results["pr_sp"][idx])

        saved_results = {}
        for obj, data in grouped_results.items():
            gt_px = np.stack(data["gt_px"], axis=0).astype(np.uint8)
            gt_px = (gt_px > 0).astype(np.uint8)
            pr_px = np.stack(data["pr_px"], axis=0)
            gt_sp = np.asarray(data["gt_sp"])
            pr_sp = np.asarray(data["pr_sp"])

            auroc_sp = safe_auc(gt_sp, pr_sp)
            auroc_px = safe_auc(gt_px.ravel(), pr_px.ravel())

            table_ls.append([obj, f"{np.round(auroc_sp * 100, 1)}", f"{np.round(auroc_px * 100, 1)}"])
            auroc_sp_ls.append(auroc_sp)
            auroc_px_ls.append(auroc_px)

            saved_results[obj] = {"auroc_sp": float(auroc_sp), "auroc_px": float(auroc_px)}

        sp_mean = float(np.nanmean(auroc_sp_ls)) if len(auroc_sp_ls) else float("nan")
        px_mean = float(np.nanmean(auroc_px_ls)) if len(auroc_px_ls) else float("nan")
        table_ls.append(["mean", f"{np.round(sp_mean * 100, 1)}", f"{np.round(px_mean * 100, 1)}"])

        results_str = tabulate(table_ls, headers=["objects", "auroc_sp", "auroc_px"], tablefmt="pipe")
        logger.info("\n%s", results_str)

        if resume_path:
            with open(resume_path, "w") as f:
                json.dump(saved_results, f)

        return saved_results

    for model_key, res in results_all.items():
        logger.info(f"==== Evaluating {model_key} ====")
        evaluate_results(res, logger, resume_path=f"{save_path}/{model_key}_results.json")
