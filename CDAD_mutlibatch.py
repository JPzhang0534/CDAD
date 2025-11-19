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


from torch.utils.data import Subset
from torch.utils.data import DistributedSampler, DataLoader
import torch.distributed as dist

import time
import random

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

            random.seed(args.round)  # 用 round 作为随机种子，可复现
            max_start = max(0, len(candidates) - k_shot)
            round_idx = random.randint(0, max_start)

            # candidates = [
            #     f"/home/data/zjp/dataset/mvtec/{cls_name.replace(' ', '_')}/train/good/{str(i).zfill(3)}.png"
            #     for i in range(round_idx, round_idx + k_shot)
            # ]

        elif dataset_name == "mvtec_loco":
            candidates = [
                f"/home/data/zjp/dataset/mvtec_loco/{cls_name.replace(' ', '_')}/train/good/{str(i).zfill(3)}.png"
                for i in range(round_idx, round_idx + k_shot)
            ]

        elif dataset_name == "visa":
            good_dir = os.path.join(dataset_dir, cls_name.replace(" ", "_"), "train", "good")
            # if cls_name.replace(" ", "_") in ["capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]:
            #     candidates = [
            #         os.path.join(good_dir, f"{str(i).zfill(3)}.JPG")
            #         for i in range(round_idx, round_idx + k_shot)
            #     ]
            # else:
            #     candidates = [
            #         os.path.join(good_dir, f"{str(i).zfill(4)}.JPG")
            #         for i in range(round_idx, round_idx + k_shot)
            #     ]

            good_dir = os.path.join(dataset_dir, cls_name.replace(" ", "_"), "train", "good")
            candidates = sorted(
                glob.glob(os.path.join(good_dir, "*.png")) +
                glob.glob(os.path.join(good_dir, "*.PNG")) +
                glob.glob(os.path.join(good_dir, "*.jpg")) +
                glob.glob(os.path.join(good_dir, "*.JPG"))
            )

            random.seed(args.round)  # 用 round 作为随机种子，可复现
            max_start = max(0, len(candidates) - k_shot)
            round_idx = random.randint(0, max_start)


        elif dataset_name in ["mvtec2", "mvtec_ad2"]:
            # train + validation 都可能用来取 few-shot
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

        # 截取 round_idx 对应的 few-shot
        normal_image_paths = candidates[:k_shot] if dataset_name not in ["mvtec",'visa'] else candidates[
                                                                                 round_idx:round_idx + k_shot]

        normal_images = []
        for x in normal_image_paths:
            with Image.open(x).convert("RGB") as img:
                normal_images.append(image_transform(img).unsqueeze(0))
        normal_images = torch.cat(normal_images, dim=0).to(device)
    else:
        normal_images = torch.empty((k_shot, 3, image_size, image_size), device=device)
        normal_image_paths = None

        # 同步到所有进程
    if world_size > 1:
        dist.broadcast(normal_images, src=0)

    return normal_images, normal_image_paths




def init_distributed(args):
    # 优化：明确分支，返回 is_distributed
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # multi-node / multi-gpu launched by torch.distributed.launch or torchrun
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = True
    else:
        # single GPU / single process
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

def resize_tokens(x):
    B, N, C = x.shape
    x = x.view(B, int(math.sqrt(N)), int(math.sqrt(N)), C)
    return x


def safe_auc(y_true, y_score):
    try:
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        # 空数组保护
        if y_true.size == 0 or y_score.size == 0:
            return 0.5
        y_true = (y_true > 0.5).astype(np.uint8)
        if len(np.unique(y_true)) < 2:
            return 0.5  # 没有正负样本
        if np.all(y_score == y_score[0]):
            return 0.5  # 预测全常数
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")

def cal_score(obj,data):
    gt_px = np.array(data["gt_px"])
    pr_px = np.array(data["pr_px"])
    gt_sp = np.array(data["gt_sp"])
    pr_sp = np.array(data["pr_sp"])

    auroc_sp = safe_auc(gt_sp, pr_sp)
    auroc_px = safe_auc(gt_px.ravel(), pr_px.ravel())

    table = [obj,
             str(np.round(auroc_sp * 100, 1)),
             str(np.round(auroc_px * 100, 1))]

    with lock:
        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        saved_results[obj] = {"auroc_sp": auroc_sp, "auroc_px": auroc_px}

def setup_logger(log_file, rank):
    logger = logging.getLogger("test")
    # 清理旧 handler
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
        # 非主进程不输出到控制台，减少干扰
        null_handler = logging.NullHandler()
        logger.addHandler(null_handler)

    return logger

lock = threading.Lock()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test", add_help=True)
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--k_shot", type=int, default=0, help="k-shot")
    parser.add_argument(
        "--round", type=int, default=42, help="round"    # （1，2，3，4，5）  #42
    )
    parser.add_argument("--gpu", type=int, default=3, help="单机模式下使用的 GPU id")


    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")

    # parser.add_argument("--device", type=str, default="cuda:2", help="device")
    parser.add_argument("--gpu_clip", type=int, default=0, help="单机模式下使用的 GPU id")


    parser.add_argument(
        "--dataset", type=str, default="mvtec", help="train dataset name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/data/zjp/dataset/mvtec",
        help="path to test dataset",
    )

    DS_FACTOR = 1
    #------------------------------  visa  ---------------------------------------
    # parser.add_argument(
    #     "--dataset", type=str, default="visa", help="train dataset name"
    # )
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="/home/data/zjp/dataset/VisA_pytorch/1cls",
    #     help="path to test dataset",
    # )

    #------------------------------  mvtec_loco  ---------------------------------------
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
    #     "--dataset", type=str, default="mvtec2", help="train dataset name"
    # )
    # parser.add_argument(
    #     "--test_type", type=str, default="test_private_mixed", help="test_type"    #test_private_mixed   test_private
    # )
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="/home/data/zjp/dataset/MVTec2",
    #     help="path to test dataset",
    # )

    # parser.add_argument(
    #     "--dataset", type=str, default="brainmri", help="train dataset name"
    # )
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="/home/data/zjp/dataset/medical_datasets/BrainMRI",
    #     help="path to test dataset",
    # )

    # parser.add_argument(
    #     "--dataset", type=str, default="resc", help="train dataset name"
    # )
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="/home/data/zjp/dataset/medical_datasets/RESC",
    #     help="path to test dataset",
    # )



    parser.add_argument(
        "--save_path", type=str, default=f"./results/clip/", help="path to save results"
    )

    parser.add_argument("--class_name", type=str, default="None", help="device")
    args = parser.parse_args()

    # ======================== 【多卡核心】 ========================
    rank, world_size, local_rank, device, is_distributed = init_distributed(args)
    print(f"[Rank {local_rank}] Running on {device}")


    dataset_name = args.dataset
    dataset_dir = args.data_path
    k_shot = args.k_shot

    image_size = args.image_size
    save_path = args.save_path + "/" + dataset_name + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, "log.txt")
    resume_path = os.path.join(save_path, "resume_results.json")


    logger = setup_logger(txt_path, rank)

    # record parameters
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


    # UniVAD_model = UniVAD(image_size=args.image_size).to(device)  加载模型
    UniVAD_model = CLIP_Diffusion(args.image_size,device=device).to(device)
    # dataset
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    few_shot_cache = {}


    # 数据集
    if dataset_name == "mvtec":
        test_data = MVTecDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
            cls_only=None,  # bottle  #toothbrush  wood  metal_nut        transistor  50.8 |       71.4 |       52.8 |       71.8 |
        )

    elif dataset_name == "visa":
        test_data = VisaDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            mode="test",
        )

    elif dataset_name == "mvtec_loco":
        test_data = MVTecLocoDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )

    elif dataset_name in ["mvtec2", "mvtec_ad2"]:
        from dataset.mvtec2 import MVTec2Dataset  # 假设路径和命名如此

        test_data = MVTec2Dataset(
            root=dataset_dir,  # /path/to/mvtec_ad_2
            transform=transform,
            target_transform=transform,
            mode="test",  # train / validation / test
            test_type=args.test_type,  # test_public / test_private / test_private_mixed
            cls_only=None,  # 如果只想跑某个类（如 "can"），就填 "can"
        )


    elif dataset_name == "brainmri":
        test_data = BrainMRIDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "his":
        test_data = HISDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "resc":
        test_data = RESCDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "chestxray":
        test_data = ChestXrayDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "oct17":
        test_data = OCT17Dataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "liverct":
        test_data = LiverCTDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    else:
        raise NotImplementedError("Dataset not supported")


    if is_distributed:
        sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)
        test_dataloader = DataLoaderX(
            test_data,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=False
        )
    else:
        test_dataloader = DataLoaderX(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False

        )

    with torch.no_grad():
        obj_list = [x.replace("_", " ") for x in test_data.get_cls_names()]

    results = {}
    results["cls_names"] = []
    results["imgs_masks"] = []
    results["anomaly_maps"] = []
    results["gt_sp"] = []
    results["pr_sp"] = []

    cls_last = None

    image_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )

    import tifffile

    all_latencies = []
    submission_items = []
    for items in tqdm(test_dataloader):

        images = items["img"].to(device)  # (B, C, H, W)
        images_pil = items["img_pil"]  # list
        image_paths = items["img_path"]  # list
        cls_names = items["cls_name"]  # list
        gt_masks = items["img_mask"]  # (B, H, W)
        anomalies = items["anomaly"]

        # cls 选择
        # if cls_names[0] in ( "cable",   "bottle",):
        #     pass
        # else:
        #     continue

    # "bottle",
    # "cable",
    # "capsule",
    # "carpet",
    # "grid",
    # "hazelnut",
    # "leather",
    # "metal_nut",
    # "pill",
    # "screw",
    # "tile",
    # "toothbrush",
    # "transistor",
    # "wood",
    # "zipper",


        if args.dataset in ["mvtec2", "mvtec_ad2"]:
            need_infer_mask = []
            for i in range(len(cls_names)):
                cls_name = cls_names[i]
                image_path = image_paths[i]
                basename = os.path.splitext(os.path.basename(image_path))[0]
                tiff_path = os.path.join(save_path, "submission_folder", "anomaly_images", cls_name, args.test_type,
                                         f"{basename}.tiff")

                # 如果 TIFF 已存在，就跳过这一张图
                if os.path.exists(tiff_path):
                    need_infer_mask.append(False)
                else:
                    need_infer_mask.append(True)

            # 如果这一 batch 全部都已经存在，则跳过整个 batch
            if not any(need_infer_mask):
                continue



        if args.class_name != "None":
            mask = [args.class_name in p for p in image_paths]
            if not any(mask):
                continue
            # 可选：只保留符合条件的样本
            images = images[mask]
            images_pil = [images_pil[i] for i in range(len(mask)) if mask[i]]
            image_paths = [image_paths[i] for i in range(len(mask)) if mask[i]]
            cls_names = [cls_names[i] for i in range(len(mask)) if mask[i]]
            gt_masks = gt_masks[mask]
            anomalies = anomalies[mask]

        if k_shot!=0:

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
                if k_shot == 0:
                    pass
                else:
                    UniVAD_model.setup(setup_data)
                cls_last = cls_name


        UniVAD_model.eval()

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            pred_value = UniVAD_model(images, image_paths, images_pil, cls_names)
            anomaly_scores, anomaly_maps = pred_value["pred_score"], pred_value["pred_mask"]

        torch.cuda.synchronize()
        end = time.time()

        latency = (end - start) * 1000  # ms
        all_latencies.append(latency)




        # for i in range(len(cls_names)):
        #     # 保存到磁盘而不是 results
        #     cls_name = cls_names[i]
        #     image_path = image_paths[i]
        #     am_np = anomaly_maps[i].detach().cpu().squeeze().astype(np.float16)
        #     save_path_cls = os.path.join(save_path, cls_name)
        #     os.makedirs(save_path_cls, exist_ok=True)
        #     np.save(os.path.join(save_path_cls, os.path.basename(image_path) + "_anom.npy"), am_np)
        #
        #     # 保存图像级 score
        #     with open(os.path.join(save_path_cls, os.path.basename(image_path) + "_score.txt"), "w") as f:
        #         f.write(str(float(anomaly_scores[i].item())))

        for i in range(len(cls_names)):
            cls_name = cls_names[i]
            image_path = image_paths[i]
            anomaly_label = anomalies[i].item()    # score = 0;  表示正常，score 1 表示异常，
            score = float(anomaly_scores[i].item())

            # 统一 append 最小信息（gt_sp/pr_sp 等）以便之后统计/保存
            results["cls_names"].append(cls_name)
            results["gt_sp"].append(int(anomaly_label))
            results["pr_sp"].append(score)

            if args.dataset in ["mvtec2", "mvtec_ad2"] and args.test_type in ["test_private", "test_private_mixed"]:
                # basename: 保持原始文件名（例如 000_regular / 000_mixed）
                basename = os.path.splitext(os.path.basename(image_path))[0]

                # 转 numpy 并保证是 HxW float16
                am = anomaly_maps[i].detach().cpu().squeeze().numpy()
                if am.ndim == 3:
                    am = am[0]  # 常见情况：C x H x W -> 取第一通道
                am = am.astype(np.float16)

                # 保存路径（官方期望的结构）
                submission_root = os.path.join(save_path, "submission_folder")
                tiff_dir = os.path.join(submission_root, "anomaly_images", cls_name, args.test_type)
                png_dir = os.path.join(submission_root, "anomaly_images_thresholded", cls_name, args.test_type)
                os.makedirs(tiff_dir, exist_ok=True)
                os.makedirs(png_dir, exist_ok=True)

                # 保存非二值化 anomaly map (.tiff, float16)
                tiff_path = os.path.join(tiff_dir, f"{basename}.tiff")
                if not os.path.exists(tiff_path):
                    # 保存非二值化 anomaly map (.tiff, float16)
                    tifffile.imwrite(tiff_path, am)

                    # 保存二值化 anomaly map (.png, uint8) —— 可选
                    am_bin = (am > 0.5).astype(np.uint8) * 255
                    png_path = os.path.join(png_dir, f"{basename}.png")
                    Image.fromarray(am_bin).save(png_path)

                    # 记录 submission 项（用于本地汇总）
                    submission_items.append((cls_name, basename, score, tiff_path, png_path))
                else:
                    # 如果已经存在，说明重复保存了，跳过
                    pass
                    # print(f"Skip duplicate: {tiff_path}")

            else:
                # test_public 或非 mvtec2 数据集：保留 anomaly map 以便后面做 AUROC（原来的逻辑）
                am_np = anomaly_maps[i].detach().cpu().squeeze().numpy()
                if am_np.ndim == 3:
                    am_np = am_np[0]
                # 可能做下采样或其它处理（保持原逻辑）
                if DS_FACTOR > 1:
                    am_np = am_np[::DS_FACTOR, ::DS_FACTOR]
                results["anomaly_maps"].append(am_np)

                gt_np = gt_masks[i].detach().cpu().squeeze().numpy()
                if DS_FACTOR > 1:
                    gt_np = gt_np[::DS_FACTOR, ::DS_FACTOR]
                results["imgs_masks"].append(gt_np)

        del images, pred_value, anomaly_maps
        torch.cuda.empty_cache()


        # 有问题
        # #保存 batch 内所有样本
        # for i in range(len(cls_names)):
        #     cls_name = cls_names[i]
        #     image_path = image_paths[i]
        #     gt_mask = gt_masks[i]
        #     anomaly_label = anomalies[i].item()
        #
        #     results["cls_names"].append(cls_name)
        #     gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        #     results["imgs_masks"].append(gt_mask)
        #
        #     results["gt_sp"].append(anomaly_label)
        #     overall_anomaly_score = anomaly_scores[i].item()
        #
        #     # am_np = anomaly_maps[i].detach().cpu().squeeze()
        #     am_np = anomaly_maps[i].detach().cpu().squeeze().numpy().astype(np.float16)
        #
        #     if am_np.ndim == 3:
        #         am_np = am_np[0]
        #     if DS_FACTOR > 1:
        #         am_np = am_np[::DS_FACTOR, ::DS_FACTOR]
        #     results["anomaly_maps"].append(am_np)
        #     results["pr_sp"].append(float(overall_anomaly_score))
        #
        # del images, pred_value, anomaly_maps
        # torch.cuda.empty_cache()

    if is_distributed:
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, results)

        if rank == 0:
            merged_results = {"cls_names": [], "imgs_masks": [], "anomaly_maps": [], "gt_sp": [], "pr_sp": []}
            for g in gathered_results:
                if g is None:
                    continue
                merged_results["cls_names"].extend(g["cls_names"])
                merged_results["imgs_masks"].extend(g["imgs_masks"])
                merged_results["anomaly_maps"].extend(g["anomaly_maps"])
                merged_results["gt_sp"].extend(g["gt_sp"])
                merged_results["pr_sp"].extend(g["pr_sp"])
            results = merged_results
        else:
            results = None
    else:
        # 单卡直接用
        results = results

    def to_numpy(x):
        if isinstance(x, np.ndarray):
                return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)


    if args.dataset in ["mvtec2", "mvtec_ad2"] and args.test_type in ["test_private", "test_private_mixed"]:
        submission_root = os.path.join(save_path, "submission_folder")
        # 生成一个 CSV / JSON 总表，便于检查（官方不要求，但方便你本地查看）
        csv_path = os.path.join(save_path, f"submission_{args.test_type}_scores.csv")
        import csv

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "basename", "score", "tiff_path", "png_path"])
            for cls_name, basename, score, tiff_path, png_path in submission_items:
                writer.writerow([cls_name, basename, score, tiff_path, png_path])

        logger.info(f"⚠️ 私有测试（{args.test_type}）没有 GT，已跳过 AUROC 计算。")
        logger.info(f"✅ 已按官方结构把 anomaly maps 保存到: {os.path.join(submission_root, 'anomaly_images')}")
        logger.info(f"✅ 二值图（若需）保存在: {os.path.join(submission_root, 'anomaly_images_thresholded')}")
        logger.info(f"✅ 汇总分数保存在: {csv_path}")
        logger.info("⚠️ 请使用官方 checker 打包并校验：tar -czvf submission_mvtec2.tar.gz submission_folder/")

            # =====================================================
            # 🔽 保存官方提交所需的 anomaly maps
            # =====================================================
            # import tifffile
            #
            # submission_root = os.path.join(save_path, "submission_folder")
            # anomaly_dir = os.path.join(submission_root, "anomaly_images")
            # anomaly_bin_dir = os.path.join(submission_root, "anomaly_images_thresholded")
            #
            # for idx, cls_name in enumerate(results["cls_names"]):
            #     # ⚠️ test_type 要和你加载数据时一致（"test_private" / "test_private_mixed"）
            #     test_type = args.test_type
            #
            #     # 文件名规则：000_regular / 000_mixed
            #     if test_type == "test_private":
            #         file_id = f"{str(idx).zfill(3)}_regular"
            #     elif test_type == "test_private_mixed":
            #         file_id = f"{str(idx).zfill(3)}_mixed"
            #     else:
            #         raise ValueError(f"Unsupported test_type {test_type}")
            #
            #     # ========== 保存 anomaly maps ==========
            #     am = results["anomaly_maps"][idx]
            #     am = np.asarray(am, dtype=np.float16)  # float16
            #     save_path_tiff = os.path.join(anomaly_dir, cls_name, test_type)
            #     os.makedirs(save_path_tiff, exist_ok=True)
            #     tifffile.imwrite(os.path.join(save_path_tiff, file_id + ".tiff"), am)
            #
            #     # ========== 保存二值化 anomaly maps ==========
            #     am_bin = (am > 0.5).astype(np.uint8) * 255
            #     save_path_png = os.path.join(anomaly_bin_dir, cls_name, test_type)
            #     os.makedirs(save_path_png, exist_ok=True)
            #     Image.fromarray(am_bin).save(os.path.join(save_path_png, file_id + ".png"))
            #
            # logger.info(f"✅ 提交文件已生成在: {submission_root}")
            # logger.info("⚠️ 请手动打包上传: tar -czvf submission_mvtec2.tar.gz submission_folder/")

    else:
        # ================= metrics 计算逻辑（仅 test_public）=================
        table_ls = []
        auroc_sp_ls = []
        auroc_px_ls = []

        ap_sp_ls, ap_px_ls = [], []
        f1_sp_ls, f1_px_ls = [], []
        aupro_px_ls = []

        DS_FACTOR = 1
        grouped_results = {}
        for idx, cls_name in enumerate(results["cls_names"]):
            if cls_name not in grouped_results:
                grouped_results[cls_name] = {
                    "gt_px": [], "pr_px": [],
                    "gt_sp": [], "pr_sp": []
                }

            gt_px_i = to_numpy(results["imgs_masks"][idx]).squeeze()
            pr_px_i = to_numpy(results["anomaly_maps"][idx]).squeeze()

            grouped_results[cls_name]["gt_px"].append(gt_px_i)
            grouped_results[cls_name]["pr_px"].append(pr_px_i)
            grouped_results[cls_name]["gt_sp"].append(results["gt_sp"][idx])
            grouped_results[cls_name]["pr_sp"].append(results["pr_sp"][idx])

        for obj in grouped_results.keys():
            gt_px = np.stack(grouped_results[obj]["gt_px"], axis=0).astype(np.uint8)
            gt_px = (gt_px > 0).astype(np.uint8)
            pr_px = np.stack(grouped_results[obj]["pr_px"], axis=0)

            if DS_FACTOR > 1:
                H, W = gt_px.shape[1], gt_px.shape[2]
                if H >= DS_FACTOR and W >= DS_FACTOR:
                    gt_px = gt_px[:, ::DS_FACTOR, ::DS_FACTOR]
                    pr_px = pr_px[:, ::DS_FACTOR, ::DS_FACTOR]

            gt_sp = np.asarray(grouped_results[obj]["gt_sp"])
            pr_sp = np.asarray(grouped_results[obj]["pr_sp"])

            #   多指标计算方式
            # --- 统一拉平 ---
            gt_px_flat, pr_px_flat = gt_px.ravel(), pr_px.ravel()
            gt_sp_flat, pr_sp_flat = gt_sp.ravel(), pr_sp.ravel()

            # --- 指标计算 ---
            auroc_sp = safe_auc(gt_sp_flat, pr_sp_flat)
            auroc_px = safe_auc(gt_px_flat, pr_px_flat)
            ap_sp = average_precision_score(gt_sp_flat, pr_sp_flat)
            ap_px = average_precision_score(gt_px_flat, pr_px_flat)
            f1_sp = f1_score_max(gt_sp_flat, pr_sp_flat)
            f1_px = f1_score_max(gt_px_flat, pr_px_flat)
            aupro_px = compute_pro(gt_px, pr_px)

            # --- 存结果 ---
            table_ls.append([
                obj,
                f"{np.round(auroc_sp * 100, 1)}",
                f"{np.round(ap_sp * 100, 1)}",
                f"{np.round(f1_sp * 100, 1)}",
                f"{np.round(auroc_px * 100, 1)}",
                f"{np.round(ap_px * 100, 1)}",
                f"{np.round(f1_px * 100, 1)}",
                f"{np.round(aupro_px * 100, 1)}"
            ])
            auroc_sp_ls.append(auroc_sp)
            ap_sp_ls.append(ap_sp)
            f1_sp_ls.append(f1_sp)
            auroc_px_ls.append(auroc_px)
            ap_px_ls.append(ap_px)
            f1_px_ls.append(f1_px)
            aupro_px_ls.append(aupro_px)

            saved_results[obj] = {
                "auroc_sp": float(auroc_sp),
                "ap_sp": float(ap_sp),
                "f1_sp": float(f1_sp),
                "auroc_px": float(auroc_px),
                "ap_px": float(ap_px),
                "f1_px": float(f1_px),
                "aupro_px": float(aupro_px)
            }

            # --- 平均值 ---
        table_ls.append([
            "mean",
            f"{np.round(np.nanmean(auroc_sp_ls) * 100, 1)}",
            f"{np.round(np.nanmean(ap_sp_ls) * 100, 1)}",
            f"{np.round(np.nanmean(f1_sp_ls) * 100, 1)}",
            f"{np.round(np.nanmean(auroc_px_ls) * 100, 1)}",
            f"{np.round(np.nanmean(ap_px_ls) * 100, 1)}",
            f"{np.round(np.nanmean(f1_px_ls) * 100, 1)}",
            f"{np.round(np.nanmean(aupro_px_ls) * 100, 1)}"
        ])

        results_str = tabulate(
            table_ls,
            headers=["objects", "auroc_sp", "ap_sp", "f1_sp", "auroc_px", "ap_px", "f1_px", "aupro_px"],
            tablefmt="pipe"
        )
        print(results_str)
        logger.info("\n%s", results_str)

        avg_latency = np.mean(all_latencies)
        print(f"平均推理时间: {avg_latency:.2f} ms / 每张图")

        with open(resume_path, "w") as f:
            json.dump(saved_results, f)









        #      #auroc_sp  auroc_px
        #     auroc_sp = safe_auc(gt_sp, pr_sp)
        #     auroc_px = safe_auc(gt_px.ravel(), pr_px.ravel())
        #
        #     table_ls.append([obj, f"{np.round(auroc_sp * 100, 1)}", f"{np.round(auroc_px * 100, 1)}"])
        #     auroc_sp_ls.append(auroc_sp)
        #     auroc_px_ls.append(auroc_px)
        #
        #     saved_results[obj] = {"auroc_sp": float(auroc_sp), "auroc_px": float(auroc_px)}
        #
        # sp_mean = float(np.nanmean(auroc_sp_ls)) if len(auroc_sp_ls) else float("nan")
        # px_mean = float(np.nanmean(auroc_px_ls)) if len(auroc_px_ls) else float("nan")
        #
        # table_ls.append(["mean", f"{np.round(sp_mean * 100, 1)}", f"{np.round(px_mean * 100, 1)}"])
        #
        # results_str = tabulate(table_ls, headers=["objects", "auroc_sp", "auroc_px"], tablefmt="pipe")
        # print(results_str)
        # logger.info("\n%s", results_str)
        #
        # with open(resume_path, "w") as f:
        #     json.dump(saved_results, f)

    #
    # #todo: 1. clip  vit-H-14 调试  不行 视觉分支太深了
    # #      2. 4 few shot clip+diffusion数据
    # #      3. clip--> clip_img feature  --> diffusion  1. concatenate text， 挑 crossmap，  2. 添加到selfattention 来reweight  3. img_feature 是不是多层 feature