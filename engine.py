# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from typing import Iterable

import dataset.utils.misc as utils
import dataset.utils.loss_utils as loss_utils
import dataset.utils.eval_utils as eval_utils
import torch.nn.functional as F
# from models_albef.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xywh_to_cxcywh, box_xywh_to_cxcywh_scale

from CD_VG import CLIP_Diffusion
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def contractive_learning(logits, gt_bbox):  # b, n, sz, sz
    b, n, sz, sz = logits.shape
    logits = logits.reshape(-1, 1, sz, sz)
    gt_bbox = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, n, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
    ctr = (gt_bbox[:, :2] + gt_bbox[:, 2:]).reshape(b * n, 1, 1, 2) / 2
    neg_logits = sample_negative(logits, gt_bbox, sz).to(logits)
    sample_points = ctr * 2 - 1
    pos_logits = (F.grid_sample(logits, sample_points, padding_mode="border", align_corners=True)
                  .reshape(b * n, -1))  # b, 1, 1, 10
    logits = torch.cat([pos_logits, neg_logits], dim=-1)
    target = torch.zeros(b * n).to(gt_bbox.device).long()
    return logits, target  # check


def sample_negative(logits, gt_bboxes, size):
    bboxes = gt_bboxes  # b, 4
    cood_1d = (torch.arange(size) + 0.5) / size
    cood = cood_1d.unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda()  # b, sz
    x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1)  # b, 1, w
    y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2)  # b, h, 1
    mask = (x_mask & y_mask)  # b, h, w
    mask = (mask.reshape(gt_bboxes.shape[0], -1)) * (-1e9)  # background == 1
    sample_logits = torch.sort(logits.reshape(gt_bboxes.shape[0], -1) + mask, descending=True, dim=-1).values[:, :9]
    return sample_logits


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target, _, _ = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        pred_boxes, _, _ = model(img_data, text_data)
        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, target)

        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    # model.eval()

    save_every = 10
    import os.path as osp

    checkpoint_file = osp.splitext(args.output_file)[0] + "_checkpoint.pkl"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
        pred_box_list = checkpoint["pred_box_list"]
        gt_box_list = checkpoint["gt_box_list"]
        processed_ids = checkpoint["processed_ids"]
        print(f"恢复进度：已处理 {len(processed_ids)} 个样本")
    else:
        pred_box_list = []
        gt_box_list = []
        processed_ids = set()

    # pred_box_list = []
    # gt_box_list = []
    all_outputs = []
    if not os.path.exists(args.output_file):
        with open(args.output_file, "wb") as f:
            pass

    save_counter = 0

    for _, batch in enumerate(tqdm(data_loader)):
        # img_data: resize(512,512)-> normalize([0.5][0.5])   target：gt_bbox   pharse: str language
        img_data, target, img_id, img_paths, pharse, class_img = batch
        # 到上面没问题，
        # print(pharse)

        sample_id = str(img_id[0].item() if isinstance(img_id[0], torch.Tensor) else img_id[0])
        if sample_id in processed_ids:
            continue

        # batch =1
        import os.path as osp
        image_path = img_paths

        # output = model(image_path, img_data, pharse)
        #------------------------------------------------------------
        output = model.forward_sentences(image_path, img_data, pharse)

        #---------------------------------原来--------------------
        # batch_size = img_data.tensors.size(0)
        # # copy to GPU
        # img_data = img_data.to(device)
        # text_data = text_data.to(device)
        # target = target.to(device)
        #
        # output = model(img_data, text_data)
        # output, _, _ = model(img_data, text_data)

        pred_box_list.append(torch.tensor(output).unsqueeze(0).cpu())
        gt_box_list.append(target.cpu())
        print(target.cpu())

        processed_ids.add(sample_id)
        save_counter += 1

        if save_counter >= save_every:
            with open(checkpoint_file, "wb") as f:
                pickle.dump({
                    "pred_box_list": pred_box_list,
                    "gt_box_list": gt_box_list,
                    "processed_ids": processed_ids,
                }, f)
            save_counter = 0  # 重置计数器

    with open(checkpoint_file, "wb") as f:
        pickle.dump({
            "pred_box_list": pred_box_list,
            "gt_box_list": gt_box_list,
            "processed_ids": processed_ids,
        }, f)

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    # total_num = gt_boxes.shape[0]
    # pr5, pr6, pr7, pr8, pr9, mean_iou, cum_iou = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    acc = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)


    # result_tensor = torch.tensor([accu_num, total_num]).to(device)
    # if torch.distributed.is_available() and torch.distributed.is_initialized():
    #     torch.cuda.synchronize()
    #     dist.all_reduce(result_tensor)
    #     accuracy = float(result_tensor[0]) / float(result_tensor[1])
    # else:
    #     accuracy = float(result_tensor[0]) / float(result_tensor[1])

    # with open(args.output_file, "wb") as f:
    #     torch.save(all_outputs, f)

    # return pr5, pr6, pr7, pr8, pr9, mean_iou, cum_iou

    return acc
