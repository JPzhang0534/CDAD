import torch
import numpy as np
from sklearn.cluster import KMeans

import torch
import numpy as np
from sklearn.cluster import KMeans


def select_anomaly_prototypes(image_token, weights, M=200, K=10, threshold=None):
    """
    基于高分筛选+聚类的异常原型选择
    Args:
        image_token: Tensor [B, L, C], 每个patch的特征
        weights: Tensor [B, L], 每个patch的异常响应分数
        M: int, 初选的Top-M候选（如果threshold为None时用）
        K: int, 聚类后选K个原型
        threshold: float or None, 如果设定则直接用阈值筛选

    Returns:
        prototypes: Tensor [B, K, C]
    """
    B, L, C = image_token.shape
    prototypes = []

    for b in range(B):
        # Step1: 筛选高分patch
        if threshold is not None:
            mask = weights[b] > threshold
            candidate_idx = mask.nonzero(as_tuple=False).squeeze(1)
        else:
            _, topM_idx = torch.topk(weights[b], M, dim=0)
            candidate_idx = topM_idx

        if candidate_idx.numel() == 0:    #  不会运行这个，因为对map归一化了
            # 没有候选，fallback: 直接取全局top-K
            _, topK_idx = torch.topk(weights[b], K, dim=0)
            feats = image_token[b, topK_idx]
            prototypes.append(feats)
            # print('没有候选')
            continue

        feats = image_token[b, candidate_idx]  # [Nc, C]
        scores = weights[b, candidate_idx]  # [Nc]

        # Step2: 聚类
        feats_np = feats.detach().cpu().numpy()
        if feats_np.shape[0] < K:
            # 候选不足K，直接全取
            selected = feats
        else:
            kmeans = KMeans(n_clusters=K, random_state=0).fit(feats_np)
            selected = []
            for i in range(K):
                cluster_indices = np.where(kmeans.labels_ == i)[0]
                if len(cluster_indices) == 0:
                    continue
                # Step3: 在簇内选响应值最高的patch，而不是质心
                cluster_scores = scores[cluster_indices]
                best_idx = cluster_indices[cluster_scores.argmax().item()]
                selected.append(feats[best_idx])
            selected = torch.stack(selected, dim=0)

        prototypes.append(selected)

    prototypes = torch.stack(prototypes, dim=0)  # [B, K, C]
    return prototypes


import torch
import torch.nn.functional as F


import torch
import torch.nn.functional as F

class PrototypeMemoryBank:
    def __init__(
        self,
        anchors: torch.Tensor,          # [S, D_proj]
        normal_anchors: torch.Tensor,   # [1, D_proj] 或 [D_proj]
        proj_layer: torch.nn.Parameter, # [D_raw, D_proj]
        capacity: int = 50,
        momentum: float = 0.9,
        unknown_capacity: int = 100,
        sim_thresh: float = 0.05,
    ):
        self.device = anchors.device
        self.dtype = anchors.dtype

        self.anchors = anchors  # [S, D_proj]
        self.S, self.D_proj = self.anchors.shape

        self.proj = proj_layer  # nn.Parameter [D_raw, D_proj]
        self.D_raw = proj_layer.shape[0]

        self.capacity = capacity
        self.momentum = momentum
        self.unknown_capacity = unknown_capacity
        self.sim_thresh = 0

        self.slots = [torch.zeros((0, self.D_raw), device=self.device, dtype=self.dtype) for _ in range(self.S)]
        self.unknown = torch.zeros((0, self.D_raw), device=self.device, dtype=self.dtype)

        if normal_anchors.ndim == 1:
            self.normal_anchor = normal_anchors.unsqueeze(0)  # [1, D_proj]
        else:
            self.normal_anchor = normal_anchors

    def project(self, feats_raw: torch.Tensor) -> torch.Tensor:
        """Raw → proj 空间"""
        return F.normalize(feats_raw @ self.proj, dim=-1)

    def assign_slot(self, p_raw: torch.Tensor):
        """返回最合适的单个 slot id"""
        p_proj = self.project(p_raw.unsqueeze(0))  # [1, D_proj]

        sims_anom = (p_proj @ self.anchors.T).squeeze(0)  # [S]

        sim_norm = 0.0
        if self.normal_anchor is not None:
            sim_norm = (p_proj @ self.normal_anchor.T).item()

        diffs = sims_anom - sim_norm
        max_diff, slot_id = torch.max(diffs, dim=0)

        if max_diff.item() > self.sim_thresh:
            return int(slot_id.item()), float(max_diff.item())
        else:
            return -1, float(max_diff.item())

    def assign_slots(self, p_raw: torch.Tensor):
        """返回符合条件的多个 slot id"""
        p_proj = self.project(p_raw.unsqueeze(0))  # [1, D_proj]
        sims_anom = (p_proj @ self.anchors.T).squeeze(0)  # [S]

        sim_norm = 0.0
        if self.normal_anchor is not None:
            sim_norm = (p_proj @ self.normal_anchor.T).item()

        slot_ids = [i for i, sim in enumerate(sims_anom) if (sim.item() - sim_norm) > 0]
        return slot_ids

    def update(self, prototypes_raw: torch.Tensor):
        """保存 raw 特征到 memory bank"""
        if prototypes_raw.ndim == 2:
            protos = prototypes_raw
        else:
            protos = prototypes_raw.view(-1, self.D_raw)

        protos = F.normalize(protos.to(self.device, dtype=self.dtype), dim=-1)

        for p in protos:
            slot_id, score = self.assign_slot(p)
            if slot_id >= 0:
                if self.slots[slot_id].shape[0] < self.capacity:
                    self.slots[slot_id] = torch.cat([self.slots[slot_id], p.unsqueeze(0)], dim=0)
                else:
                    self.slots[slot_id][-1] = self.momentum * self.slots[slot_id][-1] + (1 - self.momentum) * p
            else:
                if self.unknown.shape[0] < self.unknown_capacity:
                    self.unknown = torch.cat([self.unknown, p.unsqueeze(0)], dim=0)
                else:
                    self.unknown[-1] = self.momentum * self.unknown[-1] + (1 - self.momentum) * p

    def fuse_with_memory(self, prototypes_raw: torch.Tensor, alpha: float = 0.5, update_after: bool = False):
        """融合 raw 特征与 memory bank (可能属于多个异常类)"""
        B, K, _ = prototypes_raw.shape
        fused_list = []

        for b in range(B):
            cur = prototypes_raw[b]  # [K, D_raw]
            fused_b = []
            for p in cur:
                slot_ids = self.assign_slots(p)
                if len(slot_ids) > 0:
                    mem_feats = [self.slots[sid].mean(dim=0) for sid in slot_ids if self.slots[sid].shape[0] > 0]
                    fused_p = alpha * p + (1 - alpha) * torch.stack(mem_feats, dim=0).mean(dim=0) if mem_feats else p
                else:
                    fused_p = p
                fused_b.append(F.normalize(fused_p, dim=-1))
            fused_list.append(torch.stack(fused_b, dim=0))

        fused = torch.stack(fused_list, dim=0)

        if update_after:
            self.update(prototypes_raw)

        return fused

    def get_bank_raw(self):
        bank = [s for s in self.slots if s.shape[0] > 0]
        if self.unknown.shape[0] > 0:
            bank.append(self.unknown)
        if not bank:
            return torch.zeros((0, self.D_raw), device=self.device, dtype=self.dtype)
        return F.normalize(torch.cat(bank, dim=0), dim=-1)

    def get_bank_proj(self):
        bank_raw = self.get_bank_raw()
        if bank_raw.shape[0] == 0:
            return torch.zeros((0, self.D_proj), device=self.device, dtype=self.dtype)
        return self.project(bank_raw)

    def expand_with_memory(self, prototypes_raw: torch.Tensor, n: int = 3, update_after: bool = False):
        """用 memory bank 中的特征扩展 prototypes"""
        B, K, _ = prototypes_raw.shape
        expanded_list = []

        for b in range(B):
            cur = prototypes_raw[b]  # [K, D_raw]
            new_feats = []
            for p in cur:
                slot_id, score = self.assign_slot(p)
                if slot_id >= 0 and self.slots[slot_id].shape[0] > 0:
                    mem_pool = self.slots[slot_id]
                    if mem_pool.shape[0] >= n:
                        idx = torch.randperm(mem_pool.shape[0], device=self.device)[:n]
                        mem_feats = mem_pool[idx]  # [n, D_raw]
                    else:
                        mem_feats = mem_pool
                    new_feats.append(mem_feats)
            if new_feats:
                new_feats = torch.cat(new_feats, dim=0)
                expanded = torch.cat([cur, new_feats], dim=0)
            else:
                expanded = cur
            expanded_list.append(expanded)

        # pad batch 内长度一致
        max_len = max(x.shape[0] for x in expanded_list)
        padded = []
        for x in expanded_list:
            if x.shape[0] < max_len:
                pad = torch.zeros((max_len - x.shape[0], self.D_raw), device=self.device, dtype=self.dtype)
                padded.append(torch.cat([x, pad], dim=0))
            else:
                padded.append(x)
        expanded = torch.stack(padded, dim=0)

        if update_after:
            self.update(prototypes_raw)

        return expanded




# class PrototypeMemoryBank:
#     def __init__(
#         self,
#         anchors: torch.Tensor,          # [S, D_proj]
#         normal_anchors:torch.Tensor,
#         proj_layer: torch.nn.Parameter, # [D_raw, D_proj]
#         capacity: int = 50,
#         momentum: float = 0.9,
#         unknown_capacity: int = 100,
#         sim_thresh: float = 0.05,
#     ):
#         """
#         anchors: [S, D_proj] CLIP text prompt 特征 (已归一化)
#         proj_layer: nn.Parameter, 形状 [D_raw, D_proj]
#         """
#         self.device = anchors.device
#         self.dtype = anchors.dtype
#
#         self.anchors = anchors # [S, D_proj]
#         self.S, self.D_proj = self.anchors.shape
#
#         self.proj = proj_layer    # nn.Parameter (raw_dim, D_proj)
#         self.D_raw = proj_layer.shape[0]
#
#         self.capacity = capacity
#         self.momentum = momentum
#         self.unknown_capacity = unknown_capacity
#         self.sim_thresh = sim_thresh
#
#         # raw 特征 slots
#         self.slots = [torch.zeros((0, self.D_raw), device=self.device, dtype=self.dtype) for _ in range(self.S)]
#         self.unknown = torch.zeros((0, self.D_raw), device=self.device, dtype=self.dtype)
#
#         self.normal_anchor = normal_anchors
#
#     def project(self, feats_raw: torch.Tensor) -> torch.Tensor:
#         """
#         raw → proj 空间
#         feats_raw: [N, D_raw]
#         return: [N, D_proj] (L2 归一化)
#         """
#         return F.normalize(feats_raw @ self.proj, dim=-1)
#
#     def assign_slot(self, p_raw: torch.Tensor):
#         """
#         返回最合适的一个 slot id（只能进一个）
#         """
#         p_proj = self.project(p_raw.unsqueeze(0))  # [1, D_proj]
#         sims_anom = torch.matmul(self.anchors, p_proj.squeeze(0))  # [S]
#
#         sim_norm = 0.0
#         if self.normal_anchor is not None:
#             sim_norm = torch.matmul(self.normal_anchor, p_proj.squeeze(0)).item()
#
#         # 计算 anomaly - normal
#         diffs = sims_anom - sim_norm
#
#         # 选最大差值对应的 slot
#         max_diff, slot_id = torch.max(diffs, dim=0)
#
#         if max_diff.item() > self.sim_thresh:
#             return int(slot_id.item()), float(max_diff.item())
#         else:
#             return -1, float(max_diff.item())
#
#     def update(self, prototypes_raw: torch.Tensor):
#         """保存 raw 特征到 memory bank"""
#         if prototypes_raw.ndim == 2:
#             protos = prototypes_raw
#         else:
#             protos = prototypes_raw.view(-1, self.D_raw)
#
#         protos = F.normalize(protos.to(self.device, dtype=self.dtype), dim=-1)
#
#         for p in protos:
#             slot_id, score = self.assign_slot(p)
#             if slot_id >= 0:
#                 if self.slots[slot_id].shape[0] < self.capacity:
#                     self.slots[slot_id] = torch.cat([self.slots[slot_id], p.unsqueeze(0)], dim=0)
#                 else:
#                     self.slots[slot_id][-1] = (
#                             self.momentum * self.slots[slot_id][-1] + (1 - self.momentum) * p
#                     )
#             else:
#                 if self.unknown.shape[0] < self.unknown_capacity:
#                     self.unknown = torch.cat([self.unknown, p.unsqueeze(0)], dim=0)
#                 else:
#                     self.unknown[-1] = self.momentum * self.unknown[-1] + (1 - self.momentum) * p
#
#
#     # def update(self, prototypes_raw: torch.Tensor):
#     #     """保存 raw 特征到 memory bank"""
#     #     if prototypes_raw.ndim == 2:
#     #         protos = prototypes_raw
#     #     else:
#     #         protos = prototypes_raw.view(-1, self.D_raw)
#     #
#     #     protos = F.normalize(protos.to(self.device, dtype=self.dtype), dim=-1)
#     #
#     #     for p in protos:
#     #         slot_id, sim = self.assign_slot(p)
#     #         if slot_id >= 0:
#     #             if self.slots[slot_id].shape[0] < self.capacity:
#     #                 self.slots[slot_id] = torch.cat([self.slots[slot_id], p.unsqueeze(0)], dim=0)
#     #             else:
#     #                 self.slots[slot_id][-1] = (
#     #                     self.momentum * self.slots[slot_id][-1] + (1 - self.momentum) * p
#     #                 )
#     #         else:
#     #             if self.unknown.shape[0] < self.unknown_capacity:
#     #                 self.unknown = torch.cat([self.unknown, p.unsqueeze(0)], dim=0)
#     #             else:
#     #                 self.unknown[-1] = self.momentum * self.unknown[-1] + (1 - self.momentum) * p
#
#     # def fuse_with_memory(self, prototypes_raw: torch.Tensor, alpha: float = 0.5, update_after: bool = False):
#     #     """
#     #     融合 raw 特征与 memory bank (同类 slot)
#     #     return: [B, K, D_raw] 融合后的 raw 特征
#     #     """
#     #     B, K, _ = prototypes_raw.shape
#     #     fused_list = []
#     #
#     #     for b in range(B):
#     #         cur = prototypes_raw[b]   # [K, D_raw]
#     #         fused_b = []
#     #         for p in cur:
#     #             slot_id, sim = self.assign_slot(p)
#     #             if slot_id >= 0 and self.slots[slot_id].shape[0] > 0:
#     #                 mem_mean = self.slots[slot_id].mean(dim=0)
#     #                 fused_p = alpha * p + (1 - alpha) * mem_mean
#     #                 # fused_p = torch.cat(p,mem_mean,dim=1)
#     #             else:
#     #                 fused_p = p
#     #             fused_b.append(F.normalize(fused_p, dim=-1))
#     #         fused_list.append(torch.stack(fused_b, dim=0))
#     #
#     #     fused = torch.stack(fused_list, dim=0)
#     #
#     #     if update_after:
#     #         self.update(prototypes_raw)
#     #
#     #     return fused
#
#     def fuse_with_memory(self, prototypes_raw: torch.Tensor, alpha: float = 0.5, update_after: bool = False):
#         """
#         融合 raw 特征与 memory bank (可能属于多个异常类)
#         return: [B, K, D_raw] 融合后的 raw 特征
#         """
#         B, K, _ = prototypes_raw.shape
#         fused_list = []
#
#         for b in range(B):
#             cur = prototypes_raw[b]  # [K, D_raw]
#             fused_b = []
#             for p in cur:
#                 slot_ids = self.assign_slot(p)
#                 if len(slot_ids) > 0:
#                     # 取多个 slot 的均值
#                     mem_feats = []
#                     for sid in slot_ids:
#                         if self.slots[sid].shape[0] > 0:
#                             mem_feats.append(self.slots[sid].mean(dim=0))
#                     if len(mem_feats) > 0:
#                         mem_mean = torch.stack(mem_feats, dim=0).mean(dim=0)
#                         fused_p = alpha * p + (1 - alpha) * mem_mean
#                     else:
#                         fused_p = p
#                 else:
#                     fused_p = p
#                 fused_b.append(F.normalize(fused_p, dim=-1))
#             fused_list.append(torch.stack(fused_b, dim=0))
#
#         fused = torch.stack(fused_list, dim=0)
#
#         if update_after:
#             self.update(prototypes_raw)
#
#         return fused
#
#     def get_bank_raw(self):
#         """取出 raw 特征"""
#         bank = []
#         for s in self.slots:
#             if s.shape[0] > 0:
#                 bank.append(s)
#         if self.unknown.shape[0] > 0:
#             bank.append(self.unknown)
#         if len(bank) == 0:
#             return torch.zeros((0, self.D_raw), device=self.device, dtype=self.dtype)
#         return F.normalize(torch.cat(bank, dim=0), dim=-1)
#
#     def get_bank_proj(self):
#         """取出 proj 特征"""
#         bank_raw = self.get_bank_raw()
#         if bank_raw.shape[0] == 0:
#             return torch.zeros((0, self.D_proj), device=self.device, dtype=self.dtype)
#         return self.project(bank_raw)
#
#     def expand_with_memory(self, prototypes_raw: torch.Tensor, n: int = 5, update_after: bool = False):
#         """
#         用 memory bank 中的特征扩展 prototypes
#         每个 prototype 只会选择一个最合适的 slot 来扩展
#         prototypes_raw: [B, K, D_raw]
#         return: [B, K+n_max, D_raw]
#         """
#         B, K, _ = prototypes_raw.shape
#         expanded_list = []
#
#         for b in range(B):
#             cur = prototypes_raw[b]  # [K, D_raw]
#             new_feats = []
#             for p in cur:
#                 slot_id, score = self.assign_slot(p)
#                 if slot_id >= 0 and self.slots[slot_id].shape[0] > 0:
#                     # 从 memory bank 对应 slot 中挑 n 个
#                     mem_pool = self.slots[slot_id]
#                     if mem_pool.shape[0] >= n:
#                         idx = torch.randperm(mem_pool.shape[0], device=self.device)[:n]
#                         mem_feats = mem_pool[idx]  # [n, D_raw]
#                     else:
#                         mem_feats = mem_pool  # 少于 n 就全拿
#                     new_feats.append(mem_feats)
#             if len(new_feats) > 0:
#                 # 把多个 prototype 的扩展结果拼接
#                 new_feats = torch.cat(new_feats, dim=0)  # [n_total, D_raw]
#                 expanded = torch.cat([cur, new_feats], dim=0)  # [K+n_total, D_raw]
#             else:
#                 expanded = cur
#             expanded_list.append(expanded)
#
#         # pad 成相同长度 (避免 batch 内长度不一致)
#         max_len = max(x.shape[0] for x in expanded_list)
#         padded = []
#         for x in expanded_list:
#             if x.shape[0] < max_len:
#                 pad = torch.zeros((max_len - x.shape[0], self.D_raw), device=self.device, dtype=self.dtype)
#                 padded.append(torch.cat([x, pad], dim=0))
#             else:
#                 padded.append(x)
#         expanded = torch.stack(padded, dim=0)  # [B, K+n_max, D_raw]
#
#         if update_after:
#             self.update(prototypes_raw)
#
#         return expanded


