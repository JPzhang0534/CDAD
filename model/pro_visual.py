import torch
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans

# ---------- Sinkhorn (pure torch, log-domain stabilized) ----------
def sinkhorn_logspace(log_alpha, n_iters=50, eps=1e-6):
    """
    Simplified log-domain Sinkhorn normalization to turn log_alpha into a
    doubly-stochastic matrix (rows sum to 1 and cols sum to 1).
    Args:
        log_alpha: torch.Tensor [N, M] (log of affinity / unnormalized scores)
        n_iters: number of Sinkhorn iterations
    Returns:
        P: torch.Tensor [N, M] approximately doubly-stochastic
    """
    # log-domain Sinkhorn: alternate row/col normalization in log-space
    u = torch.zeros_like(log_alpha[:, :1])  # [N,1]
    v = torch.zeros_like(log_alpha[:1, :])  # [1,M]
    K = log_alpha  # use log-alpha directly
    for _ in range(n_iters):
        # row normalization: subtract logsumexp over cols
        u = -torch.logsumexp(K + v, dim=1, keepdim=True)
        # col normalization: subtract logsumexp over rows
        v = -torch.logsumexp(K + u, dim=0, keepdim=True)
    P = torch.exp(K + u + v)  # [N,M]
    # numeric safety
    P = P / (P.sum(dim=1, keepdim=True) + eps)
    P = P / (P.sum(dim=0, keepdim=True) + eps)
    return P


# ---------- 主函数：构建/更新 多模态视觉原型 ----------
def aggregate_visual_prototypes(
    self,
    few_shot_samples=None,
    test_image=None,
    text_prototype=None,
    device=None,
    mode="auto",
    method="topk",         # choices: "topk", "kmeans", "sinkhorn"
    K=5,                   # for kmeans centroids
    top_k=64,              # for top-k pooling
    sinkhorn_iters=40,
    use_patch_layer=3,     # which patch layer to use in self.decoder(...) (align with run_clip)
    normalize=True,
    update_cache=False,    # whether to append high-confidence samples to self.normal cache
    cache_update_threshold=0.9,
):
    """
    Build multimodal visual prototypes (p_v) given few-shot normals or zero-shot (text_prototype + test_image).
    - If few_shot_samples provided: aggregate their global CLS features -> p_v_global,
        and aggregate patch tokens -> p_v_patch (single vector or K centroids depending on method).
    - If no few_shot_samples (zero-shot): use text_prototype to pick top-k patches from test_image and aggregate.
    - method="sinkhorn" will do a simplified OT-like reweighting between text prompts and visual particles.
    Returns:
        dict {
            "p_v_global": tensor[D] or None,
            "p_v_patch": tensor[1,D] or [K,D],  # normalized
            "visual_particles": tensor[Np, D] (raw particles used)
        }
    Notes:
        - self.clip_surgery.encode_image(...) expected to return (image_feats, patch_features)
        - self.decoder(...) maps patch_features -> list/array of layer tokens; we pick use_patch_layer
    """

    if device is None:
        device = next(self.clip_surgery.parameters()).device if hasattr(self.clip_surgery, "parameters") else torch.device("cuda")

    results = {"p_v_global": None, "p_v_patch": None, "visual_particles": None}

    # ------------------ few-shot mode ------------------
    if few_shot_samples is not None and len(few_shot_samples) > 0:
        # preprocess and encode few-shot normals
        imgs = self.transform_clip(few_shot_samples).to(device)  # assume transform_clip returns a tensor batch
        with torch.no_grad():
            image_feats, patch_feats = self.clip_surgery.encode_image(imgs)  # image_feats: [B, 1, D] or [B, L, D]
        # global proto: mean CLS token
        global_feats = image_feats[:, 0, :]  # [B, D]
        p_v_global = global_feats.mean(dim=0)
        if normalize:
            p_v_global = F.normalize(p_v_global, dim=-1)
        results["p_v_global"] = p_v_global.detach()

        # patch tokens: decode and select layer
        # if self.decoder returns list-like per-layer tokens
        patch_tokens_all = self.decoder(patch_feats)  # expected list or tensor-like
        # ensure indexing compatibility
        patch_layer_tokens = patch_tokens_all[use_patch_layer]  # [B, L, C]
        B, L, C = patch_layer_tokens.shape
        # project to CLIP image embedding dim if needed (check presence of visual.proj)
        if hasattr(self.clip_surgery.visual, "proj"):
            proj = self.clip_surgery.visual.proj.to(device)
            tokens_proj = (patch_layer_tokens @ proj)  # [B, L, D]
        else:
            tokens_proj = patch_layer_tokens
        # normalize tokens
        tokens_proj = F.normalize(tokens_proj, dim=-1)

        # flatten particles: each patch is a particle
        particles = tokens_proj.reshape(-1, tokens_proj.shape[-1])  # [B*L, D]
        results["visual_particles"] = particles.detach()

        # aggregation methods
        if method == "topk":
            # compute particle scores wrt text_prototype if provided, else mean similarity to global_feats
            if text_prototype is not None:
                # text_prototype shape [D] or [M,D] -> average
                if text_prototype.dim() == 1:
                    tvec = text_prototype.unsqueeze(0)  # [1,D]
                else:
                    tvec = text_prototype.mean(dim=0, keepdim=True)  # [1,D]
                sims = (particles @ tvec.T).squeeze(-1)  # [Np]
            else:
                sims = (particles @ p_v_global.unsqueeze(-1)).squeeze(-1)
            # pick top_k particles (global across batch)
            topk_vals, topk_idx = torch.topk(sims, k=min(top_k, particles.shape[0]), dim=0)
            selected = particles[topk_idx]  # [k, D]
            p_v_patch = selected.mean(dim=0)
            if normalize:
                p_v_patch = F.normalize(p_v_patch, dim=-1)
            results["p_v_patch"] = p_v_patch.detach()

        elif method == "kmeans":
            # KMeans on CPU (sklearn)
            flat = particles.cpu().numpy()
            k = min(K, flat.shape[0])
            kmeans = KMeans(n_clusters=k, random_state=0).fit(flat)
            centroids = torch.tensor(kmeans.cluster_centers_, device=device, dtype=torch.float32)
            if normalize:
                centroids = F.normalize(centroids, dim=-1)
            results["p_v_patch"] = centroids.detach()  # [K, D]

        elif method == "sinkhorn":
            # build text vectors matrix (if provided) or use global prototype as single text vector
            if text_prototype is None:
                # fallback to global proto as single text anchor
                text_mat = p_v_global.unsqueeze(0)  # [1, D]
            else:
                if text_prototype.dim() == 1:
                    text_mat = text_prototype.unsqueeze(0)
                else:
                    text_mat = text_prototype  # [M, D]
            # compute affinity log-space matrix between particles (N) and text anchors (M)
            # Using cosine similarities scaled by temperature
            temp = 0.07
            A = (particles @ text_mat.T) / temp  # [Np, M]
            logA = A  # we are working in log-space approx (cos sim as logits)
            P = sinkhorn_logspace(logA, n_iters=sinkhorn_iters)  # [Np, M], soft assignment
            # compute weighted particle score by summing over text anchors
            weights = P.sum(dim=1)  # [Np]
            # select top-k weighted particles
            topk_vals, topk_idx = torch.topk(weights, k=min(top_k, weights.shape[0]), dim=0)
            selected = particles[topk_idx]
            # weighted centroid
            sel_weights = weights[topk_idx].unsqueeze(-1)
            p_cent = (selected * sel_weights).sum(dim=0) / (sel_weights.sum() + 1e-8)
            if normalize:
                p_cent = F.normalize(p_cent, dim=-1)
            results["p_v_patch"] = p_cent.detach()

        else:
            raise ValueError(f"Unknown method {method}")

        # optionally update cache: add global_feats to self.normal_image_features / self.normal_patch_tokens
        if update_cache:
            try:
                # append to stored normal feats if exist
                if not hasattr(self, "normal_image_features") or self.normal_image_features is None:
                    self.normal_image_features = global_feats.detach().cpu()
                else:
                    self.normal_image_features = torch.cat([self.normal_image_features, global_feats.detach().cpu()], dim=0)
                if not hasattr(self, "normal_patch_tokens") or self.normal_patch_tokens is None:
                    self.normal_patch_tokens = tokens_proj.detach().cpu()
                else:
                    self.normal_patch_tokens = torch.cat([self.normal_patch_tokens.cpu(), tokens_proj.detach().cpu()], dim=0)
            except Exception:
                pass

        return results

    # ------------------ zero-shot mode (no few_shot_samples) ------------------
    else:
        # require test_image and text_prototype
        assert test_image is not None and text_prototype is not None, "Zero-shot needs test_image and text_prototype"
        test_img_tensor = self.transform_clip(test_image).unsqueeze(0).to(device)  # [1,3,H,W]
        with torch.no_grad():
            img_feats, patch_feats = self.clip_surgery.encode_image(test_img_tensor)
        patch_tokens_all = self.decoder(patch_feats)[use_patch_layer]  # [1,L,C]
        if hasattr(self.clip_surgery.visual, "proj"):
            proj = self.clip_surgery.visual.proj.to(device)
            tokens_proj = (patch_tokens_all @ proj)
        else:
            tokens_proj = patch_tokens_all
        tokens_proj = F.normalize(tokens_proj, dim=-1)  # [1,L,D]
        particles = tokens_proj.squeeze(0)  # [L, D]
        results["visual_particles"] = particles.detach()

        # compute sims to text_prototype (if multi-prompt average)
        if text_prototype.dim() == 1:
            tvec = text_prototype.unsqueeze(0)
        else:
            tvec = text_prototype.mean(dim=0, keepdim=True)
        sims = (particles @ tvec.T).squeeze(-1)  # [L]
        # pick top_k patches
        k = min(top_k, sims.shape[0])
        topk_vals, topk_idx = torch.topk(sims, k=k, dim=0)
        selected = particles[topk_idx]  # [k, D]
        p_v_patch = selected.mean(dim=0)
        if normalize:
            p_v_patch = F.normalize(p_v_patch, dim=-1)
        results["p_v_patch"] = p_v_patch.detach()
        # global prototype fallback: use image global or text prototype if needed
        global_feat = img_feats[:, 0, :].squeeze(0)
        if normalize:
            global_feat = F.normalize(global_feat, dim=-1)
        results["p_v_global"] = global_feat.detach()

        return results
