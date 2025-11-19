import torch
import torch.nn.functional as F


# ========== 1. 语言原型构造 ==========
def build_text_prototype(prompts, clip_model, device):
    """
    输入:
        prompts: list[str], 多个文本描述 (normal / anomaly / etc.)
        clip_model: 已加载的 CLIP 模型
        device: 运行设备
    输出:
        p_t: 归一化后的语言原型 (D,)
    步骤:
        1. 将 prompt tokenize 成 CLIP 输入
        2. 编码为 text embedding
        3. 对所有 prompt embedding 求均值
        4. L2 归一化得到原型
    """
    with torch.no_grad():
        tokens = clip_model.tokenize(prompts).to(device)
        text_feats = clip_model.encode_text(tokens)  # [N, D]
    p_t = text_feats.mean(dim=0)
    p_t = F.normalize(p_t, dim=-1)  # L2 归一化
    return p_t


# ========== 2. 视觉原型构造 ==========
def build_visual_prototype(images, clip_model, preprocess, device, mode="global", K=None):
    """
    输入:
        images: list of PIL.Image 或 tensor，normal 样本
        clip_model: 已加载的 CLIP 模型 (含 visual.proj)
        preprocess: CLIP 对图像的预处理函数
        device: 运行设备
        mode: "global" 表示全局原型 (CLS token)，
              "patch" 表示基于 patch token 的原型
        K: 若使用 patch 模式，可选的 KMeans 簇数

    输出:
        p_v: 归一化后的视觉原型 (D,) 或多个原型 [K,D]
    步骤:
        global:
            1. 编码图像得到全局特征 (CLS token)
            2. 求均值并归一化
        patch:
            1. 提取 patch token
            2. 映射到与文本同一空间 (visual.proj)
            3. 聚合 (均值 or KMeans 得到 K 个原型)
            4. 归一化
    """
    # 预处理 -> [B,3,H,W]
    X = torch.stack([preprocess(img) for img in images]).to(device)

    with torch.no_grad():
        image_feats, patch_tokens = clip_model.encode_image(X, return_patch_tokens=True)
        # image_feats: [B,1,D], patch_tokens: dict[layer -> [B,L,C]]

    if mode == "global":
        # 取 CLS token
        global_feats = image_feats[:, 0, :]  # [B,D]
        p_v = global_feats.mean(dim=0)
        p_v = F.normalize(p_v, dim=-1)
        return p_v

    elif mode == "patch":
        # 假设用 layer=3 的 patch token (可调)
        tokens = patch_tokens[3]  # [B,L,C]
        proj = clip_model.visual.proj  # [C,D]
        tokens_proj = tokens @ proj  # [B,L,D]
        tokens_proj = F.normalize(tokens_proj, dim=-1)

        if K is None:  # 简单均值
            p_v = tokens_proj.mean(dim=(0, 1))  # 所有 patch 求均值
            p_v = F.normalize(p_v, dim=-1)
            return p_v
        else:
            # 使用 KMeans 提取 K 个 patch 原型
            from sklearn.cluster import KMeans
            flat = tokens_proj.reshape(-1, tokens_proj.shape[-1]).cpu().numpy()
            kmeans = KMeans(n_clusters=K).fit(flat)
            centroids = torch.tensor(kmeans.cluster_centers_, device=device, dtype=torch.float32)
            centroids = F.normalize(centroids, dim=-1)
            return centroids  # [K,D]
