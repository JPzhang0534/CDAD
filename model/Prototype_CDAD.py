import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DDIMScheduler

from transformers import BlipProcessor, BlipForConditionalGeneration,CLIPImageProcessor, CLIPTextModel, CLIPVisionModel

from Diffusion_CLIP import (encode_imgs,split_text,split_sentences,AttentionStore, aggregate_all_attention_batch,aggregate_all_attention_batch_imagesize,
                            sd_to_clip_img,run_and_display,aggregate_all_attention,Gen_bbox_single,Gen_bbox_all,heatmap_img)

from segment_anything import sam_model_registry, SamPredictor

from model.attentionmap_save import save_attention_map, save_batch_attention_maps, overlay_attention_on_image,save_attention_map_gradcam,save_batch_attention_gradcam

import cv2
import numpy as np

from PIL import Image

import clip
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

import torch
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans
# from kornia.filters import gaussian_blur2d



from utils.prototypead import select_anomaly_prototypes,PrototypeMemoryBank




BICUBIC = InterpolationMode.BICUBIC

MY_TOKEN = None  # ‘’
LOW_RESOURCE = False
NUM_DDIM_STEPS = 1
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

from clip_prompt import encode_text_with_prompt_ensemble



def fuse_results(results_diff, results_clip):

    results = {}

    # ---- pred_score: clip 为主，但根据差异动态修正 ----
    score_diff = results_diff["pred_score"]
    score_clip = results_clip["pred_score"]

    d = abs(score_clip - score_diff)  # 两者差异
    w_clip = 1 / (1 + d.item() + 1e-6)
    w_diff = 1 - w_clip

    results["pred_score"] = (
            w_diff * score_diff + w_clip * score_clip.cpu()
    )

    # ---- pred_mask: diff 为主，但根据平均强度修正 ----
    mask_diff = results_diff["pred_mask"].float()
    mask_clip = results_clip["pred_mask"].float()

    mean_diff = torch.mean(mask_diff)
    mean_clip = torch.mean(mask_clip)

    w_diff = mean_diff / (mean_diff + mean_clip + 1e-6)
    w_clip = 1 - w_diff

    results["pred_mask"] = (
            w_diff * mask_diff + w_clip * mask_clip.cpu()
    )


    return results




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



def find_token_index_by_word(tokens, target_word):
    """
    在 tokenizer 的 token 序列中模糊匹配目标词
    如匹配 'dog' 可找到 'dog</w>'
    """
    for idx, tok in enumerate(tokens):
        cleaned = tok.replace("</w>", "").lstrip("Ġ")
        if cleaned.lower() == target_word.lower():
            return idx
    raise ValueError(f"Word '{target_word}' not found in token list: {tokens}")


class LinearLayer(nn.Module):
    def __init__(self):
        super(LinearLayer, self).__init__()

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = tokens[i][:, 1:, :]
            else:
                assert 0 == 1
        return tokens


class CLIP_Diffusion(nn.Module):
    def __init__(self,image_size,SD_path='/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/stable_diffusion',
                 clip_path= '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP-vit-large-14',
                 blip_path='/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/BLIP/blip-image-captioning-large',
                 device='cuda:2',):
        # 加载CLIP视觉model
        super().__init__()

        self.device = device


        # SD_path = '/home/data/zjp/pre-trained_model/stable_diffusion_v1_5'    # v1-5
        # SD_path = '/home/data/zjp/pre-trained_model/stable-diffusion-inpainting'  # inpainting

        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            SD_path, safety_checker=None).to(device)     #  32 v1-5   wood 99.6 |       94.3 |

        self.scheduler = DDIMScheduler.from_pretrained(
            SD_path, subfolder="scheduler")
         #------------------  加速 -------------------------------------
        self.ldm_stable.scheduler.set_timesteps(20)

        self.vae = self.ldm_stable.vae.to(device)
        self.tokenizer = self.ldm_stable.tokenizer

        self.clip_surgery, self.clip_preprocess = clip.load("CS-ViT-L/14", device=device)
        self.clip_surgery.eval()

        # self.clip_surgery, self.clip_preprocess = clip.load("/home/zjp/.cache/clip/CS-ViT-H-14.pt", device=device)
        # self.clip_surgery.eval()

        # 加载blip
        # self.blip_processor = BlipProcessor.from_pretrained(blip_path)
        # self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_path).to(device)


        # ---todo： 添加CLIP surgery相关分支

        #
        # self.clip_surgery, self.clip_preprocess = clip.load("CS-ViT-B/16", device=device)


        # from transformers import CLIPTokenizer
        # tokenizer_path = "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP-vit-large-14"
        # self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

        # tokenizer_clip = open_clip.get_tokenizer('ViT-B-32')
        self.tokenizer_clip = open_clip.get_tokenizer('ViT-L-14')

        with (torch.no_grad()):
            # self.text_prompts = clip.encode_text_with_prompt_ensemble_ad(
            #     self.clip_surgery, None, ["object"], self.tokenizer_clip, device
            # )

            self.text_prompts,self.anomaly_text_proto,self.norm_text_proto = clip.encode_text_with_prompt_ensemble_ad(
                self.clip_surgery, None, ["object"], self.tokenizer_clip, device
            )  # self.anomaly_text_proto [7,768]




            # self.text_prompts = clip.encode_text_with_prompt_ensemble_ad(
            #    self.ldm_stable,self.clip_surgery, ["object"], self.tokenizer, device
            # )

            # self.text_prompts = clip.encode_text_with_prompt_ensemble_ad_H(
            #    self.clip_surgery, ["object"], tokenizer_clip, device
            # )

            # self.text_prompts = clip.encode_text_with_prompt_ensemble_ad_H(
            #     self.ldm_stable.text_encoder.to(self.device), ["object"], self.tokenizer, device
            # )

            # self.text_prompts_diff = clip.encode_text_with_prompt_ensemble_ad_diff(
            #      self.ldm_stable.text_encoder.to(self.device), ["object"], self.tokenizer, self.device)


            self.text_prompts_diff, self.state_id, self.object_id, self.norm_id = clip.encode_text_with_prompt_ensemble_ad_diff_word(
                self.ldm_stable.text_encoder.to(self.device), ["object"], self.tokenizer, self.device)

        self.text_feature_prototype = self.text_prompts["object"].to(self.device)

        from torchvision.transforms import v2
        # self.transform_clip = v2.Compose(
        #     [
        #         v2.Resize((image_size, image_size)),
        #         v2.Normalize(
        #             mean=(0.48145466, 0.4578275, 0.40821073),
        #             std=(0.26862954, 0.26130258, 0.27577711),
        #         ),
        #     ],
        # )

        img_clip = 224
        self.transform_clip = v2.Compose(
            [
                v2.Resize((img_clip, img_clip)),
                v2.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )

        self.image_size = image_size
        self.shot = 0

        #----------
        #
        # sam_checkpoint = "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/sam/sam_vit_h_4b8939.pth"
        # model_type = "vit_h"
        # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.to(device=device)
        # self.sam_predictor = SamPredictor(sam)


        self.decoder = LinearLayer()

        self.p_v_global = None
        self.p_v_patch = None

        self.norm_anomaly = None
        self.globalscore = None

        self.selected_feats = []


    def visual_text_similarity(self, image_feature, patch_token, text_feature, aggregation):
        anomaly_maps = []

        for layer in range(len(patch_token)):
            anomaly_map = (100.0 * patch_token[layer][:, 1:, :] @ text_feature)
            anomaly_maps.append(anomaly_map)

        use_hsf = 0
        if  use_hsf:
            alpha = 0.2
            clustered_feature = self.HSF.forward(patch_token, anomaly_maps)
            # aggregate the class token and the clustered features for more comprehensive information
            cur_image_feature = alpha * clustered_feature + (1 - alpha) * image_feature
            cur_image_feature = F.normalize(cur_image_feature, dim=1)
        else:
            cur_image_feature = image_feature

        # score 方法二
        anomaly_score = (100.0 * cur_image_feature[:, 1:, :] @ text_feature)
        # anomaly_score = anomaly_score.squeeze(1)
        anomaly_score = torch.softmax(anomaly_score, dim=1)


        # score 方法三
        # anomaly_score = (100.0 * cur_image_feature.unsqueeze(1) @ text_feature)
        # anomaly_score = anomaly_score.squeeze(1)
        # anomaly_score = torch.softmax(anomaly_score, dim=1)

        # NOTE: this bilinear interpolation is not unreproducible and may occasionally lead to unstable ZSAD performance.
        for i in range(len(anomaly_maps)):
            B, L, C = anomaly_maps[i].shape
            H = int(np.sqrt(L))
            anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
            anomaly_maps[i] = F.interpolate(anomaly_maps[i], size=self.image_size, mode='bilinear', align_corners=True)


        if aggregation: # in the test stage, we firstly aggregate logits from all hierarchies and then do the softmax normalization
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_map = (anomaly_map[:, 1, :, :] + 1 - anomaly_map[:, 0, :, :]) / 2
            anomaly_score = anomaly_score[:, 1]
            return anomaly_map, anomaly_score
        else: # otherwise, we do the softmax normalization for individual hierarchies
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            return anomaly_maps, anomaly_score

        # anomaly_maps = []
        #
        # for layer in range(len(patch_token)):
        #     anomaly_map = (100.0 * patch_token[layer][:, 1:, :] @ text_feature)
        #     anomaly_maps.append(anomaly_map)
        # use_hsf = 0
        # if use_hsf:
        #     alpha = 0.2
        #     clustered_feature = self.HSF.forward(patch_token, anomaly_maps)
        #     # aggregate the class token and the clustered features for more comprehensive information
        #     cur_image_feature = alpha * clustered_feature + (1 - alpha) * image_feature
        #     cur_image_feature = F.normalize(cur_image_feature, dim=1)
        # else:
        #     cur_image_feature = image_feature
        #
        # anomaly_score = (100.0 * cur_image_feature.unsqueeze(1) @ text_feature)
        # anomaly_score = anomaly_score.squeeze(1)
        # anomaly_score = torch.softmax(anomaly_score, dim=1)
        #
        # # NOTE: this bilinear interpolation is not unreproducible and may occasionally lead to unstable ZSAD performance.
        # for i in range(len(anomaly_maps)):
        #     B, L, C = anomaly_maps[i].shape
        #     H = int(np.sqrt(L))
        #     anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
        #     anomaly_maps[i] = F.interpolate(anomaly_maps[i], size=self.image_size, mode='bilinear', align_corners=True)
        #
        # if aggregation:  # in the test stage, we firstly aggregate logits from all hierarchies and then do the softmax normalization
        #     anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
        #     anomaly_map = torch.softmax(anomaly_map, dim=1)
        #     anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
        #     anomaly_score = anomaly_score[:, 1]
        #     return anomaly_map, anomaly_score
        # else:  # otherwise, we do the softmax normalization for individual hierarchies
        #     for i in range(len(anomaly_maps)):
        #         anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
        #     return anomaly_maps, anomaly_score

    def run_clip(self,img_paths,cls_names):

        clip_transformed_image = self.transform_clip(img_paths)

        with torch.no_grad():

            image_features, patch_features = self.clip_surgery.encode_image(
                clip_transformed_image)


        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # with (torch.no_grad()):
        #     self.text_prompts = clip.encode_text_with_prompt_ensemble_ad(
        #         self.clip_surgery, cls_names, self.tokenizer_clip, self.device
        #     )
        #
        # self.text_feature_prototype = self.text_prompts[cls_names[0]].to(self.device)

        patch_tokens = self.decoder(patch_features)

        # k-shot 部分（支持多 batch）
        if self.shot:
            global_score = 1 - (image_features @ self.normal_image_features.transpose(-2, -1)).amax(dim=-1)
            sims = []
            for i in range(len(patch_tokens)):
                if i % 2 == 0:
                    continue

                # if i in [0,2,3,5] :   # vit-16
                #     continue

                # patch_tokens[i]: [B, L, C]
                pt = patch_tokens[i]
                B, L, C = pt.shape

                # 如果你的 token 里还包含 CLS（常见是 L = 1 + H*W），先去掉 CLS：
                # pt = pt[:, 1:, :]
                # B, L, C = pt.shape

                # 确保是连续内存再 view（有时上游 permute/linear 会导致非连续）
                pt = pt.contiguous().view(B, L, 1, C)  # [B, L, 1, C]

                # normal tokens: 形状通常为 [N_norm, C] 或 [?, C]
                # 原代码是 reshape(1, -1, 1024) —— 这里保持 C 一致即可：
                nt = self.normal_patch_tokens[i].reshape(1, -1, C).unsqueeze(0)  # [1, 1, N_norm, C]

                # 广播到 [B, L, N_norm, C] 后按最后一维做余弦相似度
                cosine_similarity_matrix = F.cosine_similarity(pt, nt, dim=-1)  # [B, L, N_norm]

                # 对 normal 集合取最大相似度：得到每个 patch 的最相近 normal 相似度
                sim_max, _ = torch.max(cosine_similarity_matrix, dim=2)  # [B, L]
                sims.append(sim_max)

            # 跨所选层求均值
            sim = torch.mean(torch.stack(sims, dim=0), dim=0)  # [B, L]

            # 把 [B, L] 还原到特征图网格
            H = int(np.sqrt(L))
            assert H * H == L, f"L={L} 不是完全平方（可能还包含 CLS token？）"
            sim = sim.view(B, 1, H, H)  # [B, 1, H, H]

            # 上采样到图像尺寸
            sim = F.interpolate(sim, size=self.image_size, mode="bilinear", align_corners=True)
            anomaly_map_ret = 1 - sim
        # k-shot 部分


        anomaly_map_vls = []
        for layer in range(len(patch_tokens)):

            if layer != 4: # layer%2!=0:# (layer+1)//2!=0:   # layer != 3:   # 6 12 18 24 24_xor  选择输出的layer
                continue   #  vitl-14

            # if layer != 4:   # 3 6 9 9_xor 12 12_xor  vit-16
            #     continue

            # if layer % 2 == 0:
            #     continue

            patch_tokens[layer] = patch_tokens[layer] @ self.clip_surgery.visual.proj
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(
                dim=-1, keepdim=True
            )
            anomaly_map_vl = 100.0 * patch_tokens[layer] @ self.text_feature_prototype
            B, L, C = anomaly_map_vl.shape
            H = int(np.sqrt(L))
            anomaly_map_vl = F.interpolate(
                anomaly_map_vl.permute(0, 2, 1).view(B, 2, H, H),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map_vl = torch.softmax(anomaly_map_vl, dim=1)
            anomaly_map_vl = (  anomaly_map_vl[:, 1, :, :] - anomaly_map_vl[:, 0, :, :] + 1
                             ) / 2
            anomaly_map_vls.append(anomaly_map_vl)
        anomaly_map_vls = torch.mean(
            torch.stack(anomaly_map_vls, dim=0), dim=0
        ).unsqueeze(1)


        #k-shot
        if self.shot:
            anomaly_map_ret_all = (anomaly_map_ret + anomaly_map_vls
                ) / 2

            pred_score = anomaly_map_ret_all.view(B, -1).max(dim=1).values + global_score
            # k-shot
        else:

            anomaly_map_ret_all =  anomaly_map_vls
            pred_score = anomaly_map_ret_all.view(B, -1).max(dim=1).values

        try:
            del clip_transformed_image, image_features, patch_features, img_paths, _, global_score
            del patch_tokens, nt, pt, sims, sim
            del anomaly_map_ret, anomaly_map_vl, anomaly_map_vls
            del cosine_similarity_matrix, sim_max
        except NameError:
            pass

        # --- 清理 GPU 缓存 ---
        torch.cuda.empty_cache()

        return {
            "pred_score": pred_score.detach().cpu(),
            "pred_mask": anomaly_map_ret_all.detach().cpu(),
        }


        # return {
        #     "pred_score": torch.tensor(anomaly_map_vls.max().item())+ global_score,
        #     "pred_mask": anomaly_map_vls,
        # }


    def run_diffusion_prompts_batchs(self, cls_names, img_tensors, mask_clip=None):

        # states = ["damage"]
        states = ["crack", "hole", "residue", "damage"]

        # prompts: list[list[str]]，每个样本一个 list
        prompts = [[f"A photo of a {cls_name} with {state}" for state in states] for cls_name in cls_names]

        rgb_512 = img_tensors.to(self.device)  # (B, 3, H, W)
        input_latent = encode_imgs(rgb_512, self.vae)  # (B, 4, 64, 64)


        # noise 设计
        # mask_clip = mask_clip.repeat(1,3,1,1).to(self.device)
        # mask_latent = encode_imgs(mask_clip, self.vae)

        controller = AttentionStore()
        controller.step_store = controller.get_empty_store()
        controller.attention_store = {}

        times = [1,]  # <<< 修改这里：你要跑的 step
        all_attn_finals = []  # <<< 修改这里：存放不同 t 的结果

        dtype = torch.float32

        for t in times:  # <<< 修改这里：循环不同的 t
            controller.reset()
            zero_noise = torch.zeros_like(input_latent)
            latents_noisy = self.ldm_stable.scheduler.add_noise(
                input_latent, zero_noise, torch.tensor(t, device=self.device)
            )

            # clip_noise = mask_latent
            # latents_noisy = self.ldm_stable.scheduler.add_noise(
            #     input_latent, clip_noise, torch.tensor(t, device=self.device)
            # )


            # noise = torch.randn([1, 4, 32, 32]).to(self.device)  # noise 随机初始化
            # noise = noise if dtype == torch.float32 else noise.half()
            # latents_noisy = self.ldm_stable.scheduler.add_noise(input_latent, noise,
            #                                                     torch.tensor(t, device=self.device))
            # latents_noisy = latents_noisy if dtype == torch.float32 else latents_noisy.half()

            with torch.no_grad():
                image_inv, x_t = run_and_display(
                    [p[0] for p in prompts],
                    controller,
                    ldm_stable=self.ldm_stable,
                    run_baseline=False,
                    latent=latents_noisy,
                    verbose=False,
                    file_name=None,
                    clip_image=None,
                    text_prompt=self.text_prompts_diff,
                    onlyimg=False,
                    image_size=self.image_size
                )
            # del rgb_512, input_latent, latents_noisy, image_inv, x_t, img_tensors,mask_latent,mask_clip,clip_noise

            # del rgb_512, input_latent, latents_noisy, image_inv, x_t, img_tensors,mask_clip

            torch.cuda.empty_cache()
            # ============= attention 提取部分 =============
            cross_attention_maps, resolutions = aggregate_all_attention_batch_imagesize(
                prompts, controller, ("up", "mid", "down"), True, image_size=self.image_size
            )
            self_attention_maps, _ = aggregate_all_attention_batch_imagesize(
                prompts, controller, ("up", "mid", "down"), False, image_size=self.image_size
            )

            del _
            torch.cuda.empty_cache()

            B = len(prompts)
            latent_res = self.image_size // 8

            if latent_res == 64:
                weight = [0.3, 0.5, 0.1, 0.1]
            elif latent_res == 32:
                weight = [0.2, 0.4, 0.3, 0.1]
                # weight = [0.3, 0.5, 0.1, 0.1]
            else:
                weight = [1.0 / len(resolutions)] * len(resolutions)

            batch_attn_maps = []

            for b in range(B):
                out_atts = []
                word_len = len(prompts[b][0].split(" "))
                embs_len = len(self.tokenizer.encode(prompts[b][0])) - 2

                for l, res in enumerate(resolutions):
                    cross_att = cross_attention_maps[l][b]

                    #  这部分是选择文本token 和cross attention 间对应
                    try:
                        if prompts[b][0].split(" ")[3 + word_len].endswith("ing"):
                            token_ids = [3 + embs_len, 5 + embs_len]
                        else:
                            token_ids = [3 + embs_len]
                    except:
                        token_ids = [3 + embs_len]

                    cross_att = cross_att[:, :, token_ids].mean(-1)

                    if res != latent_res:
                        cross_att = F.interpolate(
                            cross_att.unsqueeze(0).unsqueeze(0),
                            size=(latent_res, latent_res),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze()

                    cross_att = (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min() + 1e-8)
                    out_atts.append(cross_att * weight[l])

                cross_att_map = torch.stack(out_atts).sum(0).view(latent_res * latent_res, 1)

                self_att = self_attention_maps[-1][b].detach()
                self_att = self_att.view(latent_res * latent_res, latent_res * latent_res)

                att_map = torch.matmul(self_att, cross_att_map).view(latent_res, latent_res)
                batch_attn_maps.append(att_map)

            attn_final = torch.stack(batch_attn_maps, dim=0)  # (B, H, W)
            attn_final = attn_final.unsqueeze(1)  # (B,1,H,W)
            attn_final = F.interpolate(
                attn_final, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )

            all_attn_finals.append(attn_final)  # <<< 修改这里：保存每个 t 的结果

        # ========= 融合不同 t 的结果 =========
        attn_final_fused = torch.mean(torch.stack(all_attn_finals, dim=0), dim=0)  # <<< 修改这里：结果融合

        pred_score = attn_final_fused.view(B, -1).max(dim=1).values

        results_diff = {
            "pred_score": pred_score.detach().cpu(),
            "pred_mask": attn_final_fused.detach().cpu(),
        }


        del attn_final, cross_attention_maps, self_attention_maps,batch_attn_maps,pred_score
        torch.cuda.empty_cache()

        return results_diff

    # t= 500
    # # latents_noisy = input_latent
    # # for t in times:
    # #     controller.reset()
    # zero_noise = torch.zeros_like(input_latent)
    # latents_noisy = self.ldm_stable.scheduler.add_noise(
    #     input_latent, zero_noise, torch.tensor(t, device=self.device)
    # )
    #
    # with torch.no_grad():
    #     image_inv, x_t = run_and_display(
    #         [p[0] for p in prompts],
    #         controller,
    #         ldm_stable=self.ldm_stable,
    #         run_baseline=False,
    #         latent=latents_noisy,
    #         verbose=False,
    #         file_name=None,
    #         clip_image=None,
    #         text_prompt=text_prompt,
    #         onlyimg=False,
    #         image_size = self.image_size
    #     )
    # # del rgb_512, input_latent,latents_noisy,image_inv,x_t,img_tensors,text_prompt
    # # torch.cuda.empty_cache()
    #
    # # cross_attention_maps = aggregate_all_attention_batch([p[0] for p in prompts], controller, ("up", "mid", "down"), True,image_size=self.image_size,
    # #                                                )
    # # self_attention_maps = aggregate_all_attention_batch([p[0] for p in prompts], controller, ("up", "mid", "down"), False,image_size=self.image_size,
    # #                                               )
    #
    # cross_attention_maps, resolutions = aggregate_all_attention_batch_imagesize(
    #     prompts, controller, ("up", "mid", "down"), True, image_size=self.image_size
    # )
    # self_attention_maps, _ = aggregate_all_attention_batch_imagesize(
    #     prompts, controller, ("up", "mid", "down"), False, image_size=self.image_size
    # )
    #
    # B = len(prompts)
    #
    # # cross_attention_maps 已经返回 resolutions
    # # 例子：512 -> [8,16,32,64]，256 -> [4,8,16,32]
    # latent_res = self.image_size // 8
    #
    # # 动态权重，可以按需要调整
    # if latent_res == 64:
    #     weight = [0.3, 0.5, 0.1, 0.1]
    # elif latent_res == 32:
    #     weight = [0.2, 0.4, 0.3, 0.1]  # 你可以自己改权重策略
    # else:
    #     weight = [1.0 / len(resolutions)] * len(resolutions)  # fallback 均分
    #
    # batch_attn_maps = []
    #
    # for b in range(B):
    #     out_atts = []
    #     word_len = len(prompts[b][0].split(" "))
    #     embs_len = len(self.tokenizer.encode(prompts[b][0])) - 2
    #
    #     for l, res in enumerate(resolutions):
    #         # cross_attention_maps[l]: (B, H, W, D)
    #         cross_att = cross_attention_maps[l][b]  # (H, W, D)
    #
    #         try:
    #             if prompts[b][0].split(" ")[3 + word_len].endswith("ing"):
    #                 token_ids = [3 + embs_len, 5 + embs_len]
    #             else:
    #                 token_ids = [3 + embs_len]
    #         except:
    #             token_ids = [3 + embs_len]
    #
    #         cross_att = cross_att[:, :, token_ids].mean(-1)  # (H, W)
    #
    #         # 上采样到 latent_res×latent_res，而不是写死 64×64
    #         if res != latent_res:
    #             cross_att = F.interpolate(
    #                 cross_att.unsqueeze(0).unsqueeze(0),
    #                 size=(latent_res, latent_res),
    #                 mode="bilinear",
    #                 align_corners=False
    #             ).squeeze()
    #
    #         # 归一化
    #         cross_att = (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min() + 1e-8)
    #
    #         # 加权
    #         out_atts.append(cross_att * weight[l])
    #
    #     # cross map 融合
    #     cross_att_map = torch.stack(out_atts).sum(0).view(latent_res * latent_res, 1)
    #
    #     # self-attn 只用最大分辨率
    #     self_att = self_attention_maps[-1][b].detach()  # (latent_res, latent_res, N)
    #     self_att = self_att.view(latent_res * latent_res, latent_res * latent_res)
    #
    #     att_map = torch.matmul(self_att, cross_att_map).view(latent_res, latent_res)
    #
    #     batch_attn_maps.append(att_map)
    #
    # # (B, H, W)
    # attn_final = torch.stack(batch_attn_maps, dim=0)
    #
    # attn_final = attn_final.unsqueeze(1)  # (B, 1, latent_res, latent_res)
    # attn_final = F.interpolate(
    #     attn_final, size=(self.image_size, self.image_size),
    #     mode='bilinear', align_corners=False
    # )
    #
    # pred_score = attn_final.view(B, -1).max(dim=1).values

    # image_clip  --> [b,77,1024] 输入到u-net control
    def run_diffusion_clip(self,cls_names,img_tensors):

        clip_transformed_image = self.transform_clip(img_tensors)

        with torch.no_grad():

            image_features, patch_features = self.clip_surgery.encode_image(
                clip_transformed_image)

        text_features = self.text_prompts["object"].to(self.device)

        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        patch_tokens = self.decoder(patch_features)

        # k-shot 部分（支持多 batch）
        if self.shot:
            global_score = 1 - (image_features @ self.normal_image_features.transpose(-2, -1)).amax(dim=-1)
            sims = []
            for i in range(len(patch_tokens)):
                if i % 2 == 0:
                    continue

                # patch_tokens[i]: [B, L, C]
                pt = patch_tokens[i]
                B, L, C = pt.shape

                # 如果你的 token 里还包含 CLS（常见是 L = 1 + H*W），先去掉 CLS：
                # pt = pt[:, 1:, :]
                # B, L, C = pt.shape

                # 确保是连续内存再 view（有时上游 permute/linear 会导致非连续）
                pt = pt.contiguous().view(B, L, 1, C)  # [B, L, 1, C]

                # normal tokens: 形状通常为 [N_norm, C] 或 [?, C]
                # 原代码是 reshape(1, -1, 1024) —— 这里保持 C 一致即可：
                nt = self.normal_patch_tokens[i].reshape(1, -1, C).unsqueeze(0)  # [1, 1, N_norm, C]

                # 广播到 [B, L, N_norm, C] 后按最后一维做余弦相似度
                cosine_similarity_matrix = F.cosine_similarity(pt, nt, dim=-1)  # [B, L, N_norm]

                # 对 normal 集合取最大相似度：得到每个 patch 的最相近 normal 相似度
                sim_max, _ = torch.max(cosine_similarity_matrix, dim=2)  # [B, L]
                sims.append(sim_max)

            # 跨所选层求均值
            sim = torch.mean(torch.stack(sims, dim=0), dim=0)  # [B, L]

            # 把 [B, L] 还原到特征图网格
            H = int(np.sqrt(L))
            assert H * H == L, f"L={L} 不是完全平方（可能还包含 CLS token？）"
            sim = sim.view(B, 1, H, H)  # [B, 1, H, H]

            # 上采样到图像尺寸
            sim = F.interpolate(sim, size=self.image_size, mode="bilinear", align_corners=True)
            anomaly_map_ret = 1 - sim
        # k-shot 部分

        anomaly_map_vls = []
        for layer in range(len(patch_tokens)):

            if layer != 3:  # layer%2!=0:# (layer+1)//2!=0:
                continue

            patch_tokens[layer] = patch_tokens[layer] @ self.clip_surgery.visual.proj
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(
                dim=-1, keepdim=True
            )
            # anomaly_map_vl = 100.0 * patch_tokens[layer] @ text_features
            anomaly_map_vl = patch_tokens[layer] @ text_features

            B, L, C = anomaly_map_vl.shape
            H = int(np.sqrt(L))
            anomaly_map_vl = F.interpolate(
                anomaly_map_vl.permute(0, 2, 1).view(B, 2, H, H),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map_vl = torch.softmax(anomaly_map_vl, dim=1)
            anomaly_map_vl = (
                                     anomaly_map_vl[:, 1, :, :] - anomaly_map_vl[:, 0, :, :] + 1
                             ) / 2
            anomaly_map_vls.append(anomaly_map_vl)
        anomaly_map_vls = torch.mean(
            torch.stack(anomaly_map_vls, dim=0), dim=0
        ).unsqueeze(1)

        # k-shot
        if self.shot:
            anomaly_map_ret_all = (anomaly_map_ret + anomaly_map_vls
                                   ) / 2

            pred_score = anomaly_map_ret_all.view(B, -1).max(dim=1).values + global_score
            # k-shot
        else:

            anomaly_map_ret_all = anomaly_map_vls
            pred_score = anomaly_map_ret_all.view(B, -1).max(dim=1).values

        try:
            del clip_transformed_image, image_features, patch_features, text_features, _, global_score
            del patch_tokens, nt, pt, sims, sim
            del anomaly_map_ret, anomaly_map_vl, anomaly_map_vls
            del cosine_similarity_matrix, sim_max
        except NameError:
            pass

        torch.cuda.empty_cache()

        results_clip = {
            "pred_score": pred_score,
            "pred_mask": anomaly_map_ret_all,
        }
        # --- 清理 GPU 缓存 ---

        #-------------------------------- diffusion 部分 -------------------------------
        states = ["damage"]

        # prompts: list[list[str]]，每个样本一个 list
        prompts = [[f"A photo of a {cls_name} with {state}" for state in states] for cls_name in cls_names]



        rgb_512 = img_tensors.to(self.device)  # (B, 3, H, W)
        input_latent = encode_imgs(rgb_512, self.vae)  # (B, 4, 64, 64)

        # noise 设计
        mask_clip = anomaly_map_ret_all.repeat(1, 3, 1, 1).to(self.device)
        mask_latent = encode_imgs(mask_clip, self.vae)

        controller = AttentionStore()
        controller.step_store = controller.get_empty_store()
        controller.attention_store = {}

        times = [1]  # <<< 修改这里：你要跑的 step
        all_attn_finals = []  # <<< 修改这里：存放不同 t 的结果

        dtype = torch.float32

        for t in times:  # <<< 修改这里：循环不同的 t
            controller.reset()
            # zero_noise = torch.zeros_like(input_latent)
            # latents_noisy = self.ldm_stable.scheduler.add_noise(
            #     input_latent, zero_noise, torch.tensor(t, device=self.device)
            # )

            clip_noise = mask_latent
            latents_noisy = self.ldm_stable.scheduler.add_noise(
                input_latent, clip_noise, torch.tensor(t, device=self.device)
            )

            # noise = torch.randn([1, 4, 64, 64]).to(self.device)  # noise 随机初始化
            # noise = noise if dtype == torch.float32 else noise.half()
            # latents_noisy = self.ldm_stable.scheduler.add_noise(input_latent, noise,
            #                                                     torch.tensor(t, device=self.device))
            # latents_noisy = latents_noisy if dtype == torch.float32 else latents_noisy.half()

            with torch.no_grad():
                image_inv, x_t = run_and_display(
                    [p[0] for p in prompts],
                    controller,
                    ldm_stable=self.ldm_stable,
                    run_baseline=False,
                    latent=latents_noisy,
                    verbose=False,
                    file_name=None,
                    clip_image=None,
                    text_prompt=self.text_prompts_diff,
                    onlyimg=True,
                    image_size=self.image_size
                )

            # ============= attention 提取部分 =============
            cross_attention_maps, resolutions = aggregate_all_attention_batch_imagesize(
                prompts, controller, ("up", "mid", "down"), True, image_size=self.image_size
            )
            self_attention_maps, _ = aggregate_all_attention_batch_imagesize(
                prompts, controller, ("up", "mid", "down"), False, image_size=self.image_size
            )

            B = len(prompts)
            latent_res = self.image_size // 8

            if latent_res == 64:
                weight = [0.3, 0.5, 0.1, 0.1]
            elif latent_res == 32:
                weight = [0.2, 0.4, 0.3, 0.1]
            else:
                weight = [1.0 / len(resolutions)] * len(resolutions)

            batch_attn_maps = []

            for b in range(B):
                out_atts = []
                word_len = len(prompts[b][0].split(" "))
                embs_len = len(self.tokenizer.encode(prompts[b][0])) - 2

                for l, res in enumerate(resolutions):
                    cross_att = cross_attention_maps[l][b]

                    try:
                        if prompts[b][0].split(" ")[3 + word_len].endswith("ing"):
                            token_ids = [3 + embs_len, 5 + embs_len]
                        else:
                            token_ids = [3 + embs_len]
                    except:
                        token_ids = [3 + embs_len]

                    cross_att = cross_att[:, :, token_ids].mean(-1)

                    if res != latent_res:
                        cross_att = F.interpolate(
                            cross_att.unsqueeze(0).unsqueeze(0),
                            size=(latent_res, latent_res),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze()

                    cross_att = (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min() + 1e-8)
                    out_atts.append(cross_att * weight[l])

                cross_att_map = torch.stack(out_atts).sum(0).view(latent_res * latent_res, 1)

                self_att = self_attention_maps[-1][b].detach()
                self_att = self_att.view(latent_res * latent_res, latent_res * latent_res)

                att_map = torch.matmul(self_att, cross_att_map).view(latent_res, latent_res)
                batch_attn_maps.append(att_map)

            attn_final = torch.stack(batch_attn_maps, dim=0)  # (B, H, W)
            attn_final = attn_final.unsqueeze(1)  # (B,1,H,W)
            attn_final = F.interpolate(
                attn_final, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )

            all_attn_finals.append(attn_final)  # <<< 修改这里：保存每个 t 的结果

        # ========= 融合不同 t 的结果 =========
        attn_final_fused = torch.mean(torch.stack(all_attn_finals, dim=0), dim=0)  # <<< 修改这里：结果融合

        # pred_score = attn_final_fused.view(B, -1).quantile(0.90, dim=1)  # <<< 修改这里：分数计算保持和原来一致

        pred_score = attn_final_fused.view(B, -1).max(dim=1).values

        # pred_score = attn_final_fused.view(B, -1).to(torch.float32).quantile(0.95, dim=1).to(torch.float16)

        results_diff = {
            "pred_score": pred_score.detach().cpu(),
            "pred_mask": attn_final.detach().cpu(),
        }

        #-------------------------------- diffusion 部分 -------------------------------

        results = {
            key: (0.2 * results_diff[key] + 0.8 * results_clip[key].cpu()) if key == 'pred_score'
            else (0.7 * results_diff[key] + 0.3 * results_clip[key].cpu())
            for key in results_diff
        }


        return results



    def forward(self, images_tensor,img_data, text_data, cls_name, use_sam=False):

        results_clip = self.run_clip_multiproto(images_tensor,p_v_global= self.p_v_global,
        p_v_patch=self.p_v_patch, p_t=self.text_feature_prototype[:,0],cls_name=cls_name)

        #  ------------------------------ clip+diffusion ----------------------------------------------
        results_diff = self.run_diffusion_prompts_batchs_multiproto(cls_name, images_tensor, mask_clip=results_clip['pred_mask'])

        # results_diff = self.run_diffusion_prompts_batchs_multiproto(cls_name, images_tensor, )

        # results_diff = self.run_diffusion_prompts_batchs(cls_name, images_tensor)
        # #

        # results_diff = results_diff
        # # #
        #---------------------------  clip + diffusion ------------------------------
        # results = {
        #     key: (0.2 * results_diff[key] + 0.9 * results_clip[key].cpu()) if key == 'pred_score'    # auroc_sp   91.2
        #     else (0.7 * results_diff[key] + 0.3 * results_clip[key].cpu())                           # auroc_px    90.5
        #     for key in results_diff
        # }   #
        #---------------------------  clip + diffusion ------------------------------




        # results = fuse_results(results_diff,results_clip)  #  86.7 |       90.3 |
        results = {
            key: (0.1 * results_diff[key] + 0.9 * results_clip[key].cpu()) if key == 'pred_score'  # auroc_sp
            else (0.6 * results_diff[key] + 0.4 * results_clip[key].cpu())                        # auroc_px
            for key in results_diff
        }  #.

        # return results_clip
        return results

        #
        # results1 = {
        #     key: (0.2 * results_diff[key] + 0.8 * results_clip[key].cpu()) if key == 'pred_score'
        #     else (0.5 * results_diff[key] + 0.5 * results_clip[key].cpu())     #0.6   0.4      91   |       90   |
        #     for key in results_diff
        # }  #

        # del img_paths, text_data,results_diff,results_clip
        # torch.cuda.empty_cache()

        #
        # results = {
        #     key: (0.1 * results_diff[key] + 0.9 * results_clip[key]) if key == 'pred_score'    # 92.4
        #     else (0.6 * results_diff[key] + 0.4 * results_clip[key])                           #  90.7
        #     for key in results_diff
        # }

        # results_2 = {
        #     key: (0.2 * results_diff[key] + 0.8 * results_clip[key]) if key == 'pred_score'  # 90.3
        #     else (0.6 * results_diff[key] + 0.4 * results_clip[key])                         #  90.6
        #     for key in results_diff
        # }
        #
        # results_3 = {
        #     key: (0.15 * results_diff[key] + 0.85 * results_clip[key]) if key == 'pred_score'   # 88.5
        #     else (0.8 * results_diff[key] + 0.2 * results_clip[key])                          #  0.7 * results_diff[key] + 0.3 * results_clip[key])  90.5
        #     for key in results_diff
        # }


        # return  results, results1, results_3
                 # results_clip, results_diff,results1)

        #---------  CLIP 分支 ---------------------
        # img_path = img_paths[0]
        # text_data = text_data[0]
        # device = self.device
        # #
        # with Image.open(img_path) as pil_img:
        # # pil_img = Image.open(img_path)
        #
        #     if pil_img.mode != 'RGB':
        #         pil_img = pil_img.convert("RGB")
        #         pil_img = np.stack([pil_img] * 3, axis=-1)
        #
        #     cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        #     clip_preprocess = Compose([Resize((512, 512), interpolation=BICUBIC), ToTensor(),
        #                            Normalize((0.48145466, 0.4578275, 0.40821073),
        #                                      (0.26862954, 0.26130258, 0.27577711))])
        #     if use_sam == True:
        #         self.sam_predictor.set_image(np.array(pil_img))
        #
        #     # print(pil_img)
        #     image = clip_preprocess(pil_img).unsqueeze(0).to(device)
        # #
        #
        # with torch.no_grad():
        #     # CLIP architecture surgery acts on the image encoder
        #     image_features = self.clip_surgery.encode_image(image)
        #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #
        #     # Prompt ensemble for text features with normalization    text_data 应该是list
        #     # text_features = clip.encode_text_with_prompt_ensemble(self.clip_surgery, text_data, device)
        #     text_features = clip.encode_text_with_prompt_ensemble(self.clip_surgery, text_data.split(' '), device)
        #
        #     #
        #     # # Extract redundant features from an empty string
        #     redundant_features = clip.encode_text_with_prompt_ensemble(self.clip_surgery, [""], device)
        #
        #     # Apply feature surgery for single text
        #     # similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
        #
        #     # similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])
        #
        #     # print('------similarity_map----------------')
        #     # print(similarity_map.shape)
        #
        #     # Draw similarity map
        #     if use_sam:
        #         similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0]
        #         for n in range(similarity.shape[-1]):
        #
        #             points, labels = clip.similarity_map_to_points(similarity[1:, n], cv2_img.shape[:2], t=0.8)
        #             masks, scores, logits = self.sam_predictor.predict(point_labels=labels, point_coords=np.array(points),
        #                                                       multimask_output=True)
        #             mask = masks[np.argmax(scores)]
        #             mask = mask.astype('uint8')
        #             sim_map = mask
        #     else:
        #         similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
        #         similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])
        #         for b in range(similarity_map.shape[0]):
        #             for n in range(similarity_map.shape[-1]):
        #
        #                 # vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
        #                 sim_map = similarity_map[b, :, :, n].cpu().numpy()
        #                 # print('----------sim map------------\\')
        #                 # print(sim_map.shape)
        #
        #                 # 为了 可视化结果保存
        #                 # sim_map = np.nan_to_num(sim_map, nan=0.0, posinf=255.0, neginf=0.0)  # 替换非法值
        #                 vis = (sim_map * 255).astype('uint8')
        #                 vis_heatmap = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        #                 vis = cv2_img * 0.4 + vis_heatmap * 0.6
        #                 # vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
        #                 # vis_pil = Image.fromarray(vis)
        #                 # vis_pil.save(f"/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/save_clip_L14.png")
        #                 # vis_heatmap_clip = vis_heatmap
        #                 # print(vis_heatmap_clip.shape)
        #
        #         #         images_clip.append(sim_map)
        #         #     images_clips.append(torch.mean(torch.stack(images_clip, dim=0), dim=0).unsqueeze(0).repeat(3, 1, 1))
        #         #
        #         # cam = images_clips[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, 0]

        # # #---------  CLIP 分支 ---------------------


        # object_name = cls_name
        # states = ["crack", "hole", "residue", "damage"]
        #
        # prompt_normal = ' a photo of the perfect ' + object_name
        # prompts = [f"A photo of a {object_name} with {state}" for state in states]
        #
        #
        #
        # #  text data 处理
        # # prompt = text_data[0]
        # # tokens = split_text(prompt)
        # # clip_tokens_ids = self.ldm_stable.tokenizer(prompt)['input_ids']
        # # clip_tokens = [self.ldm_stable.tokenizer.decode(i) for i in clip_tokens_ids[1:-1]]
        # # if len(clip_tokens) <= 75:
        # #     sentences = [prompt]
        # # else:
        # #     splited_tokens = split_sentences(clip_tokens)
        # #     sentences = []
        # #     start = 0
        # #     clip_tokens_ids_valid = clip_tokens_ids[1:-1]  # BOS EOS
        # #     for i in range(len(splited_tokens)):
        # #         print(start, 'end={}'.format(start + len(splited_tokens[i])))
        # #         sentences += [self.ldm_stable.tokenizer.decode(
        # #             [clip_tokens_ids[0]] + clip_tokens_ids_valid[start:start + len(splited_tokens[i])] + [
        # #                 clip_tokens_ids[-1]])[15:-14]]
        # #         start += len(splited_tokens[i])
        # #
        #
        #
        #
        # images = []
        # for idx, p in enumerate(prompts):
        # #
        #     imgs = []
        #
        #     dtype = torch.float32
        #
        #     times = [100]
        #     # times = [10, 50, 100, 150, 200, 250, 300]
        #     controller = AttentionStore()
        #     g_cpu = torch.Generator(4307)
        #
        #     prompts = [p]
        #
        #     # prompt_texts = []
        #     # prompt_blip = f'a photography of {p}'
        #     # with Image.open(img_path).convert('RGB') as raw_image:
        #     #     inputs = self.blip_processor(raw_image, prompt_blip, return_tensors='pt').to(self.device)
        #     # out = self.blip_model.generate(**inputs)
        #     # out_prompt = self.blip_processor.decode(out[0], skip_special_tokens=True)
        #     # word_len = len(prompts[0].split(" "))
        #     # embs_len = len(self.tokenizer.encode(prompts[0])) - 2
        #     # out_prompt = out_prompt.split(" ")
        #     # last_word = p.split(" ")[-1]
        #     # out_prompt[2 + word_len] = f"{last_word}++"
        #     # prompt = [" ".join(out_prompt)]
        #     # prompt_texts.append(f"{idx} {prompt[0]}")
        #     # print(idx, prompt)
        #
        #     rgb_512 = img_paths.to(self.device)
        #         # print(rgb_512.shape)
        #
        #         # ------------CLIP 图像预处理---------------
        #         # with torch.no_grad():
        #         #
        #         #     img_tensor = rgb_512
        #         #     clip_img_tensor = sd_to_clip_img(img_tensor)
        #         #
        #         #     clip_features = self.clip_encoder(clip_img_tensor, output_hidden_states=True)
        #         #
        #         # image_feats = clip_features.last_hidden_state
        #         # tokens = split_text(prompt[0])
        #         #
        #         # token_index = find_token_index_by_word(tokens, p)
        #         #
        #         # # sd 语言特征提取
        #         # text_input = self.ldm_stable.tokenizer(
        #         #     prompt[0],
        #         #     padding="max_length",
        #         #     max_length=self.ldm_stable.tokenizer.model_max_length,
        #         #     truncation=True,
        #         #     return_tensors="pt",
        #         # )
        #         # text_embeddings = self.ldm_stable.text_encoder(text_input.input_ids.to(self.model.device))[0]
        #         # text_feats = text_embeddings
        #         # target_token_feat = text_feats[0, token_index]
        #         # patch_feats = image_feats[0, 1:, :]
        #         #
        #         # similarity = F.cosine_similarity(patch_feats, target_token_feat.unsqueeze(0), dim=-1)  # [N-1]
        #         # topk_indices = similarity.topk(k=2).indices  # 提取相似度最高的前两个特征
        #         # relevant_feats = patch_feats[topk_indices]
        #         #
        #         # clip_features = clip_features.hidden_states[-2][:, 1:]
        #     clip_features = None
        #
        #
        #     for t in times:
        #         controller.reset()
        #         # SD 噪声生成
        #         input_latent = encode_imgs(rgb_512, self.vae).to(self.device)
        #         noise = torch.randn([1, 4, 64, 64]).to(self.device)  # noise 随机初始化
        #         noise = noise if dtype == torch.float32 else noise.half()
        #         latents_noisy = self.ldm_stable.scheduler.add_noise(input_latent, noise, torch.tensor(t, device=self.device))
        #         latents_noisy = latents_noisy if dtype == torch.float32 else latents_noisy.half()
        #
        #         image_inv, x_t = run_and_display(prompts, controller, ldm_stable=self.ldm_stable,run_baseline=False, latent=latents_noisy,
        #                                  verbose=False,
        #                                  file_name=f'{cls_name}_{idx}', clip_image=clip_features, onlyimg=False)
        #
        #         # 注意力图获取
        #         out_atts = []
        #         weight = [0.3, 0.5, 0.1, 0.1]
        #         word_len = len(prompts[0].split(" "))
        #         embs_len = len(self.tokenizer.encode(prompts[0])) - 2
        #
        #         cross_attention_maps = aggregate_all_attention(prompts, controller, ("up", "mid", "down"), True, 0)
        #         self_attention_maps = aggregate_all_attention(prompts, controller, ("up", "mid", "down"), False, 0)
        #         for idx, res in enumerate([8, 16, 32, 64]):
        #             try:
        #                 if prompts[0].split(" ")[3 + word_len].endswith("ing"):
        #                     cross_att = cross_attention_maps[idx][:, :, [3 + embs_len, 5 + embs_len]].mean(2).view(res,
        #                                                                                                            res).float()
        #                 # print(decoder(int(tokenizer.encode(prompt[0])[3+embs_len])),decoder(int(tokenizer.encode(prompt[0])[5+embs_len])))
        #                 else:
        #                     cross_att = cross_attention_maps[idx][:, :, [3 + embs_len]].mean(2).view(res, res).float()
        #             except:
        #                 cross_att = cross_attention_maps[idx][:, :, [3 + embs_len]].mean(2).view(res, res).float()
        #
        #             if res != 64:
        #                 cross_att = F.interpolate(cross_att.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear',
        #                                           align_corners=False).squeeze().squeeze()
        #             cross_att = (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min())
        #             out_atts.append(cross_att * weight[idx])
        #
        #         # 交叉-自注意力图结合
        #         cross_att_map = torch.stack(out_atts).sum(0).view(64 * 64, 1)
        #         self_att = self_attention_maps[3].view(64 * 64, 64 * 64).float()
        #         att_map = torch.matmul(self_att, cross_att_map).view(res, res)
        #
        #         imgs.append(att_map)
        #
        #     #  不同time下得到的att_map
        #     images.append(torch.mean(torch.stack(imgs, dim=0), dim=0).unsqueeze(0).repeat(3, 1, 1))
        # # 不同sentence下得到的att_map
        # images = torch.stack(images)
        # # #
        # # # # heatmap 可视化
        # # with Image.open(img_data) as img_hw:
        # # # img_hw = Image.open(img_path)
        # #     w = img_hw.width
        # #     h = img_hw.height
        #     #
        # images = F.interpolate(images, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        # pixel_max = images.max()
        # # for i in range(images.shape[0]):
        # #     images[i] = ((images[i] - images[i].min()) / (images[i].max() - images[i].min())) * 255
        #
        # for i in range(images.shape[0]):
        #     images[i] = ((images[i] - images[i].min()) / (images[i].max() - images[i].min())+ 1e-8)
        #
        # #  attention map热力图显示
        # cam_dict = {}
        #
        # # attn = images.mean(dim=0)
        # # cam = attn.permute(1, 2, 0).cpu().detach().numpy()[:, :, 0]
        # #
        # # # cam = images[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, 0]
        # #
        # # cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # # cam_uint8 = (cam_norm * 255).astype(np.uint8)
        # # cam_colored_diffusion = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # COLORMAP_JET   COLORMAP_VIRIDIS   COLORMAP_HOT
        # # print('-----------diffusion------')
        # # print(cam_colored_diffusion)
        #
        # # cam_img = torch.softmax(images, dim=1)
        # attn_head_mean = images.mean(dim=1)
        # attn_final = attn_head_mean.mean(dim=0,keepdim=True)
        # m = attn_final.median()
        # M = attn_final.max()
        # pred_score = (m / M)
        # # print(f"Anomaly Score: {pred_score:.4f}")
        #
        # results_diff = {
        #     "pred_score": pred_score,
        #     "pred_mask": attn_final,
        # }
        #
        # results = {
        #     key: (0.3 * results_diff[key] + 0.7 * results_clip[key])
        #     for key in results_diff
        # }

        # return results_clip

        # anomaly_map_diff = torch.softmax(cam, dim=1)
        # anomaly_map_diff = (anomaly_map_diff[:, 1, :, :] - anomaly_map_diff[:, 0, :, :] + 1
        #                   ) / 2
        #
        # results_diff = {
        #     "pred_score": torch.tensor(anomaly_map_diff.max().item()),
        #     "pred_mask": anomaly_map_diff,
        # }

        #
        # cv2.imwrite('/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/cam_colored_diffusion.png',cam_colored_diffusion)
        #
        #
        #
        # #------------------ 融合 ------------------------
        #
        # heatmap_finall = cam * 0.6 + 0.4 * sim_map
        #
        #
        # merged = (cam * sim_map) + 0.5 * (cam + sim_map) / 2
        #
        # merged = np.clip(merged, 0, 255).astype(np.uint8)
        # vis_heatmap_merged = cv2.applyColorMap(merged, cv2.COLORMAP_JET)
        # cv2.imwrite(
        #     '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/heatmap_merged.png',
        #     vis_heatmap_merged)


       # # 相乘保证保留高响应，相加alpha ∈ [0, 0.5] 控制“高响应不一致位置”保留程度
        # #heatmap_finall = cam_colored_diffusion * vis_heatmap_clip + alpha * (cam_colored_diffusion + vis_heatmap_clip) / 2
        #
        # cv2.imwrite('/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/heatmap_finall.png',heatmap_finall)


        #--------------------------------------------------------------------

        #---------------------        最大外接矩形      -----------------------

        # heatmap_img(img_hw,cam_colored)

        # Gen_bbox_all(cam,img_hw)
        # with Image.open(img_path) as img_hw:
        #     bbox, mask = Gen_bbox_single(sim_map, img_hw)   # 可以设置save_path, save=True   sim_map
        # # print(bbox)
        #
        # pred_box = bbox
        #
        # print(pred_box)
        #
        # return pred_box

    def forward_sentences(self, img_paths, img_data, sentences, use_sam=False):

        # ---------  CLIP 分支 ---------------------
        img_path = img_paths[0]
        # text_data = sentences[0]
        device = self.device
        #

        images_clip = []
        images_clips = []

        with Image.open(img_path) as pil_img:
            # pil_img = Image.open(img_path)

            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert("RGB")
                # pil_img = np.stack([pil_img] * 3, axis=-1)

            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            clip_preprocess = Compose([Resize((512, 512), interpolation=BICUBIC), ToTensor(),
                                       Normalize((0.48145466, 0.4578275, 0.40821073),
                                                 (0.26862954, 0.26130258, 0.27577711))])
            if use_sam == True:
                self.sam_predictor.set_image(np.array(pil_img))

            # print(pil_img)
            image = clip_preprocess(pil_img).unsqueeze(0).to(device)

        for text_data in sentences[0].split(' '):
            #
            with torch.no_grad():
                # CLIP architecture surgery acts on the image encoder
                image_features = self.clip_surgery.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Prompt ensemble for text features with normalization    text_data 应该是list
                text_features = clip.encode_text_with_prompt_ensemble(self.clip_surgery, text_data, device)
                # text_features = clip.encode_text_with_prompt_ensemble(self.clip_surgery, text_data.split(' '), device)  # 分词

                #
                # # Extract redundant features from an empty string
                redundant_features = clip.encode_text_with_prompt_ensemble(self.clip_surgery, [""], device)

                # Apply feature surgery for single text
                # similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)

                # similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

                # print('------similarity_map----------------')
                # print(similarity_map.shape)

                # Draw similarity map
                if use_sam:
                    similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0]
                    for n in range(similarity.shape[-1]):
                        points, labels = clip.similarity_map_to_points(similarity[1:, n], cv2_img.shape[:2], t=0.8)
                        masks, scores, logits = self.sam_predictor.predict(point_labels=labels,
                                                                           point_coords=np.array(points),
                                                                           multimask_output=True)
                        mask = masks[np.argmax(scores)]
                        mask = mask.astype('uint8')
                        sim_map = mask
                else:
                    similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
                    similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])
                    for b in range(similarity_map.shape[0]):
                        for n in range(similarity_map.shape[-1]):
                            # vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
                            sim_map = similarity_map[b, :, :, n].cpu().numpy()
                            # print('----------sim map------------\\')
                            # print(sim_map.shape)
                            # 为了 可视化结果保存
                            # sim_map = np.nan_to_num(sim_map, nan=0.0, posinf=255.0, neginf=0.0)  # 替换非法值
                            vis = (sim_map * 255).astype('uint8')
                            vis_heatmap = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                            vis = cv2_img * 0.4 + vis_heatmap * 0.6
                            # vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                            # vis_pil = Image.fromarray(vis)
                            # vis_pil.save(f"/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/save_clip_L14.png")
                            # vis_heatmap_clip = vis_heatmap
                            # print(vis_heatmap_clip.shape)

                            images_clip.append(sim_map)    # 不同 text 下 的 map
        images_clip = [torch.as_tensor(x) for x in images_clip]
        images_clips.append(torch.mean(torch.stack(images_clip, dim=0), dim=0).unsqueeze(0).repeat(3, 1, 1))  # 合并不同 text 下 的 map

        cam_clip = images_clips[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, 0]  # 输出

            # # #---------  CLIP 分支 ---------------------

        #  text data 处理
        # prompt = text_data[0]
        # tokens = split_text(prompt)
        # clip_tokens_ids = self.ldm_stable.tokenizer(prompt)['input_ids']
        # clip_tokens = [self.ldm_stable.tokenizer.decode(i) for i in clip_tokens_ids[1:-1]]
        # if len(clip_tokens) <= 75:
        #     sentences = [prompt]
        # else:
        #     splited_tokens = split_sentences(clip_tokens)
        #     sentences = []
        #     start = 0
        #     clip_tokens_ids_valid = clip_tokens_ids[1:-1]  # BOS EOS
        #     for i in range(len(splited_tokens)):
        #         print(start, 'end={}'.format(start + len(splited_tokens[i])))
        #         sentences += [self.ldm_stable.tokenizer.decode(
        #             [clip_tokens_ids[0]] + clip_tokens_ids_valid[start:start + len(splited_tokens[i])] + [
        #                 clip_tokens_ids[-1]])[15:-14]]
        #         start += len(splited_tokens[i])
        # #

        # for idx, p in enumerate(sentences[0]):
        # #
        #     imgs = []
        #     images = []
        #     dtype = torch.float32
        # #
        #     # times = [100]
        #     times = [100]
        #     controller = AttentionStore()
        #     g_cpu = torch.Generator(4307)
        #
        #     prompts = [p]
        #
        # #     prompt_texts = []
        # #     prompt_blip = f'a photography of {p}'
        # #     with Image.open(img_path).convert('RGB') as raw_image:
        # #         inputs = self.blip_processor(raw_image, prompt_blip, return_tensors='pt').to(self.device)
        # #     out = self.blip_model.generate(**inputs)
        # #     out_prompt = self.blip_processor.decode(out[0], skip_special_tokens=True)
        # #     word_len = len(prompts[0].split(" "))
        # #     embs_len = len(self.tokenizer.encode(prompts[0])) - 2
        # #     out_prompt = out_prompt.split(" ")
        # #     last_word = p.split(" ")[-1]
        # #     out_prompt[2 + word_len] = f"{last_word}++"
        # #     prompt = [" ".join(out_prompt)]
        # #     prompt_texts.append(f"{idx} {prompt[0]}")
        # #     print(idx, prompt)
        # #
        #     rgb_512 = img_data.to(self.device)
        # #         # print(rgb_512.shape)
        # #
        # #         # ------------CLIP 图像预处理---------------
        # #         # with torch.no_grad():
        # #         #
        # #         #     img_tensor = rgb_512
        # #         #     clip_img_tensor = sd_to_clip_img(img_tensor)
        # #         #
        # #         #     clip_features = self.clip_encoder(clip_img_tensor, output_hidden_states=True)
        # #         #
        # #         # image_feats = clip_features.last_hidden_state
        # #         # tokens = split_text(prompt[0])
        # #         #
        # #         # token_index = find_token_index_by_word(tokens, p)
        # #         #
        # #         # # sd 语言特征提取
        # #         # text_input = self.ldm_stable.tokenizer(
        # #         #     prompt[0],
        # #         #     padding="max_length",
        # #         #     max_length=self.ldm_stable.tokenizer.model_max_length,
        # #         #     truncation=True,
        # #         #     return_tensors="pt",
        # #         # )
        # #         # text_embeddings = self.ldm_stable.text_encoder(text_input.input_ids.to(self.model.device))[0]
        # #         # text_feats = text_embeddings
        # #         # target_token_feat = text_feats[0, token_index]
        # #         # patch_feats = image_feats[0, 1:, :]
        # #         #
        # #         # similarity = F.cosine_similarity(patch_feats, target_token_feat.unsqueeze(0), dim=-1)  # [N-1]
        # #         # topk_indices = similarity.topk(k=2).indices  # 提取相似度最高的前两个特征
        # #         # relevant_feats = patch_feats[topk_indices]
        # #         #
        # #         # clip_features = clip_features.hidden_states[-2][:, 1:]
        #     clip_features = None
        # #
        # #
        #     for t in times:
        #         controller.reset()
        #         # SD 噪声生成
        #         input_latent = encode_imgs(rgb_512, self.vae).to(self.device)
        #         noise = torch.randn([1, 4, 64, 64]).to(self.device)  # noise 随机初始化
        #         noise = noise if dtype == torch.float32 else noise.half()
        #         latents_noisy = self.ldm_stable.scheduler.add_noise(input_latent, noise, torch.tensor(t, device=self.device))
        #         latents_noisy = latents_noisy if dtype == torch.float32 else latents_noisy.half()
        #
        #         image_inv, x_t = run_and_display(prompts, controller, ldm_stable=self.ldm_stable,run_baseline=False, latent=latents_noisy,
        #                                  verbose=False,
        #                                  file_name=f'{img_path}_{idx}', clip_image=clip_features, onlyimg=False)
        #
        #         # 注意力图获取
        #         out_atts = []
        #         weight = [0.3, 0.5, 0.1, 0.1]
        #         word_len = len(prompts[0].split(" "))
        #         embs_len = len(self.tokenizer.encode(prompts[0])) - 2
        #         import torch.nn.functional as F
        #
        #         cross_attention_maps = aggregate_all_attention(prompts, controller, ("up", "mid", "down"), True, 0)
        #         self_attention_maps = aggregate_all_attention(prompts, controller, ("up", "mid", "down"), False, 0)
        #         for idx, res in enumerate([8, 16, 32, 64]):
        #             try:
        #                 if prompts[0].split(" ")[3 + word_len].endswith("ing"):
        #                     cross_att = cross_attention_maps[idx][:, :, [3 + embs_len, 5 + embs_len]].mean(2).view(res,
        #                                                                                                            res).float()
        #                 # print(decoder(int(tokenizer.encode(prompt[0])[3+embs_len])),decoder(int(tokenizer.encode(prompt[0])[5+embs_len])))
        #                 else:
        #                     cross_att = cross_attention_maps[idx][:, :, [3 + embs_len]].mean(2).view(res, res).float()
        #             except:
        #                 cross_att = cross_attention_maps[idx][:, :, [3 + embs_len]].mean(2).view(res, res).float()
        #
        #             if res != 64:
        #                 cross_att = F.interpolate(cross_att.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear',
        #                                           align_corners=False).squeeze().squeeze()
        #             cross_att = (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min())
        #             out_atts.append(cross_att * weight[idx])
        #
        #         # 交叉-自注意力图结合
        #         cross_att_map = torch.stack(out_atts).sum(0).view(64 * 64, 1)
        #         self_att = self_attention_maps[3].view(64 * 64, 64 * 64).float()
        #         att_map = torch.matmul(self_att, cross_att_map).view(res, res)
        #
        #         imgs.append(att_map)
        #
        #     #  不同time下得到的att_map
        #     images.append(torch.mean(torch.stack(imgs, dim=0), dim=0).unsqueeze(0).repeat(3, 1, 1))
        # # 不同sentence下得到的att_map
        # images = torch.stack(images)
        # # # #
        # # # # # heatmap 可视化
        # with Image.open(img_path) as img_hw:
        # # img_hw = Image.open(img_path)
        #     w = img_hw.width
        #     h = img_hw.height
        #     #
        #     images = F.interpolate(images, size=(h, w), mode='bilinear', align_corners=False)
        #     pixel_max = images.max()
        # #     for i in range(images.shape[0]):
        # #         images[i] = ((images[i] - images[i].min()) / (images[i].max() - images[i].min())) * 255
        # # #
        # # #  attention map热力图显示
        # cam_dict = {}
        # cam = images[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, 0]
        # print(cam.shape)
        # #
        # cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # cam_uint8 = (cam_norm * 255).astype(np.uint8)
        # cam_colored_diffusion = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # COLORMAP_JET   COLORMAP_VIRIDIS   COLORMAP_HOT
        # # print('-----------diffusion------')
        # # print(cam_colored_diffusion)
        # #
        # cv2.imwrite('/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/cam_colored_diffusion.png',cam_colored_diffusion)
        # #
        # #
        #
        #
        # #------------------ 融合 ------------------------
        #
        # heatmap_finall = cam * 0.6 + 0.4 * sim_map
        #
        #
        # merged = (cam * sim_map) + 0.5 * (cam + sim_map) / 2
        #
        # merged = np.clip(merged, 0, 255).astype(np.uint8)
        # vis_heatmap_merged = cv2.applyColorMap(merged, cv2.COLORMAP_JET)
        # cv2.imwrite(
        #     '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/heatmap_merged.png',
        #     vis_heatmap_merged)

        # # 相乘保证保留高响应，相加alpha ∈ [0, 0.5] 控制“高响应不一致位置”保留程度
        # #heatmap_finall = cam_colored_diffusion * vis_heatmap_clip + alpha * (cam_colored_diffusion + vis_heatmap_clip) / 2
        #
        # cv2.imwrite('/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/heatmap_finall.png',heatmap_finall)

        # --------------------------------------------------------------------
        # heatmap_finall = cam_colored_diffusion * cam_colored_diffusion_clip + 0.5 * (
        #             cam_colored_diffusion + cam_colored_diffusion_clip) / 2
        # ---------------------        最大外接矩形      -----------------------

        # heatmap_img(img_hw,cam_colored)

        # Gen_bbox_all(cam,img_hw)
        with Image.open(img_path) as img_hw:
            bbox, mask = Gen_bbox_single(cam_clip, img_hw)  # 可以设置save_path, save=True   sim_map
        # print(bbox)

        pred_box = bbox

        print(pred_box)

        return pred_box



    def forward_mutliP(self, img_paths,img_data, text_data, cls_name, use_sam=False):

        # 多个prompt 一起输入到diffusion中


        clip_transformed_image = self.transform_clip(img_paths)

        with torch.no_grad():
            image_features = self.clip_surgery.encode_image(
                clip_transformed_image)
        text_features = self.text_prompts["object"].to(self.device)

        anomaly_map_vl = 100.0 * image_features[:, 1:, :] @ text_features

        B, L, C = anomaly_map_vl.shape
        H = int(np.sqrt(L))
        anomaly_map_vl = F.interpolate(
            anomaly_map_vl.permute(0, 2, 1).view(B, 2, H, H),
            size=self.image_size,
            mode="bilinear",
            align_corners=True,
        )  # [1,2,512,512]
        anomaly_map_vl = torch.softmax(anomaly_map_vl, dim=1)    # [1,2,512,512]
        anomaly_map_vl = ( anomaly_map_vl[:, 1, :, :] - anomaly_map_vl[:, 0, :, :] + 1
                         ) / 2    # [1,512,512]

        results_clip = {
            "pred_score": torch.tensor(anomaly_map_vl.max().item()),
            "pred_mask": anomaly_map_vl.to('cpu'),
        }
        #
        # return results

        #---------  CLIP 分支 ---------------------
        # img_path = img_paths[0]
        # text_data = text_data[0]
        # device = self.device
        # #
        # with Image.open(img_path) as pil_img:
        # # pil_img = Image.open(img_path)
        #
        #     if pil_img.mode != 'RGB':
        #         pil_img = pil_img.convert("RGB")
        #         pil_img = np.stack([pil_img] * 3, axis=-1)
        #
        #     cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        #     clip_preprocess = Compose([Resize((512, 512), interpolation=BICUBIC), ToTensor(),
        #                            Normalize((0.48145466, 0.4578275, 0.40821073),
        #                                      (0.26862954, 0.26130258, 0.27577711))])
        #     if use_sam == True:
        #         self.sam_predictor.set_image(np.array(pil_img))
        #
        #     # print(pil_img)
        #     image = clip_preprocess(pil_img).unsqueeze(0).to(device)
        # #
        #
        # with torch.no_grad():
        #     # CLIP architecture surgery acts on the image encoder
        #     image_features = self.clip_surgery.encode_image(image)
        #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #
        #     # Prompt ensemble for text features with normalization    text_data 应该是list
        #     # text_features = clip.encode_text_with_prompt_ensemble(self.clip_surgery, text_data, device)
        #     text_features = clip.encode_text_with_prompt_ensemble(self.clip_surgery, text_data.split(' '), device)
        #
        #     #
        #     # # Extract redundant features from an empty string
        #     redundant_features = clip.encode_text_with_prompt_ensemble(self.clip_surgery, [""], device)
        #
        #     # Apply feature surgery for single text
        #     # similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
        #
        #     # similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])
        #
        #     # print('------similarity_map----------------')
        #     # print(similarity_map.shape)
        #
        #     # Draw similarity map
        #     if use_sam:
        #         similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0]
        #         for n in range(similarity.shape[-1]):
        #
        #             points, labels = clip.similarity_map_to_points(similarity[1:, n], cv2_img.shape[:2], t=0.8)
        #             masks, scores, logits = self.sam_predictor.predict(point_labels=labels, point_coords=np.array(points),
        #                                                       multimask_output=True)
        #             mask = masks[np.argmax(scores)]
        #             mask = mask.astype('uint8')
        #             sim_map = mask
        #     else:
        #         similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
        #         similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])
        #         for b in range(similarity_map.shape[0]):
        #             for n in range(similarity_map.shape[-1]):
        #
        #                 # vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
        #                 sim_map = similarity_map[b, :, :, n].cpu().numpy()
        #                 # print('----------sim map------------\\')
        #                 # print(sim_map.shape)
        #
        #                 # 为了 可视化结果保存
        #                 # sim_map = np.nan_to_num(sim_map, nan=0.0, posinf=255.0, neginf=0.0)  # 替换非法值
        #                 vis = (sim_map * 255).astype('uint8')
        #                 vis_heatmap = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        #                 vis = cv2_img * 0.4 + vis_heatmap * 0.6
        #                 # vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
        #                 # vis_pil = Image.fromarray(vis)
        #                 # vis_pil.save(f"/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/save_clip_L14.png")
        #                 # vis_heatmap_clip = vis_heatmap
        #                 # print(vis_heatmap_clip.shape)
        #
        #         #         images_clip.append(sim_map)
        #         #     images_clips.append(torch.mean(torch.stack(images_clip, dim=0), dim=0).unsqueeze(0).repeat(3, 1, 1))
        #         #
        #         # cam = images_clips[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, 0]

        # # # #---------  CLIP 分支 ---------------------
        # object_name = cls_name
        # states = ["crack", "hole", "residue", "damage"]
        #
        # prompt_normal = ' a photo of the perfect ' + object_name
        # prompts = [f"A photo of a {object_name} with {state}" for state in states]
        #
        #
        #
        # #  text data 处理
        # # prompt = text_data[0]
        # # tokens = split_text(prompt)
        # # clip_tokens_ids = self.ldm_stable.tokenizer(prompt)['input_ids']
        # # clip_tokens = [self.ldm_stable.tokenizer.decode(i) for i in clip_tokens_ids[1:-1]]
        # # if len(clip_tokens) <= 75:
        # #     sentences = [prompt]
        # # else:
        # #     splited_tokens = split_sentences(clip_tokens)
        # #     sentences = []
        # #     start = 0
        # #     clip_tokens_ids_valid = clip_tokens_ids[1:-1]  # BOS EOS
        # #     for i in range(len(splited_tokens)):
        # #         print(start, 'end={}'.format(start + len(splited_tokens[i])))
        # #         sentences += [self.ldm_stable.tokenizer.decode(
        # #             [clip_tokens_ids[0]] + clip_tokens_ids_valid[start:start + len(splited_tokens[i])] + [
        # #                 clip_tokens_ids[-1]])[15:-14]]
        # #         start += len(splited_tokens[i])
        # #
        # images = []
        #
        # N_prompt = len(prompts)
        #
        # dtype = torch.float32
        # times = [100]
        # controller = AttentionStore()
        # g_cpu = torch.Generator(4307)
        # rgb_512 = img_paths.to(self.device)
        #
        #
        # controller.reset()
        # input_latent = encode_imgs(rgb_512, self.vae).to(self.device)  # [1, 4, 64, 64]
        # input_latent = input_latent.repeat(N_prompt, 1, 1, 1)
        #
        # noise = torch.randn([N_prompt, 4, 64, 64]).to(self.device)
        # noise = noise if dtype == torch.float32 else noise.half()
        #
        # token_idx_list = []
        # for p in prompts:
        #     token_idx = len(self.tokenizer.encode(p)) - 2  # 假设只要状态词在最后
        #     token_idx_list.append(token_idx)
        #
        # imgs_time = []
        #
        # for t in times:
        #     controller.reset()  # 重置 Attention
        #
        #     t_batch = torch.tensor([t] * N_prompt, device=self.device)
        #     latents_noisy = self.ldm_stable.scheduler.add_noise(
        #         input_latent, noise, t_batch
        #     )
        #     latents_noisy = latents_noisy if dtype == torch.float32 else latents_noisy.half()
        #
        #     image_inv, x_t = run_and_display(
        #         prompts,
        #         controller,
        #         ldm_stable=self.ldm_stable,
        #         run_baseline=False,
        #         latent=latents_noisy,
        #         verbose=False,
        #         file_name=f'{cls_name}_BATCH',
        #         clip_image=None,
        #         onlyimg=False
        #     )
        #
        #     # cross_attention_maps = aggregate_all_attention(
        #     #     prompts,
        #     #     controller,
        #     #     ("up", "mid", "down"),
        #     #     is_cross=True,
        #     #     select=0
        #     # )  # [N_layer, B, Head, Q, K]
        #     #
        #     # self_attention_maps = aggregate_all_attention(
        #     #     prompts,
        #     #     controller,
        #     #     ("up", "mid", "down"),
        #     #     is_cross=False,
        #     #     select=0
        #     # )
        #
        #     cross_attention_maps = aggregate_all_attention_mutliP(
        #         prompts,
        #         controller,
        #         ("up", "mid", "down"),
        #         is_cross=True,
        #         select=0
        #     )  # [N_layer, B, Head, Q, K]
        #
        #     self_attention_maps = aggregate_all_attention_mutliP(
        #         prompts,
        #         controller,
        #         ("up", "mid", "down"),
        #         is_cross=False,
        #         select=0
        #     )
        #
        #     weight = [0.3, 0.5, 0.1, 0.1]  # 你原来的层权重
        #     out_atts = []
        #
        #     word_len = len(prompts[0].split(" "))
        #     embs_len = len(self.tokenizer.encode(prompts[0])) - 2
        #
        #     for idx_layer, res in enumerate([8, 16, 32, 64]):
        #         layer_map = cross_attention_maps[idx_layer]  # [B, Head, Q, K]
        #
        #         # ➜ 【关键改】对 Head 做 mean
        #         outs_layer = []
        #
        #         for b in range(N_prompt):
        #             tok_idx = token_idx_list[b]
        #
        #             try:
        #                 cross_att = layer_map[b, :,  tok_idx].view(res, res)
        #             except:
        #                 cross_att = layer_map[b, :,  tok_idx].view(res, res)
        #
        #             outs_layer.append(cross_att)
        #         outs_layer = torch.stack(outs_layer, dim=0)
        #
        #         if res != 64:
        #             outs_layer  = F.interpolate(
        #                 outs_layer.unsqueeze(1), size=(64, 64),
        #                 mode='bilinear', align_corners=False
        #             ).squeeze(1)
        #
        #         # ➜ min-max normalize
        #         outs_layer = (outs_layer - outs_layer.flatten(1).min(1, keepdim=True)[0].unsqueeze(-1)) / \
        #                      (outs_layer.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1) -
        #                       outs_layer.flatten(1).min(1, keepdim=True)[0].unsqueeze(-1) + 1e-5)
        #         outs_layer = outs_layer.view(N_prompt, 64, 64)
        #
        #         out_atts.append(outs_layer * weight[idx_layer])  # [B, 64, 64]
        #
        #     cross_att_map = torch.stack(out_atts).sum(dim=0)
        #
        #     # ➜ 所有层加权求和
        #     # cross_att_map = torch.stack(out_atts).sum(dim=0)  # [B, 64, 64]
        #
        #     # ➜ 也可以加自注意力
        #     self_att = self_attention_maps[3]  # [B, Head, Q, K]
        #     # self_att = self_att.mean(dim=1)  # [B, Q, K]
        #     # self_att = self_att.sum(dim=-1)  # [B, Q]
        #     # self_att = self_att.view(N_prompt, 64 * 64, 1)
        #     self_att = self_att
        #
        #     cross_att_flat = cross_att_map.view(N_prompt, 64 * 64, 1)
        #     self_att_flat = self_att.view(N_prompt, 64 * 64, 64 * 64)
        #     att_map = torch.bmm(self_att_flat, cross_att_flat).view(N_prompt, 64, 64)  # [B, 64, 64]
        #
        #     # att_map = F.interpolate(att_map, size=(N_prompt,self.image_size, self.image_size), mode='bilinear',
        #     #                        align_corners=False)
        #
        #     # att_map = torch.bmm(self_att.transpose(1, 2), cross_att_flat).view(N_prompt, 64, 64)
        #
        #
        #     imgs_time.append(att_map)
        #
        # imgs_time = torch.stack(imgs_time).mean(dim=0)
        #
        # imgs_time = imgs_time.unsqueeze(1)
        #
        # imgs_time = F.interpolate(
        #     imgs_time,
        #     size=(512, 512),
        #     mode="bilinear",
        #     align_corners=False
        # )
        # imgs_time = imgs_time.squeeze(1)
        #
        # final_map = imgs_time.mean(dim=0)
        #
        # m = final_map.median()
        # M = final_map.max()
        # pred_score = (m / M)
        #
        # results_diff = {
        #     "pred_score": pred_score,
        #     "pred_mask": final_map,
        # }
        #
        # results = {
        #     key: (0.3 * results_diff[key] + 0.7 * results_clip[key])
        #     for key in results_diff
        # }

        return results_clip
        # for idx, p in enumerate(prompts):
        # #
        #     imgs = []
        #
        #     dtype = torch.float32
        #
        #     times = [100]
        #     # times = [10, 50, 100, 150, 200, 250, 300]
        #     controller = AttentionStore()
        #     g_cpu = torch.Generator(4307)
        #
        #     prompts = [p]
        #
        #     # prompt_texts = []
        #     # prompt_blip = f'a photography of {p}'
        #     # with Image.open(img_path).convert('RGB') as raw_image:
        #     #     inputs = self.blip_processor(raw_image, prompt_blip, return_tensors='pt').to(self.device)
        #     # out = self.blip_model.generate(**inputs)
        #     # out_prompt = self.blip_processor.decode(out[0], skip_special_tokens=True)
        #     # word_len = len(prompts[0].split(" "))
        #     # embs_len = len(self.tokenizer.encode(prompts[0])) - 2
        #     # out_prompt = out_prompt.split(" ")
        #     # last_word = p.split(" ")[-1]
        #     # out_prompt[2 + word_len] = f"{last_word}++"
        #     # prompt = [" ".join(out_prompt)]
        #     # prompt_texts.append(f"{idx} {prompt[0]}")
        #     # print(idx, prompt)
        #
        #     rgb_512 = img_paths.to(self.device)
        #         # print(rgb_512.shape)
        #
        #         # ------------CLIP 图像预处理---------------
        #         # with torch.no_grad():
        #         #
        #         #     img_tensor = rgb_512
        #         #     clip_img_tensor = sd_to_clip_img(img_tensor)
        #         #
        #         #     clip_features = self.clip_encoder(clip_img_tensor, output_hidden_states=True)
        #         #
        #         # image_feats = clip_features.last_hidden_state
        #         # tokens = split_text(prompt[0])
        #         #
        #         # token_index = find_token_index_by_word(tokens, p)
        #         #
        #         # # sd 语言特征提取
        #         # text_input = self.ldm_stable.tokenizer(
        #         #     prompt[0],
        #         #     padding="max_length",
        #         #     max_length=self.ldm_stable.tokenizer.model_max_length,
        #         #     truncation=True,
        #         #     return_tensors="pt",
        #         # )
        #         # text_embeddings = self.ldm_stable.text_encoder(text_input.input_ids.to(self.model.device))[0]
        #         # text_feats = text_embeddings
        #         # target_token_feat = text_feats[0, token_index]
        #         # patch_feats = image_feats[0, 1:, :]
        #         #
        #         # similarity = F.cosine_similarity(patch_feats, target_token_feat.unsqueeze(0), dim=-1)  # [N-1]
        #         # topk_indices = similarity.topk(k=2).indices  # 提取相似度最高的前两个特征
        #         # relevant_feats = patch_feats[topk_indices]
        #         #
        #         # clip_features = clip_features.hidden_states[-2][:, 1:]
        #     clip_features = None
        #
        #
        #     for t in times:
        #         controller.reset()
        #         # SD 噪声生成
        #         input_latent = encode_imgs(rgb_512, self.vae).to(self.device)
        #         noise = torch.randn([1, 4, 64, 64]).to(self.device)  # noise 随机初始化
        #         noise = noise if dtype == torch.float32 else noise.half()
        #         latents_noisy = self.ldm_stable.scheduler.add_noise(input_latent, noise, torch.tensor(t, device=self.device))
        #         latents_noisy = latents_noisy if dtype == torch.float32 else latents_noisy.half()
        #
        #         image_inv, x_t = run_and_display(prompts, controller, ldm_stable=self.ldm_stable,run_baseline=False, latent=latents_noisy,
        #                                  verbose=False,
        #                                  file_name=f'{cls_name}_{idx}', clip_image=clip_features, onlyimg=False)
        #
        #         # 注意力图获取
        #         out_atts = []
        #         weight = [0.3, 0.5, 0.1, 0.1]
        #         word_len = len(prompts[0].split(" "))
        #         embs_len = len(self.tokenizer.encode(prompts[0])) - 2
        #
        #         cross_attention_maps = aggregate_all_attention(prompts, controller, ("up", "mid", "down"), True, 0)
        #         self_attention_maps = aggregate_all_attention(prompts, controller, ("up", "mid", "down"), False, 0)
        #         for idx, res in enumerate([8, 16, 32, 64]):
        #             try:
        #                 if prompts[0].split(" ")[3 + word_len].endswith("ing"):
        #                     cross_att = cross_attention_maps[idx][:, :, [3 + embs_len, 5 + embs_len]].mean(2).view(res,
        #                                                                                                            res).float()
        #                 # print(decoder(int(tokenizer.encode(prompt[0])[3+embs_len])),decoder(int(tokenizer.encode(prompt[0])[5+embs_len])))
        #                 else:
        #                     cross_att = cross_attention_maps[idx][:, :, [3 + embs_len]].mean(2).view(res, res).float()
        #             except:
        #                 cross_att = cross_attention_maps[idx][:, :, [3 + embs_len]].mean(2).view(res, res).float()
        #
        #             if res != 64:
        #                 cross_att = F.interpolate(cross_att.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear',
        #                                           align_corners=False).squeeze().squeeze()
        #             cross_att = (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min())
        #             out_atts.append(cross_att * weight[idx])
        #
        #         # 交叉-自注意力图结合
        #         cross_att_map = torch.stack(out_atts).sum(0).view(64 * 64, 1)
        #         self_att = self_attention_maps[3].view(64 * 64, 64 * 64).float()
        #         att_map = torch.matmul(self_att, cross_att_map).view(res, res)
        #
        #         imgs.append(att_map)
        #
        #     #  不同time下得到的att_map
        #     images.append(torch.mean(torch.stack(imgs, dim=0), dim=0).unsqueeze(0).repeat(3, 1, 1))
        # # 不同sentence下得到的att_map
        # images = torch.stack(images)
        # # #
        # # # # heatmap 可视化
        # # with Image.open(img_data) as img_hw:
        # # # img_hw = Image.open(img_path)
        # #     w = img_hw.width
        # #     h = img_hw.height
        #     #
        # images = F.interpolate(images, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        # pixel_max = images.max()
        # # for i in range(images.shape[0]):
        # #     images[i] = ((images[i] - images[i].min()) / (images[i].max() - images[i].min())) * 255
        #
        # for i in range(images.shape[0]):
        #     images[i] = ((images[i] - images[i].min()) / (images[i].max() - images[i].min())+ 1e-8)
        #
        # #  attention map热力图显示
        # cam_dict = {}
        #
        # # attn = images.mean(dim=0)
        # # cam = attn.permute(1, 2, 0).cpu().detach().numpy()[:, :, 0]
        # #
        # # # cam = images[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, 0]
        # #
        # # cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # # cam_uint8 = (cam_norm * 255).astype(np.uint8)
        # # cam_colored_diffusion = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # COLORMAP_JET   COLORMAP_VIRIDIS   COLORMAP_HOT
        # # print('-----------diffusion------')
        # # print(cam_colored_diffusion)
        #
        # # cam_img = torch.softmax(images, dim=1)
        # attn_head_mean = images.mean(dim=1)
        # attn_final = attn_head_mean.mean(dim=0,keepdim=True)
        # m = attn_final.median()
        # M = attn_final.max()
        # pred_score = (m / M)
        # # print(f"Anomaly Score: {pred_score:.4f}")
        #
        # results_diff = {
        #     "pred_score": pred_score,
        #     "pred_mask": attn_final,
        # }
        #
        # results = {
        #     key: (0.3 * results_diff[key] + 0.7 * results_clip[key])
        #     for key in results_diff
        # }
        #
        # return results

    def run_clip_proto(self,img_paths,):

        clip_transformed_image = self.transform_clip(img_paths)

        with torch.no_grad():

            image_features, patch_features = self.clip_surgery.encode_image(
                clip_transformed_image)

        #
        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        patch_tokens = self.decoder(patch_features)

        # k-shot 部分（支持多 batch）
        if self.shot:
            global_score = 1 - (image_features @ self.normal_image_features.transpose(-2, -1)).amax(dim=-1)
            sims = []
            for i in range(len(patch_tokens)):
                if i % 2 == 0:
                    continue

                # if i in [0,2,3,5] :   # vit-16
                #     continue

                # patch_tokens[i]: [B, L, C]
                pt = patch_tokens[i]
                B, L, C = pt.shape

                # 如果你的 token 里还包含 CLS（常见是 L = 1 + H*W），先去掉 CLS：
                # pt = pt[:, 1:, :]
                # B, L, C = pt.shape

                # 确保是连续内存再 view（有时上游 permute/linear 会导致非连续）
                pt = pt.contiguous().view(B, L, 1, C)  # [B, L, 1, C]

                # normal tokens: 形状通常为 [N_norm, C] 或 [?, C]
                # 原代码是 reshape(1, -1, 1024) —— 这里保持 C 一致即可：
                nt = self.normal_patch_tokens[i].reshape(1, -1, C).unsqueeze(0)  # [1, 1, N_norm, C]

                # 广播到 [B, L, N_norm, C] 后按最后一维做余弦相似度
                cosine_similarity_matrix = F.cosine_similarity(pt, nt, dim=-1)  # [B, L, N_norm]

                # 对 normal 集合取最大相似度：得到每个 patch 的最相近 normal 相似度
                sim_max, _ = torch.max(cosine_similarity_matrix, dim=2)  # [B, L]
                sims.append(sim_max)

            # 跨所选层求均值
            sim = torch.mean(torch.stack(sims, dim=0), dim=0)  # [B, L]

            # 把 [B, L] 还原到特征图网格
            H = int(np.sqrt(L))
            assert H * H == L, f"L={L} 不是完全平方（可能还包含 CLS token？）"
            sim = sim.view(B, 1, H, H)  # [B, 1, H, H]

            # 上采样到图像尺寸
            sim = F.interpolate(sim, size=self.image_size, mode="bilinear", align_corners=True)
            anomaly_map_ret = 1 - sim
        # k-shot 部分
        else:

            top_k = 5
            patch_token_pro = patch_tokens[-2] @ self.clip_surgery.visual.proj
            sims = patch_token_pro @ self.text_feature_prototype[:,0].unsqueeze(-1)
            sims = sims.squeeze(-1)
            topk_vals, topk_idx = torch.topk(sims, k=top_k, dim=1)
            selected_tokens = torch.gather(
                patch_token_pro, 1, topk_idx.unsqueeze(-1).expand(-1, -1, patch_token_pro.size(-1))
            )
            p_v = selected_tokens.mean(dim=1)
            self.p_v = p_v / p_v.norm()

            # topk_scores = (image_features @ self.p_v.transpose(-2, -1)).topk(top_k, dim=-1).values
            # global_score = 1 - topk_scores.mean(dim=-1)

            global_score = 1 - image_features @ self.p_v.transpose(-2, -1).amax(dim=-1)



            # global_score = 0

            sims = []
            for i in range(len(patch_tokens)):
                # if i % 2 == 0:
                #     continue

                if i != 3:
                    continue

                # patch_tokens[i]: [B, L, C]
                pt = patch_tokens[i] @ self.clip_surgery.visual.proj
                B, L, C = pt.shape

                # 如果你的 token 里还包含 CLS（常见是 L = 1 + H*W），先去掉 CLS：
                # pt = pt[:, 1:, :]
                # B, L, C = pt.shape

                # 确保是连续内存再 view（有时上游 permute/linear 会导致非连续）
                pt = pt.contiguous().view(B, L, 1, C)  # [B, L, 1, C]

                # normal tokens: 形状通常为 [N_norm, C] 或 [?, C]
                # 原代码是 reshape(1, -1, 1024) —— 这里保持 C 一致即可：
                nt = self.p_v[0].reshape(1, -1, C).unsqueeze(0)  # [1, 1, N_norm, C]

                # 广播到 [B, L, N_norm, C] 后按最后一维做余弦相似度
                cosine_similarity_matrix = F.cosine_similarity(pt, nt, dim=-1)  # [B, L, N_norm]

                # 对 normal 集合取最大相似度：得到每个 patch 的最相近 normal 相似度
                sim_max, _ = torch.max(cosine_similarity_matrix, dim=2)  # [B, L]
                sims.append(sim_max)

            # 跨所选层求均值
            sim = torch.mean(torch.stack(sims, dim=0), dim=0)  # [B, L]

            # 把 [B, L] 还原到特征图网格
            H = int(np.sqrt(L))
            assert H * H == L, f"L={L} 不是完全平方（可能还包含 CLS token？）"
            sim = sim.view(B, 1, H, H)  # [B, 1, H, H]

            # 上采样到图像尺寸
            sim = F.interpolate(sim, size=self.image_size, mode="bilinear", align_corners=True)
            anomaly_map_ret = 1 - sim


        anomaly_map_vls = []
        for layer in range(len(patch_tokens)):

            if layer != 3: # layer%2!=0:# (layer+1)//2!=0:   # layer != 3:   # 6 12 18 24 24_xor  选择输出的layer
                continue   #  vitl-14



            patch_tokens[layer] = patch_tokens[layer] @ self.clip_surgery.visual.proj
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(
                dim=-1, keepdim=True
            )
            anomaly_map_vl = 100.0 * patch_tokens[layer] @ self.text_feature_prototype
            B, L, C = anomaly_map_vl.shape
            H = int(np.sqrt(L))
            anomaly_map_vl = F.interpolate(
                anomaly_map_vl.permute(0, 2, 1).view(B, 2, H, H),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map_vl = torch.softmax(anomaly_map_vl, dim=1)
            anomaly_map_vl = (anomaly_map_vl[:, 1, :, :] - anomaly_map_vl[:, 0, :, :] + 1) / 2
            anomaly_map_vls.append(anomaly_map_vl)

        anomaly_map_vls = torch.mean(
            torch.stack(anomaly_map_vls, dim=0), dim=0
        ).unsqueeze(1)

        #----- 构建原型部分 -------------------
        # score_proto, map_proto = None, None
        # if self.text_feature_prototype is not None and self.p_v is not None:
        #     p_t = self.text_feature_prototype
        #     score_v = 1 - (image_features @ p_v)
        #     score_t = 1 - (image_features @ p_t)
        #     score_proto = (score_v + score_t) / 2
        #
        #     sim_patch_v = (patch_tokens @ p_v).mean(0)  # [L]
        #     sim_patch_t = (patch_tokens @ p_t).mean(0)  # [L]
        #     sim_patch = (sim_patch_v + sim_patch_t) / 2
        #     H = int(np.sqrt(sim_patch.shape[0]))
        #     map_proto = (1 - sim_patch.view(1, 1, H, H))
        #     map_proto = F.interpolate(map_proto, size=self.image_size, mode="bilinear", align_corners=True)
        #
        # if score_proto is not None:
        #     pred_score = anomaly_map_vls.view(B, -1).max(dim=1).values
        #     pred_score =  (pred_score +  score_proto  ) / 2
        #     pred_mask = (anomaly_map_vls + map_proto  ) / 2
        # else:
        #     pred_score = anomaly_map_vls.view(B, -1).max(dim=1).values
        #     pred_mask = anomaly_map_vls

        # anomaly_map_ret_all = (0.1*anomaly_map_ret + 0.8*anomaly_map_vls
        #                        )
        #

        # anomaly_map_ret_all =   anomaly_map_vls

        anomaly_map_ret_all = ( anomaly_map_ret + anomaly_map_vls  ) /2

        #
        pred_score = anomaly_map_ret_all.view(B, -1).max(dim=1).values + global_score

        # pred_score = 0.5 * anomaly_map_ret_all.view(B, -1).max(dim=1).values \
        #              + 0.5 * anomaly_map_ret_all.view(B, -1).mean(dim=1) \
        #              + global_score

            #k-shot
        # if self.shot:
        #     anomaly_map_ret_all = (anomaly_map_ret + anomaly_map_vls
        #         ) / 2
        #
        #     pred_score = anomaly_map_ret_all.view(B, -1).max(dim=1).values + global_score
        #     # k-shot
        # else:
        #
        #     anomaly_map_ret_all =  anomaly_map_vls
        #     pred_score = anomaly_map_ret_all.view(B, -1).max(dim=1).values

        try:
            del clip_transformed_image, image_features, patch_features, img_paths, _, global_score
            del patch_tokens, nt, pt, sims, sim
            del anomaly_map_ret, anomaly_map_vl, anomaly_map_vls
            del cosine_similarity_matrix, sim_max
        except NameError:
            pass

        # --- 清理 GPU 缓存 ---
        torch.cuda.empty_cache()

        return {
            "pred_score": pred_score.detach().cpu(),
            "pred_mask": anomaly_map_ret_all.detach().cpu(),
        }

    def run_clip_multiproto(self, img_paths, p_v_global=None, p_v_patch=None, p_t=None,cls_name=None):
        """
        CLIP-based anomaly detection branch.
        - If prototypes (p_v_global, p_v_patch, p_t) are provided, use them directly.
        - Otherwise fallback to stored normal-based features (self.normal_image_features / self.normal_patch_tokens).
        """

        clip_transformed_image = self.transform_clip(img_paths)

        with torch.no_grad():


                # self.text_prompts = clip.encode_text_with_prompt_ensemble_ad(
                #     self.clip_surgery, None, ["object"], self.tokenizer_clip, device
                # )

            # self.text_prompts, self.anomaly_text_proto, self.norm_text_proto = clip.encode_text_with_prompt_ensemble_ad(
            #         self.clip_surgery, None, cls_name[0], self.tokenizer_clip, self.device
            #     )
            #
            # self.text_feature_prototype = self.text_prompts["object"].to(self.device)

            image_features, patch_features = self.clip_surgery.encode_image(clip_transformed_image)

        # CLS token作为global embedding
        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # patch tokens解码
        patch_tokens = self.decoder(patch_features)

        # ------------------- Prototype-based path -------------------
        if (p_v_patch is not None) and (p_t is not None):
            # -------- Global residual score --------
            if p_v_global is not None:
                score_v = 1 - (image_features @ p_v_global)
            else:
                score_v = 1 - (image_features @ p_v_patch)  # fallback
            score_t = 1 - (image_features @ p_t)
            score_proto = (score_v + score_t) / 2  # [B]

            # -------- Patch residual map --------
            layer = 3  # 你之前选用的层
            pt = patch_tokens[layer]  # [B, L, C]
            pt = pt @ self.clip_surgery.visual.proj
            pt = pt / pt.norm(dim=-1, keepdim=True)

            sim_patch_v =100.0 * (pt @ p_v_patch)  # [B, L]
            sim_patch_t =100.0 * (pt @ p_t)# [B, L]
            sim_patch_t_an = 100.0 * (pt @ self.text_feature_prototype[:,1])
            # sim_patch = (sim_patch_v + sim_patch_t) / 2  # [B, L]

            H = int(np.sqrt(sim_patch_v.shape[1]))
            anomaly_map_v = 1 - sim_patch_v.view(-1, 1, H, H)
            anomaly_map_v = F.interpolate(anomaly_map_v, size=self.image_size,
                                        mode="bilinear", align_corners=True)
            anomaly_map_v = torch.softmax(anomaly_map_v, dim=1)


            anomaly_map_t = 1 - sim_patch_t.view(-1, 1, H, H)
            anomaly_map_t = F.interpolate(anomaly_map_t, size=self.image_size,
                                          mode="bilinear", align_corners=True)
            anomaly_map_t = torch.softmax(anomaly_map_t, dim=1)


            sim_patch_t_an = 1 - sim_patch_t_an.view(-1, 1, H, H)
            sim_patch_t_an = F.interpolate(sim_patch_t_an, size=self.image_size,
                                          mode="bilinear", align_corners=True)
            sim_patch_t_an = torch.softmax(sim_patch_t_an, dim=1)

            anomaly_map = (sim_patch_t_an+2-anomaly_map_t-anomaly_map_v) / 3


            # anomaly_map = 1 - sim_patch.view(-1, 1, H, H)
            # anomaly_map = F.interpolate(anomaly_map, size=self.image_size,
            #                             mode="bilinear", align_corners=True)

            anomaly_map_ret_all = anomaly_map
            pred_score = anomaly_map.view(anomaly_map.size(0), -1).max(dim=1).values + score_proto

        # ------------------- Fallback: normal-based path -------------------
        else:
            if self.shot:
                global_score = 1 - (image_features @ self.normal_image_features.transpose(-2, -1)).amax(dim=-1)
                score_cls = image_features @ self.text_feature_prototype
                score_cls = score_cls[0, 1] - score_cls[0, 0]

                self.score_cls = score_cls + global_score
                sims = []
                for i in range(len(patch_tokens)):
                    if i % 2 == 0:
                        continue
                    pt = patch_tokens[i]
                    B, L, C = pt.shape
                    pt = pt.contiguous().view(B, L, 1, C)
                    nt = self.normal_patch_tokens[i].reshape(1, -1, C).unsqueeze(0)
                    cosine_similarity_matrix = F.cosine_similarity(pt, nt, dim=-1)
                    sim_max, _ = torch.max(cosine_similarity_matrix, dim=2)
                    sims.append(sim_max)

                sim = torch.mean(torch.stack(sims, dim=0), dim=0)
                H = int(np.sqrt(L))
                sim = sim.view(B, 1, H, H)
                sim = F.interpolate(sim, size=self.image_size, mode="bilinear", align_corners=True)
                anomaly_map_ret = 1 - sim
                self.norm_anomaly = anomaly_map_ret

                # 语言
                anomaly_map_vls = []
                for layer in range(len(patch_tokens)):

                    selected_layers = [len(patch_tokens)-2,]

                    if layer not in selected_layers:  # 你指定的层
                        continue

                    pt = patch_tokens[layer] @ self.clip_surgery.visual.proj
                    pt = pt / pt.norm(dim=-1, keepdim=True)  # [B, L, D]
                    text_proto = self.text_feature_prototype
                    text_proto = text_proto / text_proto.norm(dim=-1, keepdim=True)  # [2, D] normal/abnormal

                    # 使用 cosine similarity
                    sim = F.cosine_similarity(pt.unsqueeze(2), text_proto.T.unsqueeze(0).unsqueeze(0), dim=-1)
                    # sim = pt @ self.text_feature_prototype   # 95.6 |       95.2 |
                    # sim: [B, L, 2]

                    B, L, C = sim.shape
                    H = int(np.sqrt(L))
                    anomaly_map_vl = F.interpolate(
                        sim.permute(0, 2, 1).view(B, C, H, H),
                        size=self.image_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                    anomaly_map_vl = torch.softmax(anomaly_map_vl, dim=1)
                    anomaly_map_vl = (anomaly_map_vl[:, 1] - anomaly_map_vl[:, 0] + 1) / 2
                    anomaly_map_vls.append(anomaly_map_vl)

                if len(anomaly_map_vls) > 0:
                    anomaly_map_vls = torch.mean(torch.stack(anomaly_map_vls, dim=0), dim=0).unsqueeze(1)

                # 不同计算方式   下面这个在zero shot 时性能更好， 上面在 4 shot 时性能更高
                # anomaly_map_vls = []
                # for layer in range(len(patch_tokens)):
                #
                #     if layer != 3:  # layer%2!=0:# (layer+1)//2!=0:   # layer != 3:   # 6 12 18 24 24_xor  选择输出的layer
                #         continue  # vitl-14
                #
                #     # if layer != 4:   # 3 6 9 9_xor 12 12_xor  vit-16
                #     #     continue
                #
                #     # if layer % 2 == 0:
                #     #     continue
                #
                #     patch_tokens[layer] = patch_tokens[layer] @ self.clip_surgery.visual.proj
                #     patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(
                #         dim=-1, keepdim=True
                #     )
                #     anomaly_map_vl = 100.0 * patch_tokens[layer] @ self.text_feature_prototype
                #     B, L, C = anomaly_map_vl.shape
                #     H = int(np.sqrt(L))
                #     anomaly_map_vl = F.interpolate(
                #         anomaly_map_vl.permute(0, 2, 1).view(B, 2, H, H),
                #         size=self.image_size,
                #         mode="bilinear",
                #         align_corners=True,
                #     )
                #     anomaly_map_vl = torch.softmax(anomaly_map_vl, dim=1)
                #     anomaly_map_vl = (anomaly_map_vl[:, 1, :, :] - anomaly_map_vl[:, 0, :, :] + 1
                #                       ) / 2
                #     anomaly_map_vls.append(anomaly_map_vl)
                # anomaly_map_vls = torch.mean(
                #     torch.stack(anomaly_map_vls, dim=0), dim=0
                # ).unsqueeze(1)

                anomaly_map_ret_all = (anomaly_map_ret + anomaly_map_vls) / 2
                pred_score = anomaly_map_ret_all.view(B, -1).max(dim=1).values + global_score + score_cls
            else:

                score_cls = image_features @ self.text_feature_prototype
                score_cls = score_cls[0, 1] - score_cls[0, 0]
                self.score_cls = score_cls

                anomaly_map_vls = []
                for layer in range(len(patch_tokens)):

                    selected_layers = [len(patch_tokens) - 2,]

                    if layer not in selected_layers:
                        continue

                    patch_tokens[layer] = patch_tokens[layer] @ self.clip_surgery.visual.proj
                    patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(
                        dim=-1, keepdim=True
                    )

                    text_proto = self.text_feature_prototype
                    # text_proto = text_proto / text_proto.norm(dim=-1, keepdim=True)  # [2, D] normal/abnormal

                    anomaly_map_vl = patch_tokens[layer] @ text_proto

                    # anomaly_map_vl =  patch_tokens[layer] @ self.text_feature_prototype    #  100

                    B, L, C = anomaly_map_vl.shape
                    H = int(np.sqrt(L))

                    # -------------------- image 相似特征提取 ⬇------------------------------------
                    # self.selected_feats = []
                    # k = 10
                    # map_vl = anomaly_map_vl.permute(0, 2, 1).view(B, 2, H, H)
                    # # map_vl = torch.softmax(map_vl, dim=1)
                    # map_vl = (map_vl[:, 1, :, :] - map_vl[:, 0, :, :] + 1 ) / 2
                    #
                    #
                    #
                    # weights = map_vl.view(B, -1)
                    # # weights = (weights)
                    #
                    # # weights = (weights - weights.min()) / (
                    # #             weights.max() - weights.min() + 1e-8)
                    #
                    # # weights = torch.softmax(weights, dim=1)
                    # prototypes_raw = select_anomaly_prototypes(
                    #     image_token, weights, M=100, K=50, threshold=0.8
                    # )
                    #
                    #
                    # # self.selected_feats = self.memory_bank.expand_with_memory(prototypes_raw,update_after=True)  # 62 71.8
                    #
                    # self.selected_feats = self.memory_bank.fuse_with_memory(    # 63.1  71.6
                    #     prototypes_raw,  #self.selected_feats,
                    #     alpha=0.3,  # 融合权重，可调    # 0.5   #  93.2  90.6
                    #     update_after=False  # 融合时不立即更新
                    # )
                    #
                    # # self.memory_bank.update(prototypes_raw)   # 这个逻辑不对，  slots 怎么都一样呢
                    #
                    # # self.selected_feats = prototypes_raw
                    # _, self.k, _ = self.selected_feats.shape
                    #
                    #
                    #
                    # # topk_vals, topk_idx = torch.topk(weights, k, dim=1)
                    # # for b in range(B):
                    # #     feats = image_token[b, topk_idx[b]]  # [k, C]
                    # #     self.selected_feats.append(feats)
                    # # self.selected_feats = torch.stack(self.selected_feats, dim=0)

                    # -------------------- image 相似特征提取 ------------------------------------


                    anomaly_map_vl = F.interpolate(
                        anomaly_map_vl.permute(0, 2, 1).view(B, 2, H, H),
                        size=self.image_size,
                        mode="bilinear",
                        align_corners=True,
                    )

                    anomaly_map_vl = torch.softmax(anomaly_map_vl, dim=1)

                    anomaly_map_vl = (anomaly_map_vl[:, 1, :, :] - anomaly_map_vl[:, 0, :, :] + 1
                                      ) / 2
                    anomaly_map_vls.append(anomaly_map_vl)
                anomaly_map_vls = torch.mean(
                    torch.stack(anomaly_map_vls, dim=0), dim=0
                ).unsqueeze(1)


                anomaly_map_ret_all =  anomaly_map_vls
                pred_score = anomaly_map_ret_all.view(B, -1).max(dim=1).values + score_cls



        # -------- 清理缓存 --------
        try:
            del clip_transformed_image, patch_features
        except NameError:
            pass

        torch.cuda.empty_cache()

        return {
            "pred_score": pred_score.detach().cpu(),
            "pred_mask": anomaly_map_ret_all.detach().cpu(),
        }


    def run_diffusion_prompts_batchs_multiproto(self, cls_names, img_tensors, mask_clip=None):

        # states = ["damage"]
        states = ["damage", "hole", "residue", "crack"]

        # prompts: list[list[str]]，每个样本一个 list
        prompts = [[f"A photo of a {cls_name} with {state}" for state in states] for cls_name in cls_names]

        # self.text_prompts_diff,self.state_id,self.object_id,self.norm_id  = clip.encode_text_with_prompt_ensemble_ad_diff_word(
        #     self.ldm_stable.text_encoder.to(self.device), [cls_names[0]], self.tokenizer, self.device)


        rgb_512 = img_tensors.to(self.device) # (B, 3, H, W)  .to(torch.float16)
        input_latent = encode_imgs(rgb_512, self.vae)  # (B, 4, 64, 64)

        _,_,latent_h,latent_w = input_latent.shape


        controller = AttentionStore()
        controller.step_store = controller.get_empty_store()
        controller.attention_store = {}

        controller.mask = mask_clip   # 添加 clip_mask 到attn

        times = [1,20,50,100,300]  # <<< 修改这里：你要跑的 step   # t越大 图像级指标越好，语义更好， t越小  外观像素级越好     t=100  90.2 |       89.1 |
        all_attn_finals = []  # <<< 修改这里：存放不同 t 的结果

        dtype = torch.float32

        for t in times:  # <<< 修改这里：循环不同的 t
            controller.reset()

            # noise 设计
            noise = torch.randn([1, 4, 64, 64]).to(self.device)  # noise 随机初始化
            noise = noise if dtype == torch.float32 else noise.half()
            # #
            # #
            #
            #---------------- clip map noise ------------------------

            #    stable diffusion nearest
            mask_latent = F.interpolate(mask_clip, size=(latent_h, latent_w), mode="bilinear", align_corners=False).to(self.device)

            mask_resized = mask_latent.squeeze().cpu().numpy()
            mask_resized = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min())
            mask_resized = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).to(self.device)

            alpha = 2
            # delta = alpha * mask_latent * torch.sign(input_latent)
            delta = alpha * mask_resized * torch.sign(input_latent)
            #
            #
            #

            #---------------   消融 noise---------------------------------------------

            noise = noise * delta

            # noise = noise

            #---------------   消融 noise---------------------------------------------

            latents_noisy = self.ldm_stable.scheduler.add_noise(
                input_latent, noise, torch.tensor(t, device=self.device)     # 92.9   |       91.2 |  time =20  alpha =2
                # t=100 | transistor |       64.9 |       66.2 |
            )

            # latents_noisy = self.ldm_stable.scheduler.add_noise(
            #     input_latent, delta, torch.tensor(t, device=self.device)      # 93.1  90.7   time =20
            #     # t=100 | transistor |       64.9 |       66.2 |
            # )  # t=1   | transistor |       54.6 |         71 |

            #---------------- clip map noise ------------------------



            # masked_latent = input_latent * delta
            #
            # latents_noisy = masked_latent

            # latents_noisy = self.ldm_stable.scheduler.add_noise(
            #     masked_latent, delta, torch.tensor(t, device=self.device)
            #     # t=100 | transistor |       64.9 |       66.2 |
            # )



            # stable diffusion inpainting

            # mask_latent = F.interpolate(
            #     mask_clip, size=(input_latent.shape[2], input_latent.shape[3]),
            #     mode="nearest"
            # ).to(device=input_latent.device, dtype=input_latent.dtype)
            #
            # mask_resized = (mask_latent - mask_latent.min()) / (mask_latent.max() - mask_latent.min())
            #
            # masked_latent = input_latent * (1 - mask_resized)
            # input_for_inpaint = torch.cat([input_latent, masked_latent, mask_resized], dim=1)
            #
            # latents_noisy = self.ldm_stable.scheduler.add_noise(
            #     input_for_inpaint, noise=torch.randn_like(input_for_inpaint),timesteps= torch.tensor(t, device=self.device)
            #
            # )


            # latents_noisy = input_latent

            # zero_noise = torch.zeros_like(input_latent)
            # latents_noisy = self.ldm_stable.scheduler.add_noise(
            #     input_latent, zero_noise, torch.tensor(t, device=self.device)    # t=100 | transistor |       52.4 |       69.7 |
            # )                                                                    # t=1   | transistor |       51.7 |       70.1 |


            #---------------------------------------------------------------------
            # with torch.no_grad():
            #     image_inv, x_t = run_and_display(
            #         [p[0] for p in prompts],
            #         controller,
            #         ldm_stable=self.ldm_stable,
            #         run_baseline=False,
            #         latent=latents_noisy,
            #         verbose=False,
            #         file_name=None,
            #         clip_image=None,   # self.normal_patch_tokens  (k,256,1024)
            #         text_prompt=self.text_prompts_diff,
            #         onlyimg=False,
            #         image_size=self.image_size
            #     )


            # #------------------------- k-shot 版本 --------------------------------------------
            with torch.no_grad():
                image_inv, x_t = run_and_display(
                    [p[0] for p in prompts],
                    controller,
                    ldm_stable=self.ldm_stable,
                    run_baseline=False,
                    latent=latents_noisy,
                    verbose=False,
                    file_name=None,
                    clip_image= self.selected_feats,                             #self.normal_patch_tokens,  # self.normal_patch_tokens  (k,256,1024)
                    text_prompt=self.text_prompts_diff,
                    onlyimg=True,
                    image_size=self.image_size
                )



            # del rgb_512, input_latent, latents_noisy, image_inv, x_t, img_tensors,mask_latent,mask_clip,

            # del rgb_512, input_latent, latents_noisy, image_inv, x_t, img_tensors,mask_clip

            torch.cuda.empty_cache()
            # ============= attention 提取部分 =============
            # cross_attention_maps, resolutions = aggregate_all_attention_batch_imagesize(
            #     prompts, controller, ("up", "mid", "down"), True, image_size=self.image_size
            # )
            #  U-net 中 up down 对应 解码 和编码么？ 注意力图有什么不同    down --> encoder   up--> decoder
            cross_attention_maps, resolutions = aggregate_all_attention_batch_imagesize(
                prompts, controller, ('up', "mid","down",), True, image_size=self.image_size   #( 'up',"mid","down",),  ( "mid","down",)
            )
            self_attention_maps, _ = aggregate_all_attention_batch_imagesize(
                prompts, controller, ("up", "mid",'down'), False, image_size=self.image_size  #("up", "mid", "down")
            )

            del _
            torch.cuda.empty_cache()

            B = len(prompts)
            latent_res = self.image_size // 8

            if latent_res == 64:
                # weight =  [0,2,1,0.6] #[0, 1, 1, 0]
                # weight = [0, 2, 1, 0]   # stable diffusion v2-1

                # weight =   [0, 2, 1, 0]  #  [0, 2, 1, 0.6]   wood 99.6   94.5 # stable diffusion v1-5
                weight =   [1, 1, 1, 1]  #  [0, 2, 1, 0.6]   wood 99.6   94.5 # stable diffusion v1-5



            elif latent_res == 32:

                # weight =  [0, 0, 1, 0]

                weight = [1, 1, 1, 1]
                # weight =

                # weight = [0.2, 0.4, 0.3, 0.1]
                # weight = [0.3, 0.5, 0.1, 0.1]
            else:
                weight = [1.0 / len(resolutions)] * len(resolutions)

            batch_attn_maps = []

            latent_res = 32
            for b in range(B):
                out_atts = []
                out_atts_norm = []
                out_obj = []

                word_len = len(prompts[b][0].split(" "))
                embs_len = len(self.tokenizer.encode(prompts[b][0])) - 2

                for l, res in enumerate(resolutions):
                    cross_att = cross_attention_maps[l][b]

                    try:
                        if prompts[b][0].split(" ")[3 + word_len].endswith("ing"):
                            token_ids = [3 + embs_len, 5 + embs_len]
                        else:
                            token_ids = [3 + embs_len]
                    except:
                        token_ids = [3 + embs_len]

                    # cross_att = cross_att[:, :, token_ids].mean(-1)   #(4,4)
                    #mean

                    #---------------------------- img cross map -------------------------------------
                    # last5 = list(range(cross_att.size(2) - 10, cross_att.size(2)))

                    # last5 = list(range(cross_att.size(2) - self.k, cross_att.size(2)))
                    #
                    #
                    # # all_ids = self.state_id + last5
                    # # cross_att_state = cross_att[:, :, all_ids].mean(-1)   # 是取均值还是max
                    #
                    # cross_img_state = cross_att[:,:,last5].mean(-1)
                    #
                    # if res != latent_res:
                    #     cross_img_state = F.interpolate(
                    #         cross_img_state.unsqueeze(0).unsqueeze(0),
                    #         size=(latent_res, latent_res),
                    #         mode="bilinear",
                    #         align_corners=False
                    #     ).squeeze()

                    #---------------------------- img cross map -------------------------------------

                    cross_att_state = cross_att[:, :, self.state_id].mean(-1)   # 是取均值还是max


                    # max
                    # state_atts = cross_att[:, :, self.state_id]
                    # token_scores = state_atts.mean(dim=(0, 1))
                    # best_idx = token_scores.argmax().item()
                    # best_token_id = self.state_id[best_idx]
                    # cross_att_state = cross_att[:, :, best_token_id]

                    # cross_att_norm = cross_att[:, :, self.norm_id].mean(-1)
                    # cross_att_obj = cross_att[:, :, self.object_id].mean(-1)
                    cross_att = cross_att_state     #state   50.2 |       87.7   # normal      50.7 |       86.1 |   # obj 会有cls 是 nan metal nut  transistor

                    # save_attention_map_gradcam(cross_att_state,
                    #                            save_path='/home/data/zjp/code/VGDiffZero/vis_attention/Cres16.png',
                    #                            blur=False)

                    # cross_att_combined = torch.stack([cross_att_state, cross_att_norm], dim=0)

                    if res != latent_res:
                        cross_att = F.interpolate(
                            cross_att.unsqueeze(0).unsqueeze(0),
                            size=(latent_res, latent_res),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze()

                    # ---------------------------- img cross map -------------------------------------

                    # cross_att = cross_att * (1 + 5 * cross_img_state)    #   整体 41   |       46.9 |

                    # weights = torch.sigmoid(cross_img_state)    #    transistor     54.7 |       70.9 |
                    # cross_att = cross_att  *  weights

                    # weights = torch.softmax(cross_img_state, dim=-1)   #  transistor    55.4 |       71.3 |   #  mean  总 82.8 |       86.3 |
                    # cross_att = cross_att * weights

                    # cross_img_state = cross_img_state / (cross_img_state.norm(dim=-1, keepdim=True) + 1e-6)  #
                    # cross_att = cross_att * cross_img_state

                    #   -----------------  image diffusion -------------
                    # weights = cross_img_state
                    # cross_att = cross_att + weights

                    # cross_att = (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min() + 1e-8)   # cross_att = torch.softmax(cross_att.flatten(), dim=0).view_as(cross_att)  后面这个可能更平滑些
                    # weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)   # cross_att = torch.softmax(cross_att.flatten(), dim=0).view_as(cross_att)  后面这个可能更平滑些
                    #
                    # cross_att = (cross_att + weights) /2




                    #   -----------------  image diffusion -------------

                    # cross_img_state = (cross_img_state - cross_img_state.min()) / (cross_img_state.max() - cross_img_state.min() + 1e-8)
                    # self_img_state_map = cross_img_state.view(latent_res, latent_res)


                    # image_state 作为 selfmap

                                              # transistor    55.4 |       71.3 |   #  mean  总 82.8 |       86.3 |
                    # cross_att = (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min() + 1e-8)


                    # ---------------------------- img cross map -------------------------------------



                    cross_att = (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min() + 1e-8)   # cross_att = torch.softmax(cross_att.flatten(), dim=0).view_as(cross_att)  后面这个可能更平滑些

                    out_atts.append(cross_att * weight[l])   # list 4  (64,64)     # 16 更好  64 down  不好 ， up 效果好

                    # obj
                    # if res != latent_res:
                    #     cross_att_obj = F.interpolate(
                    #         cross_att_obj.unsqueeze(0).unsqueeze(0),
                    #         size=(latent_res, latent_res),
                    #         mode="bilinear",
                    #         align_corners=False
                    #     ).squeeze()
                    #
                    # cross_att_obj = (cross_att_obj - cross_att_obj.min()) / (cross_att_obj.max() - cross_att_obj.min() + 1e-8)
                    # out_obj.append(cross_att_obj * weight[l])  # list 4  (64,64)
                    #
                    # if res != latent_res:
                    #     cross_att_norm = F.interpolate(
                    #         cross_att_norm.unsqueeze(0).unsqueeze(0),
                    #         size=(latent_res, latent_res),
                    #         mode="bilinear",
                    #         align_corners=False
                    #     ).squeeze()
                    #
                    # cross_att_norm = (cross_att_norm - cross_att_norm.min()) / (
                    #             cross_att_norm.max() - cross_att_norm.min() + 1e-8)
                    # out_atts_norm.append(cross_att_norm * weight[l])  # list 4  (64,64)

                # att_map = torch.stack(out_atts).sum(0).view(latent_res , latent_res)

                cross_att_map = torch.stack(out_atts).sum(0).view(latent_res * latent_res, 1)

                #
                # if  self.shot:
                #     cross_att_map_2d = cross_att_map.view(latent_res, latent_res)
                #
                #     alpha = 1  # 控制引导强度   #89  92
                #     cross_att_map_enhanced = cross_att_map_2d * (1 + alpha * (cross_att_map_2d - self.normal_prior) )   #(cross_att_map_2d + self.normal_prior)
                #
                #     # k = 5.0  # 控制自适应敏感度
                #     # alpha = torch.tanh(k * (cross_att_map_2d.mean() - self.normal_prior.mean()))
                #     # cross_att_map_enhanced = cross_att_map_2d * (1 + alpha * self.normal_prior)   # 90.4 91.9     # 87.6 |       92.1 |
                #
                #     cross_att_map = cross_att_map_enhanced.view(latent_res * latent_res, 1)   #  cable  89.3 |       91.6 |   4 shot

                # att_map = torch.stack(out_atts).sum(0).view(latent_res , latent_res)       # 90.8  90.6     no diffusion shot


              #-------------------  单层 self attention 融合 ------------------------------
                # self_att = self_attention_maps[-1][b].detach()
                # self_att = self_attention_maps[-2][b].detach()
                # self_att_map = self_att.view(latent_res * latent_res, latent_res * latent_res)   #加selfattention 62.8 86.2 有助于px   # selfattention为什么会都这么低呢
                # -------------------  单层 self attention 融合 ------------------------------

                #-------------------  多层 self attention 融合 ------------------------------
                # weiht_self=[0,0,1,0]    stable diffusion v2-1

                weiht_self=[0,0,1,0]

                # weiht_self=[1,1,1,1]



                out_selfatts = []
                for l, res in enumerate(resolutions):
                    # if l in [2,3]:
                    self_att = self_attention_maps[l][b]
                    self_att_32 = self_att.view(res * res, res * res)
                    if res != latent_res:
                        self_att_32 = F.interpolate(
                            self_att_32.unsqueeze(0).unsqueeze(0),
                            size=(latent_res * latent_res, latent_res * latent_res),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze()

                    self_att_32 = (self_att_32 - self_att_32.min()) / (self_att_32.max() - self_att_32.min() + 1e-8)
                    out_selfatts.append(self_att_32 * weiht_self[l])

                self_att_map = torch.stack(out_selfatts).sum(0).view(latent_res * latent_res, latent_res * latent_res)

                # #----------------------------  多层selfattn ----------------------------------
                #
                #
                # #------------------- self attention 融合 ------------------------------


                #  clip_map guide result

                mask_clip_latent  = F.interpolate(mask_clip, size=(latent_res, latent_res), mode="bilinear",
                                            align_corners=False).to(self.device)

                mask_resized = mask_clip_latent.squeeze().cpu().numpy()
                mask_resized = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min())
                mask_resized = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).to(self.device)
                mask_clip_flat = mask_resized.flatten(start_dim=2).squeeze(0).squeeze(0)

                self_modulated = self_att_map * mask_clip_flat.unsqueeze(0)
                cross_modulated = cross_att_map * mask_clip_flat.unsqueeze(1)

                fusion_raw = torch.matmul(self_modulated, cross_modulated)  # [1024, 1]   # 93  92

                fusion_base = torch.matmul(self_att_map, cross_att_map)

                ## 消融  fusion---------
                fusion_raw = fusion_base + 0.7 * (fusion_raw - fusion_base)   # 0.5  93 92  0.8  93 92.3
                # fusion_raw = fusion_base


                att_map = fusion_raw.view(latent_res, latent_res)


                # att_map = self_modulated.view(latent_res, latent_res)




                #  clip_map guide result




                # att_map = torch.matmul(self_att_map, cross_att_map).view(latent_res, latent_res)
                batch_attn_maps.append(att_map)  # list 0  (64,64)

            attn_final = torch.stack(batch_attn_maps, dim=0)  # (B, H, W)
            attn_final = attn_final.unsqueeze(1)  # (B,1,H,W)
            attn_final = F.interpolate(
                attn_final, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )
            # attn_final = torch.softmax(attn_final,dim=1)   #可能会导致 50 50   #


            all_attn_finals.append(attn_final)  # <<< 修改这里：保存每个 t 的结果

        # ========= 融合不同 t 的结果 =========
        attn_final_fused = torch.mean(torch.stack(all_attn_finals, dim=0), dim=0)  # <<< 修改这里：结果融合

        # pred_score = attn_final_fused.view(B, -1).max(dim=1).values

        # 像素级对比
        # img_array = np.transpose(image_inv, (0, 3, 1, 2))
        # img_tensor = torch.from_numpy(img_array).float()
        # img_tensor = img_tensor / 255.0
        # Dp = ((img_tensors - img_tensor.to(self.device)) ** 2).sum(dim=1, keepdim=True)
        # # Dp = torch.abs(img_tensors - img_tensor.to(self.device)).sum(dim=1, keepdim=True)
        # flat_Dp = Dp.view(Dp.size(0), Dp.size(1), -1)
        # max_dp, _ = flat_Dp.max(dim=-1, keepdim=True)
        # flat = attn_final_fused.view(attn_final_fused.size(0), attn_final_fused.size(1), -1)
        # max_val, _ = flat.max(dim=-1, keepdim=True)
        #
        # attn_final_fused =  Dp + attn_final_fused
        # 像素级对比


        # attn_final_fused = attn_final_fused.float()
        # save_batch_attention_gradcam(attn_final_fused,
        #                              save_dir='/home/data/zjp/code/VGDiffZero/vis_attention/attn_final_fused')

        if self.shot:
            attn_final_fused = (attn_final_fused - attn_final_fused.min()) / (
                    attn_final_fused.max() - attn_final_fused.min() + 1e-8)

            attn_final_fused = (attn_final_fused + self.norm_anomaly) / 2   #  92.9 |       95.9 |

            flat = attn_final_fused.view(attn_final_fused.size(0), attn_final_fused.size(1), -1)   #  shot=2  95.5 |       96.1 |
            max_val, _ = flat.max(dim=-1, keepdim=True)

            median_val = flat.median(dim=-1, keepdim=True).values

            pred_score = 1.0 - (median_val / (max_val + 1e-8)).view(-1) + self.score_cls

            # pred_score = attn_final_fused.view(B, -1).max(dim=1).values + self.score_cls

        else:

            attn_final_fused = (attn_final_fused - attn_final_fused.min()) / (
                    attn_final_fused.max() - attn_final_fused.min() + 1e-8)
            # 计算 pre score

            flat = attn_final_fused.view(attn_final_fused.size(0), attn_final_fused.size(1), -1)
            max_val, _ = flat.max(dim=-1, keepdim=True)

            median_val = flat.median(dim=-1, keepdim=True).values

            pred_score = 1.0 - (median_val / (max_val + 1e-8)).view(-1) + self.score_cls



        # pred_score = 1.0 - (median_val / (max_val + 1e-8)).view(-1)    # mean 或 median 效果在mvtec 上一样

        # mean_val = flat.mean(dim=-1, keepdim=True)
        # pred_score = 1.0 - (mean_val / (max_val + 1e-8)).view(-1)

        # shot
        # attn_final_fused = (attn_final_fused + self.norm_anomaly)/ 2
        # pred_score = attn_final_fused.view(B, -1).max(dim=1).values + self.globalscore + self.score_cls

        results_diff = {
            "pred_score": pred_score.detach().cpu(),  #pred_score.detach().cpu()
            "pred_mask": attn_final_fused.detach().cpu(),
        }


        del attn_final, cross_attention_maps, self_attention_maps,batch_attn_maps,pred_score
        torch.cuda.empty_cache()

        return results_diff






    # normal k-shot

    # ---------- 主函数：构建/更新 多模态视觉原型 ----------
    def aggregate_visual_prototypes(
            self,
            few_shot_samples=None,
            test_image=None,
            text_prototype=None,
            device=None,
            mode="auto",
            method="topk",  # choices: "topk", "kmeans", "sinkhorn"
            K=5,  # for kmeans centroids
            top_k=64,  # for top-k pooling
            sinkhorn_iters=40,
            use_patch_layer=3,  # which patch layer to use in self.decoder(...) (align with run_clip)
            normalize=True,
            update_cache=False,  # whether to append high-confidence samples to self.normal cache
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
            device = next(self.clip_surgery.parameters()).device if hasattr(self.clip_surgery,
                                                                            "parameters") else torch.device("cuda")

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
                        self.normal_image_features = torch.cat(
                            [self.normal_image_features, global_feats.detach().cpu()], dim=0)
                    if not hasattr(self, "normal_patch_tokens") or self.normal_patch_tokens is None:
                        self.normal_patch_tokens = tokens_proj.detach().cpu()
                    else:
                        self.normal_patch_tokens = torch.cat(
                            [self.normal_patch_tokens.cpu(), tokens_proj.detach().cpu()], dim=0)
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
    def setup(self, data: dict, re_seg=True) -> None:

        few_shot_samples = data.get("few_shot_samples")
        self.class_name = data.get("dataset_category")
        # image_paths = data.get("image_path")

        self.shot = len(few_shot_samples)

        # ret = self.aggregate_visual_prototypes(few_shot_samples=few_shot_samples,
        #                           text_prototype=self.text_prompts["object"][:,0],  # normal text proto
        #                           method="topk", top_k=64,
        #                           device=self.device,
        #                           update_cache=False)
        #
        # self.p_v_patch = ret["p_v_patch"]
        # self.p_v_global = ret["p_v_global"]


        clip_transformed_normal_image = self.transform_clip(few_shot_samples).to(
            self.device
        )

        with torch.no_grad():
            self.normal_image_features, self.normal_patch_tokens = (
                self.clip_surgery.encode_image(
                    clip_transformed_normal_image
                )
            )
            self.normal_image_features = self.normal_image_features[:, 0, :]
            self.normal_image_features = (
                self.normal_image_features / self.normal_image_features.norm()
            )    #（k，768）

            self.normal_patch_tokens = self.decoder(self.normal_patch_tokens)



        #-----  diffusion shot--------------
        # states = ["damage", "hole", "residue", "crack"]
        #
        # # prompts: list[list[str]]，每个样本一个 list
        # prompts = [[f"A photo of a {cls_name} with {state}" for state in states] for cls_name in [self.class_name]]
        #
        # normal_cross_maps = []
        # with torch.no_grad():
        #     for img in few_shot_samples:
        #         img = img.unsqueeze(0)
        #         rgb_512 = img.to(self.device)
        #         input_latent = encode_imgs(rgb_512, self.vae)
        #         _, _, latent_h, latent_w = input_latent.shape
        #         controller = AttentionStore()
        #         controller.step_store = controller.get_empty_store()
        #         controller.attention_store = {}
        #         controller.mask = None
        #
        #         t = 20
        #         # noise = torch.randn([1, 4, 64, 64]).to(self.device)  # noise 随机初始化
        #         # dtype = torch.float32
        #         # noise = noise if dtype == torch.float32 else noise.half()
        #         # latents_noisy = self.ldm_stable.scheduler.add_noise(
        #         #     input_latent, noise, torch.tensor(t, device=self.device)  # 92.9   |       91.2 |  time =20  alpha =2
        #         #     # t=100 | transistor |       64.9 |       66.2 |
        #         # )
        #         latents_noisy = input_latent
        #         with torch.no_grad():
        #             image_inv, x_t = run_and_display(
        #                 [p[0] for p in prompts],
        #                 controller,
        #                 ldm_stable=self.ldm_stable,
        #                 run_baseline=False,
        #                 latent=latents_noisy,
        #                 verbose=False,
        #                 file_name=None,
        #                 clip_image=self.selected_feats,  # self.normal_patch_tokens,  # self.normal_patch_tokens  (k,256,1024)
        #                 text_prompt=self.text_prompts_diff,
        #                 onlyimg=True,
        #                 image_size=self.image_size
        #             )
        #         del rgb_512, input_latent, latents_noisy, image_inv, x_t,
        #         torch.cuda.empty_cache()
        #
        #         cross_attention_maps, resolutions = aggregate_all_attention_batch_imagesize(
        #             prompts, controller, ('up', "mid", "down",), True, image_size=self.image_size
        #             # ( 'up',"mid","down",),  ( "mid","down",)
        #         )
        #         latent_res = 32
        #         out_atts = []
        #         weight = [0, 2, 1, 0]
        #         for l, res in enumerate(resolutions):
        #             cross_att = cross_attention_maps[l][0]
        #             cross_att_state = cross_att[:, :, self.state_id].mean(-1)
        #             if res != latent_res:
        #                 cross_att_state = F.interpolate(
        #                     cross_att_state.unsqueeze(0).unsqueeze(0),
        #                     size=(latent_res, latent_res),
        #                     mode="bilinear",
        #                     align_corners=False
        #                 ).squeeze()
        #             cross_att_state = (cross_att_state - cross_att_state.min()) / (
        #                         cross_att_state.max() - cross_att_state.min() + 1e-8)
        #             out_atts.append(cross_att_state * weight[l])
        #         cross_att_map_normal = torch.stack(out_atts).sum(0)
        #         normal_cross_maps.append(cross_att_map_normal)
        #
        # normal_prior = torch.mean(torch.stack(normal_cross_maps, dim=0), dim=0)  # (latent_res, latent_res)
        # normal_prior = normal_prior / (normal_prior.max() + 1e-8)
        #
        # self.normal_prior = normal_prior



        # diffusion visual normal shot
        # self.few_shot_samples = few_shot_samples
        #
        # rgb_512_norm = few_shot_samples.to(self.device)  # (B, 3, H, W)
        # input_latent_shot = encode_imgs(rgb_512_norm, self.vae)




if __name__ == "__main__":
    device = 'cuda:3'

    model = CLIP_Diffusion(device=device)

