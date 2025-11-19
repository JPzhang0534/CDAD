import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DDIMScheduler

from transformers import BlipProcessor, BlipForConditionalGeneration,CLIPImageProcessor, CLIPTextModel, CLIPVisionModel

from Diffusion_CLIP import (encode_imgs,split_text,split_sentences,AttentionStore, aggregate_all_attention_batch,aggregate_all_attention_batch_imagesize,
                            sd_to_clip_img,run_and_display,aggregate_all_attention,Gen_bbox_single,Gen_bbox_all,heatmap_img)

from segment_anything import sam_model_registry, SamPredictor


import cv2
import numpy as np

from PIL import Image

import clip
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC

MY_TOKEN = None  # ‘’
LOW_RESOURCE = False
NUM_DDIM_STEPS = 1
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

from clip_prompt import encode_text_with_prompt_ensemble

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
    def __init__(self,image_size,SD_path='/home/data/zjp/pre-trained_model/stable_diffusion',
                 clip_path= '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP-vit-large-14',
                 blip_path='/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/BLIP/blip-image-captioning-large',
                 device='cuda:2',):
        # 加载视CLIP视觉model
        super().__init__()

        self.device = device

        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            SD_path, ).to(device)

        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            SD_path, safety_checker=None).to(device)

        self.scheduler = DDIMScheduler.from_pretrained(
            SD_path, subfolder="scheduler")
         #------------------  加速 -------------------------------------
        self.ldm_stable.scheduler.set_timesteps(20)

        self.vae = self.ldm_stable.vae.to(device)
        self.tokenizer = self.ldm_stable.tokenizer


        # 加载blip
        # self.blip_processor = BlipProcessor.from_pretrained(blip_path)
        # self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_path).to(device)


        # ---todo： 添加CLIP surgery相关分支

        #
        # self.clip_surgery, self.clip_preprocess = clip.load("CS-ViT-B/16", device=device)
        self.clip_surgery, self.clip_preprocess = clip.load("CS-ViT-L/14", device=device)
        self.clip_surgery.eval()

        # from transformers import CLIPTokenizer
        # tokenizer_path = "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP-vit-large-14"
        # self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

        tokenizer = open_clip.get_tokenizer('ViT-B-32')

        with (torch.no_grad()):
            self.text_prompts = clip.encode_text_with_prompt_ensemble_ad(
                self.clip_surgery, ["object"], tokenizer, device
            )

            # self.text_prompts = clip.encode_text_with_prompt_ensemble_ad(
            #     self.ldm_stable.text_encoder, ["object"], self.tokenizer, device
            # )

            self.text_prompts_diff = clip.encode_text_with_prompt_ensemble_ad_diff(
                self.ldm_stable.text_encoder.to(device),["object"], self.tokenizer, device)

        from torchvision.transforms import v2
        self.transform_clip = v2.Compose(
            [
                v2.Resize((image_size, image_size)),
                v2.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )

        # self.transform_clip = v2.Compose(
        #     [
        #         v2.Resize((224, 224)),
        #         v2.Normalize(
        #             mean=(0.48145466, 0.4578275, 0.40821073),
        #             std=(0.26862954, 0.26130258, 0.27577711),
        #         ),
        #     ],
        # )

        self.image_size = image_size

        #----------
        #
        # sam_checkpoint = "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/sam/sam_vit_h_4b8939.pth"
        # model_type = "vit_h"
        # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.to(device=device)
        # self.sam_predictor = SamPredictor(sam)


        self.decoder = LinearLayer()

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

    def run_clip(self,img_paths,):

        clip_transformed_image = self.transform_clip(img_paths)

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

            if layer != 3: # layer%2!=0:# (layer+1)//2!=0:   # layer != 3:   # 6 12 18 24 24_xor  选择输出的layer
                continue   #  vitl-14

            # if layer != 4:   # 3 6 9 9_xor 12 12_xor  vit-16
            #     continue

            # if layer % 2 == 0:
            #     continue

            patch_tokens[layer] = patch_tokens[layer] @ self.clip_surgery.visual.proj
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(
                dim=-1, keepdim=True
            )
            anomaly_map_vl = 100.0 * patch_tokens[layer] @ text_features
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
            del clip_transformed_image, image_features, patch_features, text_features, img_paths, _, global_score
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

        states = ["damage"]

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

        times = [1]  # <<< 修改这里：你要跑的 step
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
                    onlyimg=False,
                    image_size=self.image_size
                )
            # del rgb_512, input_latent, latents_noisy, image_inv, x_t, img_tensors,mask_latent,mask_clip,clip_noise

            del rgb_512, input_latent, latents_noisy, image_inv, x_t, img_tensors,mask_clip

            torch.cuda.empty_cache()
            # ============= attention 提取部分 =============
            cross_attention_maps, resolutions = aggregate_all_attention_batch_imagesize(
                prompts, controller, ("up", "mid", "down"), True, image_size=self.image_size
            )
            self_attention_maps, _ = aggregate_all_attention_batch_imagesize(
                prompts, controller, ("up", "mid", "down"), False, image_size=self.image_size
            )

            del controller, _
            torch.cuda.empty_cache()

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

        pred_score = attn_final_fused.view(B, -1).max(dim=1).values

        results_diff = {
            "pred_score": pred_score.detach().cpu(),
            "pred_mask": attn_final.detach().cpu(),
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
            anomaly_map_vl = 100.0 * patch_tokens[layer] @ text_features
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

    def forward(self, img_paths,img_data, text_data, cls_name, use_sam=False):


        # clip 分支代码
        # results_clip = self.run_clip(img_paths)
        # #
        # results = results_clip
        # 多 batchs
        results_diff = self.run_diffusion_prompts_batchs(cls_name, img_paths, mask_clip=results_clip['pred_mask'])

        results_diff = self.run_diffusion_prompts_batchs(cls_name, img_paths)

        # results = results_diff
        # # # return results_diff
        # #
        # #
        results = {
            key: (0.1 * results_diff[key] + 0.9 * results_clip[key].cpu()) if key == 'pred_score'
            else (0.7 * results_diff[key] + 0.3 * results_clip[key].cpu())
            for key in results_diff
        }   #
        # del img_paths, text_data,results_diff,results_clip
        # torch.cuda.empty_cache()


        # results = {
        #     key: (0.1 * results_diff[key] + 0.9 * results_clip[key]) if key == 'pred_score'
        #     else (0.4 * results_diff[key] + 0.6 * results_clip[key])
        #     for key in results_diff
        # }



        return results

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




    # normal k-shot

    def setup(self, data: dict, re_seg=True) -> None:

        few_shot_samples = data.get("few_shot_samples")
        self.class_name = data.get("dataset_category")
        image_paths = data.get("image_path")

        self.shot = len(few_shot_samples)

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
            )

            self.normal_patch_tokens = self.decoder(self.normal_patch_tokens)


if __name__ == "__main__":
    device = 'cuda:3'

    model = CLIP_Diffusion(device=device)

