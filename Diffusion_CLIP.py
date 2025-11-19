from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
# import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import torch.nn as nn
import re
import cv2
import matplotlib.pyplot as plt

MY_TOKEN = None  # ‘’
LOW_RESOURCE = False    # 控制 context = torch.cat([uncond_embeddings_, text_embeddings])  False 只保留text_embeddings 对应的特征
NUM_DDIM_STEPS = 1
GUIDANCE_SCALE = 7.5   # 89.3 |       88.3 |
# GUIDANCE_SCALE = 5   # 89.3 |       88.3 |
MAX_NUM_WORDS = 77

# 设置所有随机种子
def set_seed(seed=42):
    # torch.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 固定随机性
set_seed(4307)    # 设置随机数


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class LocalBlend:

    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1 - int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:

            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2,
                 th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0
        self.th = th


class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)

                # attn_uncond, attn_cond = attn[:h // 2], attn[h // 2:]   # 为什么这部分用差值不行呢  是text_pormpt_diff 生成 两者区别不大么 mean
                # diff =  attn_cond

                #  原来
                # attn = self.forward(attn, is_cross, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    # def forward(self, attn, is_cross: bool, place_in_unet: str):
    #     key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
    #     if attn.shape[1] <= 64 ** 2:  # avoid memory overhead
    #         # self.step_store[key].append(attn.cpu())
    #         self.step_store[key].append(attn.detach().to(dtype=torch.float16).clone())
    #
    #     return attn

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        import torch.nn.functional as F

        # 消融
        # self.mask = None

        if is_cross:
        #
            if self.mask is not None:
                Q = attn.shape[1]
                latent_hw = int(Q ** 0.5)

                # mask_resized = F.interpolate(self.mask, size=(latent_hw, latent_hw), mode="nearest")      # 90.1 |       85.5 |   softmax(a) * b

                mask_resized = F.interpolate(self.mask, size=(latent_hw, latent_hw), mode="bilinear")

                img = mask_resized.squeeze().cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())
                mask_resized = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

                # mask_resized = (mask_resized > mask_resized.median()).float()
                # mask_resized = mask_resized + 1e-6

                th = mask_resized.median()
                # th = 0.7
                mask_resized = torch.where(mask_resized > th,   2 * torch.ones_like(mask_resized),         mask_resized)

                weight = mask_resized.flatten(2)
                weight = weight.squeeze(1)
                weight = weight.unsqueeze(-1)
                weight = weight.expand(attn.shape[0], attn.shape[1], attn.shape[2]).to(attn.device)

                attn = attn * (1 + 3 * weight)

                del img, weight, mask_resized
                torch.cuda.empty_cache()

            else:
                attn = attn

            # weight = weight / (weight.norm(dim=-1, keepdim=True) + 1e-6)
            # attn = attn  * weight

            # Step5: renormalize
            # attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        #
        #     pass
        else:
        #
            if self.mask is not None:
                mask_resized = F.interpolate(self.mask, size=(int(attn.shape[-1] ** 0.5), int(attn.shape[-1] ** 0.5)), mode="bilinear")

                img = mask_resized.squeeze().cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())
                mask_resized = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                mask_resized = 1 - mask_resized

                # mask_resized = (mask_resized > mask_resized.median()).float()
                # mask_resized = mask_resized+ 1e-6

                th = mask_resized.median()   # transistor |       52.1 |       72.2 |
                # th = 0.7
                mask_resized = torch.where(mask_resized > th,
                                       torch.ones_like(mask_resized),
                                       mask_resized)

                weight = mask_resized.flatten(2)
                weight = weight.to(attn.device)
                weight = weight.expand(attn.shape[0], attn.shape[1], -1)

                # attn = attn * (1 + 3 * weight)      #   91.7   |       90.5   |

                weight = weight / (weight.norm(dim=-1, keepdim=True) + 1e-6)
                attn = attn * weight

                del img, weight, mask_resized
                torch.cuda.empty_cache()

            else:
                attn = attn

        attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)

        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        if attn.shape[1] <= 64 ** 2:
            self.step_store[key].append(attn.detach().to(dtype=torch.float16).clone())



        # ========================

        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        #------------------------
        for key in self.attention_store:
            self.attention_store[key] = [x for x in self.attention_store[key]]

        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    pil_img = ptp_utils.view_images(np.stack(images, axis=0))
    return pil_img


def save_noun_cross_attenton(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,
                             noun_idx: List[int] = [], image_id=None):
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    save_ts = attention_maps
    torch.save(save_ts, f'./outputs/attn_db/{image_id}.pt')


def save_torch_tensor(ts, path):
    # if not osp.exists(path):
    torch.save(ts, path)


def save_multi_scale_attention_map(attention_store: AttentionStore, tag_id=None, idx=None):
    if not osp.exists(f'./outputs/attn_db/{tag_id}/cross16_{idx}.pt'):
        cross_16 = aggregate_attention(attention_store, 16, ['down', 'up'], is_cross=True, select=0)
        save_torch_tensor(cross_16, f'./outputs/attn_db/{tag_id}/cross16_{idx}.pt')

    if not osp.exists(f'./outputs/attn_db/{tag_id}/cross32_{idx}.pt'):
        cross_32 = aggregate_attention(attention_store, 32, ['down', 'up'], is_cross=True, select=0)
        save_torch_tensor(cross_32, f'./outputs/attn_db/{tag_id}/cross32_{idx}.pt')

    if not osp.exists(f'./outputs/attn_db/{tag_id}/cross64_{idx}.pt'):
        cross_64 = aggregate_attention(attention_store, 64, ['down', 'up'], is_cross=True, select=0)
        save_torch_tensor(cross_64, f'./outputs/attn_db/{tag_id}/cross64_{idx}.pt')

    if not osp.exists(f'./outputs/attn_db/{tag_id}/self_16.pt'):
        self_16 = aggregate_attention(attention_store, 16, ['down', 'up'], is_cross=False, select=0)
        save_torch_tensor(self_16, f'./outputs/attn_db/{tag_id}/self_16.pt')

    if not osp.exists(f'./outputs/attn_db/{tag_id}/self_32.pt'):
        self_32 = aggregate_attention(attention_store, 32, ['down', 'up'], is_cross=False, select=0)
        save_torch_tensor(self_32, f'./outputs/attn_db/{tag_id}/self_32.pt')
    #
    if not osp.exists(f'./outputs/attn_db/{tag_id}/self_64.pt'):
        self_64 = aggregate_attention(attention_store, 64, ['down', 'up'], is_cross=False, select=0)
        save_torch_tensor(self_64, f'./outputs/attn_db/{tag_id}/self_64.pt')


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis].repeat(3, axis=2)
    else:
        image = image_path
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class NullInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output  # noise？
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)  # 先得到初始隐变量对应的噪声
            latent = self.next_step(noise_pred, t, latent)  #
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        # bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                # bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            # for j in range(j + 1, num_inner_steps):
            #     bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        # bar.close()
        return uncond_embeddings_list

    def invert(self, image_path: str, prompt: str, offsets=(0, 0, 0, 0), num_inner_steps=10, early_stop_epsilon=1e-5,
               verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings

    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img


def save_visualize_cross_attn_map(cross_attn, decoder, tokens, file_name):
    cross_attn = cross_attn.permute(2, 0, 1)
    vis_images = []
    for j in range(len(tokens)):
        vis_img = cross_attn[j]
        vis_img = vis_img - vis_img.min()
        vis_img = 255 * vis_img / vis_img.max()
        vis_img = np.repeat(np.expand_dims(vis_img, axis=2), 3, axis=2).astype(np.uint8)
        vis_img = np.array(Image.fromarray(vis_img).resize((256, 256)))
        vis_img = ptp_utils.text_under_image(vis_img, decoder(int(tokens[j])))
        vis_img = np.array(vis_img)
        vis_images.append(vis_img)
    view_images(np.concatenate(vis_images, axis=1)).save(f'./outputs/nulltext_png_multi-step/{file_name}.png')

    # view_images(np.concatenate(vis_images, axis=1)).save('/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/test/diffmask.png')



@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        start_time=50,
        return_type='image',
        file_name=None,
        clip_image = None,
        text_prompt = None,
        image_size = None,
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    # height = width = 512
    height = width = image_size

    # 这些都可以不要的
    # text_input = model.tokenizer(
    #     prompt,
    #     padding="max_length",
    #     max_length=model.tokenizer.model_max_length,
    #     truncation=True,
    #     return_tensors="pt",
    # )
    # text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    # max_length = text_input.input_ids.shape[-1]

    uncond_embeddings= None

    # if uncond_embeddings is None:
    #     uncond_input = model.tokenizer(
    #         [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    #     )
    #     uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    # else:
    #     uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
        # pass
        # if uncond_embeddings_ is None:
        #     # context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        #     context = torch.cat([uncond_embeddings_, text_embeddings], dim=0)
        # else:
        # #     # text_embeddings = torch.cat([clip_image, text_embeddings], dim=1)
        # #     # uncond_embeddings_ = torch.cat([clip_image, uncond_embeddings_], dim=1)
        # #
        #
        #     # text_embeddings = next(iter(text_prompt.values())).mean(dim=0)
        #     # # text_embeddings = text_prompt['object'].mean(dim=0)    #
        #     # text_embeddings = text_embeddings[0, :, :].unsqueeze(0)   # 0 表示normal  1 表示abnormal
        #     # B = uncond_embeddings_.shape[0]
        #     # text_embeddings = text_embeddings.expand(B, -1, -1)
        #     # context = torch.cat([uncond_embeddings_, text_embeddings])    # [2,77,1024]
        #
        #     # text_embeddings = text_prompt['object'].mean(dim=0)  #
        #     # text_embeddings_normal = text_embeddings[0, :, :].unsqueeze(0) #
        #     #
        #     # text_embeddings_ab = text_embeddings[1, :, :].unsqueeze(0)  # 0 表示normal  1 表示abnormal
        #     # B = uncond_embeddings_.shape[0]
        #     # text_embeddings_normal = text_embeddings_normal.expand(B, -1, -1)
        #     # text_embeddings_ab = text_embeddings_ab.expand(B, -1, -1)
        #     # context = torch.cat([text_embeddings_normal, text_embeddings_ab])  #   为什么它没有用呢 正常异常没区分呢
        #
        #     # norm_test
        #     text_embeddings = next(iter(text_prompt.values()))  #   # text_embeddings = text_prompt['object']
        #     text_embeddings_normal = text_embeddings[0, :, :] #
        #
        #     # text_embeddings_ab = text_embeddings[1, :, :]  # 0 表示normal  1 表示abnormal
        #     # B = uncond_embeddings_.shape[0]
        #     # text_embeddings_normal = text_embeddings_normal.expand(B, -1, -1)
        #     # text_embeddings_ab = text_embeddings_ab.expand(B, -1, -1)
        #
        #     # context = torch.cat([text_embeddings_normal, text_embeddings_ab])
        #     context = torch.cat([uncond_embeddings_, text_embeddings_normal])
        #
        # # context = torch.cat([uncond_embeddings, text_embeddings], dim=0)
        #
        # # context = [text_embeddings]
        # # text_prompt_new = text_prompt['object'][:, 1, :, :]
        # # text_prompt_new = text_prompt_new[:1, :, :]
        #
        # # context = text_prompt['object'].mean(dim=0)
        # # context = text_prompt['object'].mean(dim=0)
        # # context = context[1,:,:].unsqueeze(0)
        # # latents = ptp_utils.diffusion_step(model, controller, latents, text_prompt_new, t, guidance_scale, low_resource=False)
        text_embeddings = next(iter(text_prompt.values()))  # # text_embeddings = text_prompt['object']
        text_embeddings_normal = text_embeddings[0, :, :]
        context = text_embeddings_normal
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent

def text2image_ldm_stable_img(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        start_time=50,
        return_type= 'image', #'image',
        file_name=None,
        clip_image = None,
        text_prompt=None,
        image_size=None,
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = image_size

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]

    # uncond_embeddings= None

    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
        # pass
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            # text_embeddings = torch.cat([clip_image, text_embeddings], dim=1)
            # uncond_embeddings_ = torch.cat([clip_image, uncond_embeddings_], dim=1)
            # text_embeddings = text_prompt['object'].mean(dim=0)
            # text_embeddings = text_embeddings[1, :, :].unsqueeze(0)
            # B = uncond_embeddings_.shape[0]
            # text_embeddings = text_embeddings.expand(B, -1, -1)
            # context = torch.cat([uncond_embeddings_, text_embeddings])  #【2，77，1024】

            # image_norm = clip_image[3]
            # B = uncond_embeddings_.shape[0]
            # uncond_embeddings_ = uncond_embeddings_.expand(B, -1, -1)
            # context = torch.cat([uncond_embeddings_, image_norm])

            text_embeddings = next(iter(text_prompt.values()))  # # text_embeddings = text_prompt['object']
            text_embeddings_normal = text_embeddings[0, :, :]

            # uncond_image = torch.zeros_like(clip_image)
            # uncond_context = torch.cat([uncond_embeddings_, uncond_image], dim=1)  # (1,87,1024)
            # cond_context = torch.cat([text_embeddings_normal, clip_image], dim=1)  # (1,87,1024)
            # context = torch.cat([uncond_context, cond_context])

            context = torch.cat([uncond_embeddings_,text_embeddings_normal])


        # text_embeddings = next(iter(text_prompt.values()))  # # text_embeddings = text_prompt['object']
        # text_embeddings_normal = text_embeddings[0, :, :]
        # context = torch.cat([text_embeddings_normal,clip_image],dim=1)
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent



def run_and_display(prompts, controller, ldm_stable,latent=None, run_baseline=False, generator=None, uncond_embeddings=None,
                    verbose=True, file_name=None,clip_image=None,text_prompt=None, onlyimg=True,image_size=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False,
                                         generator=generator)
        print("with prompt-to-prompt")
    if not onlyimg:
        images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent,
                                            num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                                            generator=generator, uncond_embeddings=uncond_embeddings, file_name=file_name,clip_image=clip_image,text_prompt=text_prompt,image_size=image_size)
    else:
        images, x_t = text2image_ldm_stable_img(ldm_stable, prompts, controller, latent=latent,
                                            num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                                            generator=generator, uncond_embeddings=uncond_embeddings, file_name=file_name,
                                            clip_image=clip_image,text_prompt=text_prompt,image_size=image_size)


    if verbose:
        ptp_utils.view_images(images)
    return images, x_t


def split_text(text):
    words_and_punctuation = re.findall(r"[\w']+|[.,!?;]", text)
    return words_and_punctuation


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_main_process():
    return dist.get_rank() == 0


from collections import Counter


def get_unique_elements(my_list):
    element_counts = Counter(my_list)
    unique_elements = [element for element, count in element_counts.items() if count == 1]
    return unique_elements


def find_nearest_period_index(word_list):
    target_index = 74
    nearest_period_index = None

    for i, word in enumerate(word_list[:75]):
        if word == '.':
            nearest_period_index = i
        elif i == target_index:
            break

    return nearest_period_index


def split_sentences(token_list):
    assert len(token_list) > 75

    splited_sentences = []
    while len(token_list) > 75:
        s_end_idx = find_nearest_period_index(token_list)
        if s_end_idx is None:
            splited_sentences.append(token_list[:75])
            token_list = token_list[75:]
        else:
            splited_sentences.append(token_list[:s_end_idx + 1])
            token_list = token_list[s_end_idx + 1:]
    if len(token_list) != 0:
        splited_sentences.append(token_list)
    return splited_sentences

#--------------DiffSeg
def aggregate_all_attention(prompts,attention_store: AttentionStore, from_where: List[str], is_cross: bool, select: int):
    attention_maps = attention_store.get_average_attention()
    att_8 = []
    att_16 = []
    att_32 = []
    att_64 = []
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == 8*8:
                cross_maps = item.reshape(len(prompts), -1, 8, 8, item.shape[-1])[select]
                att_8.append(cross_maps)
            if item.shape[1] == 16*16:
                cross_maps = item.reshape(len(prompts), -1, 16, 16, item.shape[-1])[select]
                att_16.append(cross_maps)
            if item.shape[1] == 32*32:
                cross_maps = item.reshape(len(prompts), -1, 32, 32, item.shape[-1])[select]
                att_32.append(cross_maps)
            if item.shape[1] == 64*64:
                cross_maps = item.reshape(len(prompts), -1, 64, 64, item.shape[-1])[select]
                att_64.append(cross_maps)
    atts = []
    for att in [att_8,att_16,att_32,att_64]:
        att = torch.cat(att, dim=0)
        att = att.sum(0) / att.shape[0]

        # att = torch.cat(att, dim=1)
        # att = att.sum(1) / att.shape[1]

        atts.append(att.detach())
    return atts


def aggregate_all_attention_batch(prompts, attention_store: AttentionStore, from_where: List[str], is_cross: bool, select: Optional[int] = None,image_size=None):
    attention_maps = attention_store.get_average_attention()
    att_8, att_16, att_32, att_64 = [], [], [], []

    B = len(prompts)

    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if is_cross:
                if item.shape[1] == 8 * 8:
                    cross_maps = item.reshape(B, -1, 8, 8, item.shape[-1])  # (B, heads, 8, 8, D)
                    att_8.append(cross_maps if select is None else cross_maps[select:select + 1])
                if item.shape[1] == 16 * 16:
                    cross_maps = item.reshape(B, -1, 16, 16, item.shape[-1])
                    att_16.append(cross_maps if select is None else cross_maps[select:select + 1])
                if item.shape[1] == 32 * 32:
                    cross_maps = item.reshape(B, -1, 32, 32, item.shape[-1])
                    att_32.append(cross_maps if select is None else cross_maps[select:select + 1])
                # if item.shape[1] == 64 * 64:
                #     cross_maps = item.reshape(B, -1, 64, 64, item.shape[-1])
                #     att_64.append(cross_maps if select is None else cross_maps[select:select + 1])

                latent_res = image_size // 8
                if item.shape[1] == latent_res * latent_res:
                    cross_maps = item.reshape(B, -1, latent_res, latent_res, item.shape[-1])
                    att_64.append(cross_maps if select is None else cross_maps[select:select + 1])


            else:
                # if item.shape[1] == 64 * 64:
                #     cross_maps = item.reshape(B, -1, 64, 64, item.shape[-1])
                #     att_64.append(cross_maps if select is None else cross_maps[select:select + 1])

                latent_res = image_size // 8
                if item.shape[1] == latent_res * latent_res:
                    cross_maps = item.reshape(B, -1, latent_res, latent_res, item.shape[-1])
                    att_64.append(cross_maps if select is None else cross_maps[select:select + 1])
    atts = []
    if is_cross:
        for att in [att_8, att_16, att_32, att_64]:
            att = torch.cat(att, dim=1)  # 拼 head 维度
            att = att.sum(1) / att.shape[1]  # 平均 head → (B, H, W, D)
            # atts.append(att.cpu())
            atts.append(att.detach())
        del att_8, att_16, att_32, attention_maps,cross_maps,item  # 及时清理
    else:

        for att in [att_64]:
            att = torch.cat(att, dim=1)  # 拼 head 维度
            att = att.sum(1) / att.shape[1]  # 平均 head → (B, H, W, D)
            # atts.append(att.cpu())
            atts.append(att.detach())
        del att_64,attention_maps,cross_maps,item

        # for att in [att_32]:
        #     att = torch.cat(att, dim=1)  # 拼 head 维度
        #     att = att.sum(1) / att.shape[1]  # 平均 head → (B, H, W, D)
        #     # atts.append(att.cpu())
        #     atts.append(att.detach())

    torch.cuda.empty_cache()


    return atts


def aggregate_all_attention_batch_imagesize(prompts, attention_store: AttentionStore, from_where: List[str],
                                  is_cross: bool, select: Optional[int] = None, image_size=None):
    attention_maps = attention_store.get_average_attention()

    # 初始化 4 个桶（动态层级，而不是固定 att_8, att_16, att_32, att_64）
    att_levels = [[] for _ in range(4)]

    B = len(prompts)
    latent_res = image_size // 8  # 最大 latent 分辨率（64 对应 512，32 对应 256）

    # 动态确定 4 层分辨率
    # 例子：512 -> [8,16,32,64]；256 -> [4,8,16,32]
    resolutions = [latent_res // (2 ** i) for i in range(3, -1, -1)]

    # 把不同where 的相同分辨率map 放到att_levels[idx]同一个idx下
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if is_cross:
                for idx, res in enumerate(resolutions):
                    if item.shape[1] == res * res:
                        cross_maps = item.reshape(B, -1, res, res, item.shape[-1])
                        att_levels[idx].append(cross_maps if select is None else cross_maps[select:select + 1])
            else:
                # # self-attn 只保留最大分辨率层
                # if item.shape[1] == latent_res * latent_res:
                #     cross_maps = item.reshape(B, -1, latent_res, latent_res, item.shape[-1])
                #     att_levels[-1].append(cross_maps if select is None else cross_maps[select:select + 1])
                for idx, res in enumerate(resolutions):
                    if item.shape[1] == res * res:
                        cross_maps = item.reshape(B, -1, res, res, item.shape[-1])
                        att_levels[idx].append(cross_maps if select is None else cross_maps[select:select + 1])

    # 拼接并平均 head
    atts = []
    if is_cross:
        for att in att_levels:
            if len(att) > 0:
                att = torch.cat(att, dim=1)
                att = att.sum(1) / att.shape[1]   # 原来的
                # att_softmax = torch.softmax(att, dim=1)  # [1, N, H, W]
                # att = (att * att_softmax).sum(dim=1)
                atts.append(att.detach())
    else:
        # for att in [att_levels[-1]]:  # 只用最大分辨率
        #     if len(att) > 0:
        #         att = torch.cat(att, dim=1)
        #         att = att.sum(1) / att.shape[1]
        #         atts.append(att.detach())

        for att in att_levels:
            if len(att) > 0:
                att = torch.cat(att, dim=1)
                att = att.sum(1) / att.shape[1]   # 原来的
                # att_softmax = torch.softmax(att, dim=1)  # [1, N, H, W]
                # att = (att * att_softmax).sum(dim=1)
                atts.append(att.detach())

    del attention_maps, item
    torch.cuda.empty_cache()

    return atts, resolutions



def encode_imgs(imgs,vae):
    # imgs: [B, 3, H, W]
    imgs = 2 * imgs - 1
    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample() * 0.18215

    return latents


def sd_to_clip_img(img_tensor):
    # hard-coded
    # step 1: resize to 14/16
    H, W = img_tensor.shape[-2:]
    H = H * 7 // 16  # H = H * 7 // 8    GG——LLM中是 8512
    W = W * 7 // 16
    img_tensor = nn.functional.interpolate(img_tensor, size=(H, W), mode='bilinear', align_corners=False)
    # step 2: re-normalize
    sd_mean = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device, dtype=img_tensor.dtype).view(1, 3, 1, 1)
    sd_std = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device, dtype=img_tensor.dtype).view(1, 3, 1, 1)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img_tensor.device,
                             dtype=img_tensor.dtype).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img_tensor.device,
                            dtype=img_tensor.dtype).view(1, 3, 1, 1)
    img_tensor = img_tensor * sd_std + sd_mean
    img_tensor = (img_tensor - clip_mean) / clip_std
    return img_tensor

def heatmap_img(img,attention_map,save_path=None):
    if save_path==None:
        save_path="/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/heatmap/1.png"
    image_np = np.array(img) / 255.0
    heatmap = attention_map[:, :, ::-1] / 255.0
    overlay = 0.5 * image_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    plt.imsave(save_path, overlay)
    print(f"Saved attention overlay at {save_path}")


def Gen_bbox_all(cam, img_hw, save_path=None, threshold=0.5, save=False):
    image_np = np.array(img_hw) / 255.0

    attention_map = (cam / cam.max())

    binary_mask = (attention_map > threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append((x, y, x + w, y + h))

    img_with_box = (image_np * 255).astype(np.uint8).copy()
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

    if save:
    # 保存或显示
        Image.fromarray(img_with_box).save(
            save_path)

    return bounding_boxes

def Gen_bbox_single(cam, img_hw, save_path=None, threshold=0.5, save=False):
    image_np = np.array(img_hw) / 255.0

    attention_map = (cam / cam.max())

    binary_mask = (attention_map > threshold).astype(np.uint8) * 255

    max_loc = np.unravel_index(np.argmax(attention_map), attention_map.shape)  # (y, x)

    print(binary_mask.shape)
    # 用 floodFill 找出包含这个点的区域
    h, w = binary_mask.shape
    print(binary_mask.shape)
    mask = np.zeros((h + 2, w + 2), np.uint8)  # floodFill 的 mask 要比图大2
    filled = binary_mask.copy()

    # floodFill 会改变 filled，因此我们传进去
    cv2.floodFill(filled, mask, seedPoint=(max_loc[1], max_loc[0]), newVal=2)

    # 提取标记为 2 的区域（最大响应点所在连通域）
    region_mask = (filled == 2).astype(np.uint8) * 255

    # 提取该区域的 bounding box
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not contours:
    #     return None, region_mask

    x, y, w, h = cv2.boundingRect(contours[0])

    if save:
        img_with_box = (image_np * 255).astype(np.uint8).copy()

        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)
        Image.fromarray(img_with_box).save(save_path)

        # Image.fromarray(img_with_box).save("/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/img_with_bbox/boxed_output_single.png")

    return (x, y, x + w, y + h), region_mask



import json
import os.path as osp
import torch.distributed as dist
import os

if __name__ == '__main__':

    # dist.init_process_group(backend="nccl", init_method='env://', world_size=-1, rank=-1, group_name='')
    #
    # setup_for_distributed(is_main_process())
    # local_rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # torch.cuda.set_device(local_rank)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    local_rank = 0
    world_size = 1
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    torch.cuda.set_device(device)  # 如果有多卡，设置默认使用第0张卡

    # 模拟原分布式环境的设置函数
    if "setup_for_distributed" in dir():  # 如果定义了此函数
        setup_for_distributed(True)  # 强制设置为单卡的主进程

    # 使用 DiffPNG 设置的scheduler
    # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,   # DiffPNG 设置的scheduler
    #                           set_alpha_to_one=False)
    # ldm_stable = StableDiffusionPipeline.from_pretrained("/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/stable_diffusion", use_auth_token=MY_TOKEN,
    #                                                      scheduler=scheduler).to(device)
    # 使用默认的 scheduler
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/stable_diffusion", ).to(device)
    scheduler = DDIMScheduler.from_pretrained("/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/stable_diffusion",subfolder="scheduler")

    #-----vae
    vae = ldm_stable.vae.to(device)
    #--------------

    tokenizer = ldm_stable.tokenizer
    # data = json.load(open("./ppmn_narr_list.json"))
    null_inversion = NullInversion(ldm_stable)

    #-----------   加载CLIP -------------------------
    from transformers import CLIPImageProcessor, CLIPTextModel, CLIPVisionModel

    clip_model = '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP-vit-large-14'  #[1,256,1024]

    # clip_model = '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP_vit_H-14'  # image feature [1,256,1280]
    clip_encoder = CLIPVisionModel.from_pretrained(clip_model).to("cuda")
    clip_encoder.eval()
    clip_encoder.to(device)


     # clip_img_tensor = sd_to_clip_img(img_tensor)
     # clip_features = clip_encoder(clip_img_tensor, output_hidden_states=True)
     # clip_features = clip_features.hidden_states[-2][:, 1:]

    #-----------------------------------------------------

    #
    # for idx, i in tqdm(enumerate(range(len(data))), total=len(data)):
    #     if i % world_size != local_rank:
    #         continue
    #     image_id = data[i]['image_id']
    #     tag_id = data[i]['tag_id']
    #     if osp.exists(f'./outputs/attn_db/{tag_id}/'):
    #         pass
    #     else:
    #         os.makedirs(f'./outputs/attn_db/{tag_id}')
    #
    #s
    #     # load image
    #     image_path = osp.join(
    #         "./datasets/coco/val2017",
    #         "{:012d}.jpg".format(int(image_id)),
    #     )

    with open('/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/data/refcoco_testa.jsonl',
              encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    for idx, i in tqdm(enumerate(range(len(data))), total=len(data)):
        if i % world_size != local_rank:
            continue
        image_id = data[i]['image_id']
        tag_id = data[i]['ann_id']
        if osp.exists(f'./outputs/attn_db/{tag_id}/'):
            pass
        else:
            os.makedirs(f'./outputs/attn_db/{tag_id}')

        # load image
        image_path = osp.join(
            "/home/data/zjp/dataset/coco2014/images/train2014/",
            "COCO_train2014_000000{}.jpg".format((image_id)),
        )
        # image_path= '/home/data/zjp/dataset/coco2014/images/train2014/COCO_train2014_000000581282.jpg'


    # for tag_id,datum in tqdm(data):
    #     if "coco" in datum["file_name"].lower():
    #         file_name = "_".join(datum["file_name"].split("_")[:-1]) + ".jpg"
    #     else:
    #         file_name = datum["file_name"]
    #
    #     image_root = '/home/data/zjp/dataset/coco2014/images/train2014/'
    #     img_path = os.path.join(image_root, file_name)

        prompt = data[i]['sentences'][0]['sent']

        #---------------------  测试 --------------------------
        image_path = '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/diffseg_testpng.jpg'
        prompt = 'Dog'
        #-----------------------------------------------------

        # prompt = data[i]['caption']
        tokens = split_text(prompt)
        clip_tokens_ids = ldm_stable.tokenizer(prompt)['input_ids']
        clip_tokens = [ldm_stable.tokenizer.decode(i) for i in clip_tokens_ids[1:-1]]
        if len(clip_tokens) <= 75:
            sentences = [prompt]
        else:
            splited_tokens = split_sentences(clip_tokens)
            sentences = []
            start = 0
            clip_tokens_ids_valid = clip_tokens_ids[1:-1]  # BOS EOS
            for i in range(len(splited_tokens)):
                print(start, 'end={}'.format(start + len(splited_tokens[i])))
                sentences += [ldm_stable.tokenizer.decode(
                    [clip_tokens_ids[0]] + clip_tokens_ids_valid[start:start + len(splited_tokens[i])] + [
                        clip_tokens_ids[-1]])[15:-14]]
                start += len(splited_tokens[i])

        for idx, p in enumerate(sentences):
            # --------------------------- DiffPNG -------------------------------------------------------
            # (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, p, offsets=(0, 0, 0, 0),
            #                                                                       verbose=True)
            # prompts = [p]

            # controller = AttentionStore()
            # image_inv, x_t = run_and_display(prompts, controller, run_baseline=False, latent=x_t,
            #                                  uncond_embeddings=uncond_embeddings, verbose=False,
            #                                  file_name=f'{tag_id}_{idx}')
            # # save_multi_scale_attention_map(controller, tag_id, idx)
            #
            # # show_cross_attention(controller,res=int(32),['up'])
            # print(f'save:{tag_id}_{idx}')
            #--------------------------- DiffPNG -------------------------------------------------------

            # ------------------DiffSegmenter-----------------------------------
            imgs = []
            images = []
            dtype = torch.float32

            times = [100]
            controller = AttentionStore()
            g_cpu = torch.Generator(4307)

            for t in times:


                controller.reset()
                prompts = [p]

                blip_path = '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/BLIP/blip-image-captioning-large'
                from transformers import BlipProcessor, BlipForConditionalGeneration
                processor = BlipProcessor.from_pretrained(blip_path)
                model = BlipForConditionalGeneration.from_pretrained(blip_path).to(device)
                prompt_texts = []

                prompt_blip = f'a photography of {p}'
                raw_image = Image.open(image_path).convert('RGB')
                inputs = processor(raw_image,prompt_blip,return_tensors='pt').to(device)
                out = model.generate(**inputs)
                out_prompt = processor.decode(out[0], skip_special_tokens=True)
                word_len = len(prompts[0].split(" "))
                embs_len = len(tokenizer.encode(prompts[0])) - 2
                out_prompt = out_prompt.split(" ")
                last_word = p.split(" ")[-1]
                out_prompt[2 + word_len] = f"{last_word}++"
                prompt = [" ".join(out_prompt)]
                prompt_texts.append(f"{idx} {prompt[0]}")
                print(idx, prompt)

               #  ---------------------------   DiffSeg Diffusion---------------------------------
                rgb_512= load_512(image_path)

                from torchvision import transforms
                to_tensor = transforms.Compose([
                    # transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])

                img_open = to_tensor(rgb_512)
                rgb_512 = img_open.unsqueeze(0).to(device)


                #------------CLIP 图像预处理----------------
                with torch.no_grad():
                    img_tensor = rgb_512
                    clip_img_tensor = sd_to_clip_img(img_tensor)
                    clip_features = clip_encoder(clip_img_tensor, output_hidden_states=True)

                # 提取目标相关clip_image 特征
                image_feats =clip_features.last_hidden_state
                tokens = split_text(prompt[0])
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
                token_index = find_token_index_by_word(tokens, p)
                text_input = ldm_stable.tokenizer(
                    prompt[0],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(model.device))[0]
                text_feats = text_embeddings
                target_token_feat = text_feats[0, token_index]
                patch_feats = image_feats[0, 1:, :]
                import torch
                import torch.nn.functional as F

                similarity = F.cosine_similarity(patch_feats, target_token_feat.unsqueeze(0), dim=-1)  # [N-1]
                topk_indices = similarity.topk(k=2).indices
                relevant_feats = patch_feats[topk_indices]
                #




                clip_features = clip_features.hidden_states[-2][:, 1:]    # [1 256 1024]   # CLIP
                # uncond_embeddings = clip_features
                #------------CLIP 图像预处理----------------

                input_latent = encode_imgs(rgb_512,vae).to(device)
                noise = torch.randn([1, 4, 64, 64]).to(device)  # noise 随机初始化
                noise = noise if dtype == torch.float32 else noise.half()
                latents_noisy = ldm_stable.scheduler.add_noise(input_latent, noise, torch.tensor(t, device=device))
                latents_noisy = latents_noisy if dtype == torch.float32 else latents_noisy.half()
                # image_inv, x_t = run_and_display(prompts, controller, run_baseline=False, latent=latents_noisy,
                #                                  verbose=False,
                #                                  file_name=f'{tag_id}_{idx}')

                image_inv, x_t = run_and_display(prompts, controller, run_baseline=False, latent=latents_noisy,
                                                 verbose=False,
                                                 file_name=f'{tag_id}_{idx}',clip_image=clip_features,onlyimg=False)

                #  ---------------------------   DiffSeg Diffusion---------------------------------


                #  ---------------------------   DiffPNG Diffusion---------------------------------
                #  不同点： 在于 noise 生成方式不一样
                # (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, p,
                #                                                                     offsets=(0, 0, 0, 0),
                #                                                                     verbose=True)
                #
                # image_inv, x_t = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,
                #                                  verbose=False,
                #                                  file_name=f'{tag_id}_{idx}')

                #  ---------------------------   DiffPNG Diffusion---------------------------------


                #  可视化
                # image_data = np.squeeze(image_inv, axis=0)
                # Image.fromarray(image_data, 'RGB').save(
                #     "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/test/VG.jpg")

                out_atts = []
                weight = [0.3, 0.5, 0.1, 0.1]
                word_len = len(prompts[0].split(" "))
                embs_len = len(tokenizer.encode(prompts[0])) - 2
                import torch.nn.functional as F

                cross_attention_maps = aggregate_all_attention(prompts, controller, ("up", "mid", "down"), True, 0)
                self_attention_maps = aggregate_all_attention(prompts, controller, ("up", "mid", "down"), False, 0)
                for idx, res in enumerate([8, 16, 32, 64]):
                    try:
                        if prompt[0].split(" ")[3 + word_len].endswith("ing"):
                            cross_att = cross_attention_maps[idx][:, :, [3 + embs_len, 5 + embs_len]].mean(2).view(res,
                                                                                                                   res).float()
                        # print(decoder(int(tokenizer.encode(prompt[0])[3+embs_len])),decoder(int(tokenizer.encode(prompt[0])[5+embs_len])))
                        else:
                            cross_att = cross_attention_maps[idx][:, :, [3 + embs_len]].mean(2).view(res, res).float()
                    except:
                        cross_att = cross_attention_maps[idx][:, :, [3 + embs_len]].mean(2).view(res, res).float()

                    if res != 64:
                        cross_att = F.interpolate(cross_att.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear',
                                                  align_corners=False).squeeze().squeeze()
                    cross_att = (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min())
                    out_atts.append(cross_att * weight[idx])

                cross_att_map = torch.stack(out_atts).sum(0).view(64 * 64, 1)
                self_att = self_attention_maps[3].view(64 * 64, 64 * 64).float()
                att_map = torch.matmul(self_att, cross_att_map).view(res, res)

                imgs.append(att_map)

            #  不同time下得到的att_map
            images.append(torch.mean(torch.stack(imgs, dim=0), dim=0).unsqueeze(0).repeat(3, 1, 1))
        # 不同sentence下得到的att_map
        images = torch.stack(images)

        #-------  heatmap 格式 可视化 -----------------
        img_hw = Image.open(image_path)
        w = img_hw.width
        h = img_hw.height

        images = F.interpolate(images, size=(h, w), mode='bilinear', align_corners=False)
        pixel_max = images.max()
        for i in range(images.shape[0]):
            images[i] = ((images[i] - images[i].min()) / (images[i].max() - images[i].min())) * 255

        #  attention map热力图显示
        cam_dict = {}
        cam = images[0].permute(1,2,0).cpu().detach().numpy()[:,:,0]

        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_uint8 = (cam_norm * 255).astype(np.uint8)
        cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)   #COLORMAP_JET   COLORMAP_VIRIDIS   COLORMAP_HOT

        # heatmap_img(img_hw,cam_colored)
        # Gen_bbox_all(cam,img_hw)
        bbox,mask=Gen_bbox_single(cam,img_hw)


        # cv2.imwrite("/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/test/cam_output_time100.png", cam_colored)
        # cv2.imwrite("/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/random/rand_seed4307.png", cam_colored)
        cv2.imwrite('/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/test_dog_noisediffPNG_Normalize.png',cam_colored)
        # plt.imshow(cam, cmap="jet")
        # plt.axis('off')
        # plt.savefig(
        #     '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/test/heatmap_image.png',
        #     bbox_inches='tight', pad_inches=0)
        # plt.close()
        #-------  heatmap 格式 可视化 -----------------

        #--------------DiffSeg mat保存 ------------

        # images = F.interpolate(images, size=(h, w), mode='bilinear', align_corners=False)
        # pixel_max = images.max()
        # for i in range(images.shape[0]):
        #     images[i] = ((images[i] - images[i].min()) / (images[i].max() - images[i].min())) * 255
        # cam_dict = {}
        # for i in range(0, len(y)):
        #     cam_dict[str(embs.index(y[i]))] = images[i].permute(1, 2, 0).cpu().numpy()[:, :, 0]
        #
        # import scipy.io as sio
        # sio.savemat(os.path.join(img_output_path, 'COCO_val2014_' + img_path[-16:-4] + '.mat'), cam_dict,
        #             do_compression=True)

        #--------------DiffSeg mat保存 ------------

# TODO：问题： 1. 随机数影响比较大 --> Noise 是随机生成的   2. Diffusion 中 attention map 融合方式
# TODO： 1. 添加CLIP 相关图 融入 Noise 中，另外随机数的选择问题？  2.   外观语义引导       3. CLIP_attention map 生成方式  4. 得到map后，如何定位--> bbox

# DiffPNG 源代码

#  问题：实现了提取目标语义相关CLIP特征，但是把CLIP特征当成条件输入u-net时，报错