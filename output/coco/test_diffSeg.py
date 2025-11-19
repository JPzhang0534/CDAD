# 现在实现： 可以正常读取，提取流程
# 问题： attention map 中 up 总是 nan


from typing import Optional, Union, Tuple, List, Callable, Dict
import sys

sys.path.insert(0, sys.path[0] + "/../..")
import scipy.io as sio
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import abc
import ptp_utils
import visual_code.seq_aligner as seq_aligner
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset.mydatasets import build_dataset
import argparse
import os
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
import torch.nn.functional as nnf
from tqdm import tqdm

from interpreter import *
from collections import defaultdict

import os
import shutil
from glob import glob



MY_TOKEN = None
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, f'{dstpath}/{fname}')          # 复制文件
        print ("copy %s -> %s"%(srcfile, f'{dstpath}/{fname}'))


class LocalBlend:
    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


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


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
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


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                            1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                                                local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


from PIL import Image


def aggregate_all_attention(prompts, attention_store: AttentionStore, from_where: List[str], is_cross: bool,
                            select: int):
    attention_maps = attention_store.get_average_attention()
    att_8 = []
    att_16 = []
    att_32 = []
    att_64 = []
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == 8 * 8:
                cross_maps = item.reshape(len(prompts), -1, 8, 8, item.shape[-1])[select]
                att_8.append(cross_maps)
            if item.shape[1] == 16 * 16:
                cross_maps = item.reshape(len(prompts), -1, 16, 16, item.shape[-1])[select]
                att_16.append(cross_maps)
            if item.shape[1] == 32 * 32:
                cross_maps = item.reshape(len(prompts), -1, 32, 32, item.shape[-1])[select]
                att_32.append(cross_maps)
            if item.shape[1] == 64 * 64:
                cross_maps = item.reshape(len(prompts), -1, 64, 64, item.shape[-1])[select]
                att_64.append(cross_maps)
    atts = []
    for att in [att_8, att_16, att_32, att_64]:
        att = torch.cat(att, dim=0)
        att = att.sum(0) / att.shape[0]
        atts.append(att.cpu())
    return atts


def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool,
                        select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out).mean(0)
    return out.cpu()


def show_cross_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    texts = []
    maps = []
    tokens = tokenizer.encode(prompts[select])
    map = []
    for j in range(len(tokens)):
        map.append(attention_maps[:, :, j])
        # image = attention_maps[:, :, i]
        # image = 255 * image / image.max()
        # image = image.unsqueeze(-1).expand(*image.shape, 3)
        # image = image.numpy().astype(np.uint8)
        # image = np.array(Image.fromarray(image).resize((256, 256)))
        # image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        # images.append(image)
        texts.append(decoder(int(tokens[j])))
    maps.append(torch.stack(map))

    # ptp_utils.view_images(np.stack(images, axis=0))


    return images, texts, maps


def show_self_attention_comp(prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                             max_com=10, select: int = 0):
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, False, select).view(res ** 2,
                                                                                                        res ** 2)
    # u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    # for i in range(max_com):
    #     image = vh[i].reshape(res, res)
    #     image = image - image.min()
    #     image = 255 * image / image.max()
    #     image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
    #     image = Image.fromarray(image).resize((256, 256))
    #     image = np.array(image)
    #     images.append(image)
    return images, attention_maps
    # ptp_utils.view_images(np.concatenate(images, axis=1))


def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, t=100, noise_sample_num=2):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False,
                                         generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=t,
                                                  guidance_scale=GUIDANCE_SCALE, generator=generator,
                                                  low_resource=LOW_RESOURCE, noise_sample_num=noise_sample_num)
    # ptp_utils.view_images(images)
    return images, x_t


def encode_imgs(imgs, vae):
    # imgs: [B, 3, H, W]
    imgs = 2 * imgs - 1
    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample() * 0.18215

    return latents

def find_nearest_period_index(word_list):
    target_index = 74
    nearest_period_index = None

    for i, word in enumerate(word_list[:75]):
        if word == '.':
            nearest_period_index = i
        elif i == target_index:
            break

    return nearest_period_index
def split_text(text):
    words_and_punctuation = re.findall(r"[\w']+|[.,!?;]", text)
    return words_and_punctuation


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




# if __name__ == '__main__':
    # ---------------------- 路径和数据集设置 --------------
output_path=f'./output/coco'
img_output_path = f"{output_path}/images"
attmap_output_path = f"{output_path}/self_att_maps"
for paths in [output_path,img_output_path]:
    if not os.path.exists(paths):
        os.makedirs(paths)
mycopyfile(os.path.abspath(__file__), output_path)

dtype = torch.float16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# img_path = "/home/data/zjp/dataset/TNL2K_test/advSamp_Baseball_game_002-Done/imgs/00001.jpg"
# prompt = ["the women who is holding a tennis racket in red vest and white hat and short skirt"]


img_path = '/home/data/zjp/fig1.jpg'
prompt=["A photo of vase "]
model_key = "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/stable_diffusion"
clip_model='/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP-vit-large-14'
# 定义模型

# CLIP
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPVisionModel
clip_encoder = CLIPVisionModel.from_pretrained(clip_model).to("cuda")
clip_encoder.requires_grad_(False)
clip_img = clip_encoder.to(device)


# stable diffusion
ldm_stable = StableDiffusionPipeline.from_pretrained(model_key, local_files_only=True, torch_dtype=dtype).to(device)
tokenizer = ldm_stable.tokenizer
ldm_stable.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
vae = ldm_stable.vae.to(device)
controller = AttentionStore()  # ??
g_cpu = torch.Generator(4307)  #4307
times = [100]
weight = [0.3, 0.5, 0.1, 0.1]  # 超参数


to_tensor = transforms.Compose([
        # transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
import PIL.Image
img_open = PIL.Image.open(img_path).convert("RGB")
img_open = to_tensor(img_open)
img_open = img_open.unsqueeze(0)
#
# data=[]
# data.append(img_open)

with open('/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/data/refcoco_testa.jsonl',
          encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

for idx, i in tqdm(enumerate(range(len(data))), total=len(data)):
    # if i % world_size != local_rank:
    #     continue
    # image_id = data[i]['file_name']
    # tag_id = data[i]['ann_id']
    # if osp.exists(f'./outputs/attn_db/{tag_id}/'):
    #     pass
    # else:
    #     os.makedirs(f'./outputs/attn_db/{tag_id}')

    # load image
    # image_path = osp.join(
    #     "/home/data/zjp/dataset/coco2014/images/train2014/",
    #     "{}".format((image_id)),
    # )
    image_path = '/home/data/zjp/dataset/coco2014/images/train2014/COCO_train2014_000000581282.jpg'

    # for tag_id,datum in tqdm(data):
    #     if "coco" in datum["file_name"].lower():
    #         file_name = "_".join(datum["file_name"].split("_")[:-1]) + ".jpg"
    #     else:
    #         file_name = datum["file_name"]
    #
    #     image_root = '/home/data/zjp/dataset/coco2014/images/train2014/'
    #     img_path = os.path.join(image_root, file_name)

    prompt = data[i]['sentences'][0]['sent']

    data_img = []
    import PIL.Image

    to_tensor = transforms.Compose([
        # transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])

    img_open = PIL.Image.open(img_path).convert("RGB")
    img_open = to_tensor(img_open)
    img_open = img_open.unsqueeze(0)
    data_img.append(img_open)

    # 处理img--> sd 格式
    img_temp = data_img[0].permute(0, 2, 3, 1)
    h = data_img[0].shape[-2]
    w = data_img[0].shape[-1]
    orig_images = torch.zeros_like(img_temp)
    orig_images[:, :, :, 0] = img_temp[:, :, :, 0]  # * 0.229 + 0.485 R
    orig_images[:, :, :, 1] = img_temp[:, :, :, 1]  # * 0.224 + 0.456 G
    orig_images[:, :, :, 2] = img_temp[:, :, :, 2]  # * 0.225 + 0.406 B    # 没用到呀
    orig_images[orig_images > 1] = 1
    orig_images[orig_images < 0] = 0
    rgb_512 = F.interpolate(orig_images.permute(0, 3, 1, 2), (512, 512), mode='bilinear', align_corners=False).to(
        device)
    rgb_512 = rgb_512 if dtype == torch.float32 else rgb_512.half()
    input_latent = encode_imgs(rgb_512, vae).to(device)  # img --> vae --> z 隐空间特征
    noise_sample_num = 1
    noise = torch.randn([1, 4, 64, 64]).to(device)  # noise 随机初始化
    noise = noise if dtype == torch.float32 else noise.half()
    images = []
    imgs = []
    #
    # imgs = []

#----------------prompt处理--》DiffPNG


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
                [clip_tokens_ids[0]] + clip_tokens_ids_valid[start:start + len(splited_tokens[i])] + [clip_tokens_ids[-1]])[
                          15:-14]]
            start += len(splited_tokens[i])


#----------------

    for idx, prompt in enumerate(sentences):

        for t in times:
        # t = 261
        #     t = 300
            prompt = 'a photo of ' + str(prompt)
            prompt = [prompt]
            controller.reset()
            latents_noisy = ldm_stable.scheduler.add_noise(input_latent, noise, torch.tensor(t, device=device))
            latents_noisy = latents_noisy if dtype == torch.float32 else latents_noisy.half()
            # 控制器注入， 传入 Stable Diffusion U-Net 的注意力层钩子中
            _, _ = run_and_display(prompt, controller, latent=latents_noisy, run_baseline=False, generator=g_cpu, t=t,
                                   noise_sample_num=noise_sample_num)
            out_atts = []

            cross_attention_maps = aggregate_all_attention(prompt, controller, ("up", "mid", "down"), True, 0)
            self_attention_maps = aggregate_all_attention(prompt, controller, ("up", "mid", "down"), False, 0)      # 问题：up 32 attention map nan
            embs_len = len(tokenizer.encode(prompt[0])) - 2
            word_len = len(prompt[0].split(" "))
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
            #
            cross_att_map = torch.stack(out_atts).sum(0).view(64 * 64, 1)
            self_att = self_attention_maps[3].view(64 * 64, 64 * 64).float()
            att_map = torch.matmul(self_att, cross_att_map).view(res, res)
            #
            imgs.append(att_map)
        images.append(torch.mean(torch.stack(imgs,dim=0),dim=0).unsqueeze(0).repeat(3,1,1))

        img_hw = Image.open(image_path)
        w = img_hw.width
        h = img_hw.height
        images = F.interpolate(images, size=(h, w), mode='bilinear', align_corners=False)
        pixel_max = images.max()
        for i in range(images.shape[0]):
            images[i] = ((images[i] - images[i].min()) / (images[i].max() - images[i].min())) * 255

        cam_dict = {}
        cam = images[0].permute(1, 2, 0).cpu().numpy()[:, :, 0]

        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_uint8 = (cam_norm * 255).astype(np.uint16)
        cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_VIRIDIS)

        cv2.imwrite(
            "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/test/cam_output.png",
            cam_colored)


    # images.append(torch.mean(torch.stack(imgs, dim=0), dim=0).unsqueeze(0).repeat(3, 1, 1))
    # images = torch.stack(images)
    # images = F.interpolate(images, size=(h, w), mode='bilinear', align_corners=False)
    # pixel_max = images.max()
    # for i in range(images.shape[0]):
    #     images[i] = ((images[i] - images[i].min()) / (images[i].max() - images[i].min())) * 255
    # cam_dict = {}
    # for i in range(0, len(y)):
    #     cam_dict[str(embs.index(y[i]))] = images[i].permute(1, 2, 0).cpu().numpy()[:, :, 0]
    # sio.savemat(os.path.join(img_output_path, 'COCO_val2014_' + img_path[-16:-4] + '.mat'), cam_dict,
    #             do_compression=True)