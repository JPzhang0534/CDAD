from diffusers import StableDiffusionPipeline, DDIMScheduler
# from attention_store import AttentionStore
# from utils import load_image, save_attention_maps
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

MODEL_ID= "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/stable_diffusion"
# MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
PROMPT = "the women who is holding a tennis racket in red vest and white hat and short skirt"
IMAGE_PATH = "/home/data/zjp/dataset/TNL2K_test/advSamp_Baseball_game_002-Done/imgs/00001.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path, size=512):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return transform(image).unsqueeze(0)

def save_attention_maps(attn_store, save_dir):
    import os
    os.makedirs(save_dir, exist_ok=True)

    for attn_type in attn_store.attn_maps:
        for i, attn in enumerate(attn_store.attn_maps[attn_type]):
            torch.save(attn, f"{save_dir}/{attn_type}_attn_{i}.pt")

class AttentionStore:
    def __init__(self):
        self.attn_maps = {"cross": [], "self": []}

    def register_hooks(self, unet):
        def hook_fn(module, input, output, attn_type):
            if hasattr(module, "attention_probs"):
                self.attn_maps[attn_type].append(module.attention_probs.detach().cpu())

        for name, module in unet.named_modules():
            if "CrossAttention" in str(type(module)):
                module.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, attn_type="cross"))
            if "Attention" in str(type(module)) and "CrossAttention" not in str(type(module)):
                module.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, attn_type="self"))

def main():
    # 初始化模型
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 图像处理成 latent
    init_image = load_image(IMAGE_PATH).to(DEVICE).half()
    init_latents = pipe.vae.encode(init_image).latent_dist.mean * 0.18215

    # 文本处理
    text_inputs = pipe.tokenizer([PROMPT], padding="max_length", return_tensors="pt").to(DEVICE)
    text_embeddings = pipe.text_encoder(**text_inputs)[0]

    # 注册 attention hook
    attn_store = AttentionStore()
    attn_store.register_hooks(pipe.unet)

    # 给 UNet 输入
    t = pipe.scheduler.timesteps[50]
    noise = torch.randn_like(init_latents)
    noisy_latent = pipe.scheduler.add_noise(init_latents, noise, t)

    with torch.no_grad():
        _ = pipe.unet(noisy_latent, t, encoder_hidden_states=text_embeddings)

    # 保存 attention maps
    save_attention_maps(attn_store, "./outputs/attention_maps/")
    print("Saved attention maps.")

if __name__ == "__main__":
    main()