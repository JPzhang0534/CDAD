import torch
from PIL import Image
from torchvision import transforms
from SDCLIP.sd_encoder import SDCLIPFeaturizer  # 假设你已将类保存为这个模块
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


img_path = "/home/data/zjp/dataset/TNL2K_test/advSamp_Baseball_game_002-Done/imgs/00001.jpg"
prompt = "the women who is holding a tennis racket in red vest and white hat and short skirt"
model_key = "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/stable_diffusion"

image = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
img_tensor = transform(image).unsqueeze(0).to(device)


# 1. 初始化模型

model = SDCLIPFeaturizer(
    sd_id=model_key, null_prompt=prompt,
    clip_model='/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP-vit-large-14',
)


# 4. 前向推理
with torch.no_grad():
    outputs = model.forward(img_tensor, prompt=prompt, t=261)

print(outputs)