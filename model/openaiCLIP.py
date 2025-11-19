import torch
import json
from transformers import CLIPModel, CLIPConfig
from safetensors import safe_open

# 1. 加载配置文件
config_path = "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP_vit_H-14/open_clip_config.json"

# 2. 初始化模型
config = CLIPConfig.from_pretrained(config_path)
model = CLIPModel(config)

# 3. 加载权重 (优先选择 safetensors)
try:
    # 使用 safetensors 加载 (推荐)
    state_dict = {}
    with safe_open("/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP_vit_H-14/open_clip_model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # 特殊处理：适配 CLIP 的预训练权重格式
    model.load_state_dict(state_dict)
    print("成功加载 .safetensors 权重文件")

except:
    # 备用方案：使用 .bin 文件加载
    print("尝试加载 .bin 文件作为备选方案")
    state_dict = torch.load("/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP_vit_H-14/open_clip_pytorch_model.bin", map_location="cpu")

    # 处理可能的键名前缀问题
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # 过滤掉可能的 metadata
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print("成功加载 .bin 权重文件")

# 4. 验证模型加载成功
model.eval()
print("模型加载完成！模型结构:")
print(model)

# 5. 测试推理（可选）
from transformers import CLIPProcessor

processor = CLIPProcessor.from_pretrained(config_path)
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=torch.randn(1, 3, 224, 224),  # 随机图像
    return_tensors="pt"
)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
print("\n测试输出形状:", logits_per_image.shape)