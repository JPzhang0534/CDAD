import torch
from safetensors.torch import load_file
import json
import open_clip
import argparse

def convert_to_pt(model_dir, output_path="ViT-H-14.pt", device="cpu"):
    # 路径
    config_path = f"{model_dir}/open_clip_config.json"
    weight_path = f"{model_dir}/open_clip_model.safetensors"

    # 1. 加载配置
    with open(config_path, "r") as f:
        model_cfg = json.load(f)

    arch = model_cfg.get("arch", "ViT-H-14")
    print(f"Loading OpenCLIP model: {arch}")

    # 2. 构建模型
    model = open_clip.create_model(arch, pretrained=None)
    model.to(device)
    model.eval()

    # 3. 加载权重
    state_dict = load_file(weight_path)
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully.")

    # 4. 只导出视觉分支
    vision = model

    scripted = torch.jit.script(vision)
    scripted.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,default='/home/data/zjp/pre-trained_model/clip/vit-H', required=True,
                        help="Directory containing open_clip_model.safetensors and open_clip_config.json")
    parser.add_argument("--output", type=str, default="ViT-H-14.pt",
                        help="Output TorchScript file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for tracing")
    args = parser.parse_args()

    convert_to_pt(args.model_dir, args.output, args.device)
