# huggingface_clip_surgery/hf_clip_loader.py

import torch
from transformers import CLIPProcessor
from .hf_clip_surgery import HuggingFaceCLIPSurgery


def load(path= '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP-vit-large-14', device="cuda" if torch.cuda.is_available() else "cpu",is_surgery=True):
    """
       加载本地 CLIP 模型，支持手术版本。

       Args:
           path (str): HuggingFace 权重目录路径
           device (str): 运行设备
           is_surgery (bool): 是否启用 attention surgery

       Returns:
           model (nn.Module), processor
       """
    if is_surgery:
        model = HuggingFaceCLIPSurgery(model_path=path)
    else:
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(path)

    processor = CLIPProcessor.from_pretrained(path)
    return model.to(device).eval(), processor