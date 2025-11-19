# huggingface_clip_surgery/hf_clip_surgery.py

import torch
import torch.nn as nn
from transformers import CLIPModel
from .surgery_attention import SurgeryAttention


class HuggingFaceCLIPSurgery(nn.Module):
    def __init__(self, model_path='/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP-vit-large-14', num_surgery_layers=6):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_path)

        # 替换视觉 transformer 后 num_surgery_layers 层
        layers = self.model.vision_model.encoder.layers
        dim = layers[0].self_attn.q_proj.in_features
        num_heads = layers[0].self_attn.num_heads

        for i in range(1, num_surgery_layers + 1):
            block = layers[-i]
            # new_attn = SurgeryAttention(dim=dim, num_heads=num_heads)

            from .surgery_attention import SurgerySelfAttentionWrapper

            new_attn = SurgerySelfAttentionWrapper(dim=dim, num_heads=num_heads)

            # 可选权重迁移：将 qkv 拆开合并为新的 qkv（用v权重）
            with torch.no_grad():
                v_w = block.self_attn.v_proj.weight
                v_b = block.self_attn.v_proj.bias
                qkv_weight = torch.cat([v_w, v_w, v_w], dim=0)
                qkv_bias = torch.cat([v_b, v_b, v_b], dim=0)
                # new_attn.qkv.weight.copy_(qkv_weight)
                # new_attn.qkv.bias.copy_(qkv_bias)
                # new_attn.proj.weight.copy_(block.self_attn.out_proj.weight)
                # new_attn.proj.bias.copy_(block.self_attn.out_proj.bias)
                new_attn.attn.qkv.weight.copy_(qkv_weight)
                new_attn.attn.qkv.bias.copy_(qkv_bias)
                new_attn.attn.proj.weight.copy_(block.self_attn.out_proj.weight)
                new_attn.attn.proj.bias.copy_(block.self_attn.out_proj.bias)

            block.self_attn = new_attn

    def encode_image(self, images):
        outputs = self.model.vision_model(pixel_values=images)
        pooled = outputs.pooler_output
        return self.model.visual_projection(pooled)

    def encode_text(self, input_ids, attention_mask=None):
        outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        pooled = last_hidden[torch.arange(last_hidden.shape[0]), input_ids.argmax(dim=-1)]
        return self.model.text_projection(pooled)

    def forward(self, images, input_ids, attention_mask=None):
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
