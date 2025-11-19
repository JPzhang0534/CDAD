import torch.nn as nn


from model.clip_surgery_model import CLIPSurgery

def build_clip_surgery_from_state(state_dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([
            k for k in state_dict if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIPSurgery(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # 删除 CLIP 多余 meta info
    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def CLIP_Model(clip_visual_encoder,clip_text_encoder):

    # 合并权重
    state_dict = {}
    state_dict.update({f"visual.{k}": v for k, v in clip_visual_encoder.state_dict().items()})
    state_dict.update(clip_text_encoder.state_dict())

    # 构建 surgery 模型
    clip_surgery = build_clip_surgery_from_state(state_dict)

    return clip_surgery