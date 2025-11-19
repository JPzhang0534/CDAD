#  找到clip img中与text 相关的图像特征

import torch
import torch.nn.functional as F


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

@torch.no_grad()
# def extract_relevant_image_features_openclip(
#     image: torch.Tensor,              # shape: [B, 3, H, W]
#     text: str,
#     target_word: str,
#     model,
#     processor,
#     k: int = 5,
#     device: str = "cuda"
# ):
#     """
#     提取 OpenCLIP 模型中图像 patch 特征中与目标词相关的 patch
#
#     返回：
#       - top-k patch 特征（已投影）
#       - patch 索引
#       - 相似度分数
#     """
#
#     # Step 1: 预处理输入
#     inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
#     image_input = inputs["pixel_values"]        # [1, 3, H, W]
#     input_ids = inputs["input_ids"]             # [1, T]
#
#     # Step 2: 前向提取特征
#     outputs = model(**inputs, output_hidden_states=True)
#
#     # 文本所有 token 特征 [1, T, 1024]
#     text_feats = outputs.text_model_output.last_hidden_state  # [1, T, 1024]
#     # 图像 patch 特征 [1, N, 1280]
#     image_feats = outputs.vision_model_output.last_hidden_state  # [1, N, 1280]
#
#     # Step 3: 获取 tokenizer token 列表
#     tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])
#     token_index = find_token_index_by_word(tokens, target_word)
#     target_token_feat = text_feats[0, token_index]  # [1024]
#
#     # Step 4: 去掉 CLS token，进行 projection
#     patch_feats = image_feats[0, 1:, :]  # [N-1, 1280]
#     patch_feats_proj = model.visual_projection(patch_feats)  # [N-1, D]
#     token_feat_proj = model.text_projection(target_token_feat)  # [D]
#
#     # Step 5: 相似度计算
#     similarity = F.cosine_similarity(patch_feats_proj, token_feat_proj.unsqueeze(0), dim=-1)  # [N-1]
#
#     # Step 6: top-k
#     topk_indices = similarity.topk(k=k).indices  # [k]
#     relevant_feats = patch_feats_proj[topk_indices]  # [k, D]
#
#     return relevant_feats, topk_indices, similarity

def extract_relevant_image_features_hfclip(
    image,                # Tensor [B, 3, 224, 224]
    text: str,
    target_word: str,
    model,                # HuggingFace CLIPModel
    tokenizer,            # CLIPTokenizer
    k: int = 5,
    device="cuda"
):
    # Step 1: Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Step 2: Feature extraction
    image_features = model.vision_model(image.to(device)).last_hidden_state  # [B, 257, 1024]
    text_features = model.text_model(**inputs).last_hidden_state             # [B, 77, 1024]

    # Step 3: Locate target word
    token_list = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    token_index = find_token_index_by_word(token_list, target_word)
    word_feat = text_features[0, token_index]  # [1024]

    # Step 4: Remove CLS from image features
    patch_feats = image_features[0, 1:, :]  # [256, 1024]

    patch_feats_proj = model.visual_projection(patch_feats)  # [N-1, D]
    token_feat_proj = model.text_projection(word_feat)  # [D]

    # Step 5: Cosine similarity
    sim = F.cosine_similarity(patch_feats_proj, token_feat_proj.unsqueeze(0), dim=-1)  # [256]
    topk_idx = sim.topk(k).indices

    # relevant_feats = patch_feats_proj[topk_idx] # [k, 1024]

    relevant_feats = patch_feats[topk_idx]  # [k, 1280]

    return relevant_feats, topk_idx, sim


from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPVisionModel
from PIL import Image

# # clip_model_path = '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP-vit-large-14'   # CLIP-vit-large-14
clip_model_path = '/home/data/zjp/code/bridge/lyf/VGDiffZero-main/pre-trained_model/CLIP/CLIP_vit_H-14'   # CLIP-vit-large-14
# # 加载模型（可选用 "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"）
model = CLIPModel.from_pretrained(clip_model_path).eval()
# # clip_model = CLIPModel.from_pretrained(clip_model_path).eval()
#
# preprocess = CLIPProcessor.from_pretrained(clip_model_path)

from open_clip import create_model_and_transforms, tokenize

# model, _, preprocess = create_model_and_transforms(clip_model_path, pretrained='laion2b_s32b_b79k')
model = model.eval().cuda()
# image = preprocess(Image.open("your_image.jpg")).unsqueeze(0).cuda()

tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
processor = CLIPProcessor.from_pretrained(clip_model_path)

#
# # 图像 & 文本输入
# image = Image.open("/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/diffseg_testpng.jpg").convert("RGB")
# text = "a photo of a dog running in a field with a frisbee in its mouth"
# target_word = "dog"
#
# # 获取与 "cat" 最相关的图像 patch 特征
# relevant_feats, patch_indices, similarity = extract_relevant_image_features(
#     image, text, target_word, clip_model, processor, k=5
# )
#
# print("Top-k relevant patch indices:", patch_indices.tolist())
# print("Shape of returned patch features:", relevant_feats.shape)  # [k, C]

# from open_clip import create_model_and_transforms, tokenize
# model, _, preprocess = create_model_and_transforms(clip_model_path, pretrained='laion2b_s32b_b79k')
# model = model.eval().cuda()

img_pth = "/home/data/zjp/code/bridge/lyf/VGDiffZero-main/VGDiffZero-main/outputs/attn_db/diffseg/diffseg_testpng.jpg"
# 图像处理
img = Image.open(img_pth).convert("RGB")
inputs = processor(images=img, return_tensors="pt")
image_tensor = inputs["pixel_values"].cuda()
# image = preprocess(Image.open(img_pth)).unsqueeze(0).cuda()
text = "a photo of a dog running in a field with a frisbee in its mouth"
target_word = "dog"
# 提取 "cat" 对应的 patch 特征
relevant_feats, patch_ids, sim = extract_relevant_image_features_hfclip(
    image=image_tensor,
    text=text,
    target_word=target_word,
    model=model,
    tokenizer=tokenizer,  # OpenCLIP 将 tokenizer 封装在 model
    k=5
)

print("Top-k patch IDs:", patch_ids.tolist())
print("Shape of patch features:", relevant_feats.shape)  # [k, D]
