# huggingface_clip_surgery/surgery_attention.py

import torch
import torch.nn as nn

class SurgeryAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[2], qkv[2], qkv[2]  # q = k = v

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out

class SurgerySelfAttentionWrapper(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = SurgeryAttention(dim=dim, num_heads=num_heads)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # HuggingFace 格式 -> 你的格式
        return self.attn(hidden_states), None