from __future__ import division
import torch
import torch.nn as nn


class TemplateMatching(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pre_key, cur_key, hidden):
        B, C_key, H, W = pre_key.shape

        similarity = torch.cosine_similarity(pre_key.view(B, C_key, H * W), cur_key.view(B, C_key, H * W), dim=1)
        similarity = torch.exp(similarity)
        maxs, _ = torch.max(similarity, dim=-1)
        similarity = similarity / maxs.unsqueeze(dim=1)
        similarity = similarity.view(B, H, W).unsqueeze(dim=1)
        out = cur_key * (1 - similarity) + similarity * hidden
        return out