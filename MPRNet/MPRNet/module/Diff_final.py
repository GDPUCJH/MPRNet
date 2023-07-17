from __future__ import division
import torch
import torch.nn as nn


class TemplateMatching(nn.Module):
    def __init__(self, ch_in=1024, ch_out=512):
        super().__init__()

        # self.conv1 = nn.Conv2d(513, 512, kernel_size=(5, 5), padding=2, stride=1)
        self.blend = nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=1)

    def forward(self, pre_key, cur_key, hidden):
        B, C_key, H, W = pre_key.shape


        similarity = torch.cosine_similarity(pre_key.view(B, C_key, H * W), cur_key.view(B, C_key, H * W), dim=1)
        similarity = torch.exp(similarity)
        maxs, _ = torch.max(similarity, dim=-1)
        similarity = similarity / maxs.unsqueeze(dim=1)
        similarity = similarity.view(B, H, W).unsqueeze(dim=1)
        out = cur_key * (1 - similarity)
        out = self.blend(torch.cat((hidden * similarity, out),dim=1))
        # out = self.blend(torch.cat((hidden, cur_key), dim=1))
        # out = hidden * similarity + out
        return out