from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemplateMatching(nn.Module):
    def __init__(self, ch_in=1024, ch_out=512):
        super().__init__()

        self.blend = nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=1)

    def forward(self, pre_key, pre_mask, cur_key, hidden):
        B, C_key, H, W = pre_key.shape

        # # 把current_key给操作下
        # # print(cur_key.shape)
        # # print(pre_mask.shape)
        # pre_mask = F.interpolate(pre_mask.unsqueeze(1), (H, W), mode='bilinear', align_corners=False)
        # # print(cur_key.shape)
        # # print(pre_mask.shape)
        # upheat = torch.cat([cur_key, pre_mask], dim=1)
        # motion_w = self.motion_conv_w(upheat)
        # motion_w = torch.sigmoid(motion_w)
        # motion_b = self.motion_conv_b(upheat)
        # motion_b = torch.sigmoid(motion_b)
        # coarse_current_mask = motion_w*pre_mask+motion_b # b,1,h,w
        # cur_key = self.M_Value(torch.cat([cur_key, coarse_current_mask], dim=1))

        similarity = torch.cosine_similarity(pre_key.view(B, C_key, H * W), cur_key.view(B, C_key, H * W), dim=1)
        similarity = torch.exp(similarity)
        maxs, _ = torch.max(similarity, dim=-1)
        similarity = similarity / maxs.unsqueeze(dim=1)
        similarity = similarity.view(B, H, W).unsqueeze(dim=1)
        out = cur_key * (1 - similarity)
        out = self.blend(torch.cat((hidden*similarity, out),dim=1))
        return out