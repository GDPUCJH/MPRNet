from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
# cv2.setNumThreads(0)
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys
import random
from utils.helpers import *
from module.convGRU import ConvGRUCell
import math
from module.Diff_final import *
from module.ASPP import *
import skimage.measure as measure


# 即将改成第一帧不参与记忆，64个点，3*3卷积，/gauss_map待定, 这个先不加了，留到第二个点吧/，不加预训练
# 如果这个效果不好，可以去掉gauss_map

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_random_seed(123)


def torch_make_gauss(input_size, ct, scale1, scale2, iteration):
    gauss_batch = []
    ys, xs = ct
    for i in range(xs.size(0)):
        h, w = input_size
        center_x, center_y = xs[i], ys[i]
        x = torch.arange(0., w, 1).cuda()
        y = torch.arange(0., h, 1).unsqueeze(-1).cuda()
        # if iteration < 20:
        gauss = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / 12 / 12).cuda()
        # else:
        #     if scale1[i] == 0:
        #         scale2[i] = 30
        #     if scale1[i] == 0:
        #         scale2[i] = 30
        #     gauss = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / scale1[i] / scale2[i] / 4).cuda()
        # print("123")
        gauss_batch.append(gauss)
    gauss_batch = torch.stack(gauss_batch, dim=0).unsqueeze(dim=1).cuda()
    return gauss_batch


def extract_bboxes(mask):
    regions = measure.regionprops(mask)
    if len(regions):
        y1, x1, y2, x2 = regions[0].bbox
        bbox = np.array([x1, y1, x2, y2])
        return bbox.astype(np.int32)
    else:
        return np.array([0, 0, 0, 0])


def centerpoint(mask):
    mask = np.array(mask.cpu().data)
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    batch_list = {
        'h': [],
        'w': [],
    }
    for i in range(mask.shape[0]):
        # expand to ---  no, c, h, w
        bbox = extract_bboxes(mask[i].astype(np.int32))
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        h = torch.tensor(h).cuda()
        w = torch.tensor(w).cuda()
        # print(h)
        # print(w)
        batch_list['h'].append(h.unsqueeze(0))
        batch_list['w'].append(w.unsqueeze(0))

        # Make Batch
    for k, v in batch_list.items():
        batch_list[k] = torch.cat(v, dim=0)
    return batch_list['h'], batch_list['w']


class CorrCosine(nn.Module):
    def __init__(self):
        super().__init__()
        self.corr_conv = CorrConv()

    def forward(self, ref_features, cur_features):
        ref_features = F.normalize(ref_features, p=2, dim=1)
        cur_features = F.normalize(cur_features, p=2, dim=1)
        sim_matrix = self.corr_conv(ref_features, cur_features)
        return sim_matrix


class CorrConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ref_features, cur_features):
        batch_size, num_channels, ref_h, ref_w = ref_features.shape
        _, _, cur_h, cur_w = cur_features.shape
        cur_features = cur_features.permute(0, 2, 3, 1).contiguous().view(
            batch_size * cur_h * cur_w, num_channels, 1, 1)
        ref_features = ref_features.view(1, batch_size * num_channels, ref_h, ref_w)
        corr_features = F.conv2d(ref_features, cur_features, groups=batch_size)
        corr_features = corr_features.view(batch_size, cur_h, cur_w, ref_h, ref_w)

        return corr_features


class SelfStructure(nn.Module):
    def __init__(self, keep_topk):
        super().__init__()
        self.keep_topk = keep_topk
        self.corr = CorrCosine()

    def forward(self, ref_key, cur_key, ref_masks, video_name, frame_index):
        corr_features = self.corr(ref_key, cur_key)
        batch_size, cur_h, cur_w, ref_h, ref_w = corr_features.shape
        corr_features = corr_features.view(batch_size, cur_h * cur_w, ref_h * ref_w)

        probs = self.gauss_prob(corr_features, cur_key, ref_masks, video_name, frame_index)
        probs = probs.view(batch_size, 1, cur_h, cur_w)

        return cur_key * probs

    def gauss_prob(self, corr_features, cur_features, ref_mask, video_name, frame_index):
        batch_size, channels, cur_h, cur_w = cur_features.shape
        ref_h, ref_w = cur_h, cur_w
        cur_features = cur_features.view(batch_size, channels, cur_h * cur_w)

        ref_mask = F.interpolate(ref_mask, (ref_h, ref_w), mode='bilinear', align_corners=False)
        ref_mask = ref_mask.view(batch_size, 1, ref_h * ref_w)
        fg_corr = corr_features * (ref_mask > 0.5).type(torch.float32)

        topk_indices, corr = self.get_struct_info(fg_corr)  # 将当前帧的k个关键点选出来了
        # B, C, HW = cur_features.shape
        # cur_features = cur_features * self.get_gauss_map(topk_indices, cur_h, cur_w).view(batch_size, 1, cur_h*cur_w)  # 加了gauss_map
        # gauss_map = self.get_gauss_map(topk_indices, cur_h, cur_w, h, w, iteration).view(batch_size, 1, cur_h, cur_w)
        # weight = corr[[[b] for b in range(batch_size)], topk_indices]  # [b, k, c]
        fg_struct = cur_features[[[b] for b in range(batch_size)], :, topk_indices]  # [b, k, c]
        # fg_struct = fg_struct
        gauss_similarity = torch.bmm(fg_struct, cur_features) / math.sqrt(channels)  # 自己和自己做匹配
        gauss_similarity = F.softmax(gauss_similarity, dim=-1)
        # gauss_similarity = torch.sigmoid(gauss_similarity)
        gauss_similarity = gauss_similarity.mean(dim=1)  # 帧内匹配的概率，和均值
        logit = self.soft_aggregation(gauss_similarity)
        gauss_prob = F.softmax(logit, dim=0)  # 计算多目标的概率
        # f = (gauss_prob.view(batch_size+1, cur_h, cur_w)[0]*255.).cpu().numpy().astype(np.uint8)
        # # f = (high_fre[0].squeeze(0)* 255.).cpu().numpy().astype(np.uint8)
        # # print(f)
        # fore = Image.fromarray(f)
        #
        # seq_output_viz_path = '/data/jialewang/sigmoid/'+video_name
        #
        # if not os.path.exists(seq_output_viz_path):
        #     os.makedirs(seq_output_viz_path)
        #
        # fore.save(os.path.join(seq_output_viz_path, 'f{}.jpg'.format(frame_index)))

        return gauss_prob[0:-1]

    def get_struct_info(self, corr):
        corr = torch.sum(corr, dim=-1)
        # 选出的topk是当前帧的特征
        _, topk_indices = torch.topk(corr, self.keep_topk, dim=1)
        return topk_indices, corr  # b k c

    def get_gauss_map(self, topk_indices, H, W, h, w, iteration):
        # 这个测试的时候再调吧
        y, x = ((topk_indices.float() + 1) / W).ceil(), (topk_indices + 1) % W  # 第一个是H，第二个是W
        x = torch.mean(x.float(), dim=-1)
        y = torch.mean(y.float(), dim=-1)
        # y是纵坐标的中心点，x是横坐标的中心点
        gauss_map = torch_make_gauss((H, W), (y, x), h, w, iteration)
        return gauss_map

    def soft_aggregation(self, ps):
        em = torch.prod(1 - ps, dim=0)  # bg prob
        em = torch.cat((ps, em.unsqueeze(dim=0)), dim=0)
        return em


class ResBlock(nn.Module):
    def __init__(self, backbone, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = backbone
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Encoder_M(nn.Module):
    def __init__(self, backbone):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float()  # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float()  # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1


class Encoder_Q(nn.Module):
    def __init__(self, backbone):
        super(Encoder_Q, self).__init__()

        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1


class Refine(nn.Module):
    def __init__(self, backbone, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(backbone, planes, planes)
        self.ResMM = ResBlock(backbone, planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, mdim, scale_rate, backbone):
        super(Decoder, self).__init__()
        self.backbone = backbone
        self.convFM = nn.Conv2d(1536 // scale_rate, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(backbone, mdim, mdim)
        self.RF3 = Refine(backbone, 512 // scale_rate, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(backbone, 256 // scale_rate, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))

        return F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_in, q_in):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()

        mi = m_in.view(B, D_e, T * H * W)
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb

        qi = q_in.view(B, D_e, H * W)  # b, emb, HW

        p = torch.bmm(mi, qi)  # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)  # b, THW, HW

        mo = m_in.view(B, D_e, T * H * W)
        mem = torch.bmm(mo, p)  # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_e, H, W)

        return mem


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class STM(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(STM, self).__init__()
        self.backbone = backbone
        scale_rate = 1

        self.Encoder_M = Encoder_M(backbone)
        self.Encoder_Q = Encoder_Q(backbone)

        self.M_Key = nn.Conv2d(1024 // scale_rate, 512 // scale_rate, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.M_Value = nn.Conv2d(1024 // scale_rate, 512 // scale_rate, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.KV_Q_r4 = KeyValue(1024 // scale_rate, keydim=512 // scale_rate, valdim=512 // scale_rate)
        # self.Q_Value1 = nn.Conv2d(1024 // scale_rate, 512 // scale_rate, kernel_size=(1, 1), stride=1)
        # self.Q_Value3 = nn.Conv2d(1024 // scale_rate, 1 // scale_rate, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.decoder = Decoder(256, scale_rate, backbone)
        self.totalGRU = ConvGRUCell(512, 512, 5)
        self.structure = SelfStructure(32)
        self.prop = TemplateMatching()
        self.aspp = ASPP()
        self.memory = Memory()
        # self.channel_attention_pool = nn.AdaptiveAvgPool2d(1)
        # self.channel_attention_conv1 = nn.Conv2d(512, 512//16, 1, bias=False)
        # self.channel_attention_conv2 = nn.Conv2d(512//16, 2 * 512, 1, bias=False)
        #
        # self.spatial_attention_conv = nn.Conv2d(512, 2, 1, bias=False)

    def memorize(self, frame, masks, first_masks, n_objects):

        # memorize a frame
        B, K, H, W = masks.shape
        (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

        # make batch arg list
        batch_list = {'f': [], 'm': [], 'o': [], 'first_masks': []}
        for i in range(B):
            for o in range(1, n_objects[i] + 1):  # 1 - no
                batch_list['first_masks'].append(first_masks[i, o].unsqueeze(0).unsqueeze(0))
                batch_list['f'].append(frame[i].unsqueeze(0))
                batch_list['m'].append(masks[i, o].unsqueeze(0))
                batch_list['o'].append(
                    (torch.sum(masks[i, 1:o].unsqueeze(0), dim=1) +
                     torch.sum(masks[i, o + 1:n_objects[i] + 1].unsqueeze(0), dim=1)).clamp(0, 1))
        # Make Batch
        for k, v in batch_list.items():
            batch_list[k] = torch.cat(v, dim=0)

        r4, _, _, _ = self.Encoder_M(batch_list['f'], batch_list['m'], batch_list['o'])

        return r4, batch_list['first_masks'], batch_list['m']

    def soft_aggregation(self, ps, K, n_objects):
        B = len(n_objects)
        _, H, W = ps.shape

        em = ToCuda(torch.zeros(B, K, H, W))
        for i in range(B):
            begin = sum(n_objects[:i])
            end = begin + n_objects[i]
            em[i, 0] = torch.prod(1 - ps[begin:end], dim=0)  # bg prob
            em[i, 1:n_objects[i] + 1] = ps[begin:end]  # obj prob

        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))
        return logit

    def get_hidden(self, v1, hidden):
        hidden = self.totalGRU(v1, hidden)
        return hidden

    def Run_video(self, dataset, video, num_frames, num_objects, symbol):

        F_last, M_last = dataset.load_single_image(video, 0)
        F_last = F_last.unsqueeze(0).cuda()
        M_last = M_last.unsqueeze(0).cuda()
        First_M = M_last[:, :, 0]
        E_last = M_last
        pred = torch.zeros((num_frames, M_last.shape[3], M_last.shape[4])).cuda()
        # print(M_last.shape)
        pred[0] = torch.argmax(M_last[0, :, 0], dim=0)
        all_Ms = []
        all_Ms1 = []
        all_Ms1.append(F_last)
        first_key = None
        hidden = None

        for t in range(1, num_frames):
            # memorize
            F_, M_ = dataset.load_single_image(video, t)
            F_ = F_.unsqueeze(0).cuda()
            M_ = M_.unsqueeze(0).cuda()
            all_Ms.append(M_)
            all_Ms1.append(F_)
            # del M_
            # segment
            with torch.no_grad():
                logit, first_key, hidden = self.segment(F_last[:, :, 0], F_[:, :, 0], E_last[:, :, 0],
                                                        First_M, num_objects, t, first_key, hidden, video)
            E = F.softmax(logit, dim=1)
            del logit
            pred[t] = torch.argmax(E[0], dim=0)
            # print(M_.shape)
            # pred[t] = torch.argmax(M_[0, :, 0], dim=0)
            E_last = E.unsqueeze(2).cuda()
            F_last = F_.cuda()
            # torch.cuda.empty_cache()
        Ms = torch.cat(all_Ms, dim=2).cuda()
        Ms1 = torch.cat(all_Ms1, dim=2).cuda()

        return pred, Ms, Ms1

    def segment(self, pre_frame, cur_frame, pre_masks, first_masks, n_objects, frame_index, first_key, hidden,
                video_name):

        B, K, _, _ = pre_masks.shape
        [cur_frame], pad = pad_divide_by([cur_frame], 16, (cur_frame.size()[2], cur_frame.size()[3]))

        r4, r3, r2, _ = self.Encoder_Q(cur_frame)
        k4, v4 = self.KV_Q_r4(r4)
        # v41 = self.Q_Value1(r4)
        # v43 = self.Q_Value1(r4)

        batch_list = {
            'k4e': [],
            'v4e': [],
            # 'v41': [],
            # 'v43': [],
            'r3e': [],
            'r2e': []
        }
        for i in range(B):
            # expand to ---  no, c, h, w
            _k4e = k4[i].expand(n_objects[i], -1, -1, -1)
            _v4e = v4[i].expand(n_objects[i], -1, -1, -1)
            # _v41 = v41[i].expand(n_objects[i], -1, -1, -1)
            # _v43 = v43[i].expand(n_objects[i], -1, -1, -1)
            _r3e = r3[i].expand(n_objects[i], -1, -1, -1)
            _r2e = r2[i].expand(n_objects[i], -1, -1, -1)
            batch_list['k4e'].append(_k4e)
            batch_list['v4e'].append(_v4e)
            # batch_list['v41'].append(_v41)
            # batch_list['v43'].append(_v43)
            batch_list['r3e'].append(_r3e)
            batch_list['r2e'].append(_r2e)

        for k, v in batch_list.items():
            batch_list[k] = torch.cat(v, dim=0)

        p_r4, first_masks, pre_mask = self.memorize(pre_frame, pre_masks, first_masks, n_objects)

        if frame_index == 1:
            first_key = self.M_Key(p_r4)
            hidden = torch.zeros(p_r4.shape[0], 512, p_r4.shape[-2], p_r4.shape[-1]).cuda()
            # 传播分支
            pre_v4_hidden = self.M_Value(p_r4)
            hidden = pre_v4_hidden
            # hidden = self.get_hidden(pre_v4_hidden, hidden)
            memory_feature = first_key.unsqueeze(dim=2)
        else:
            pre_key = self.M_Key(p_r4)
            # 传播分支
            pre_v4_hidden = self.M_Value(p_r4)
            # hidden = self.get_hidden(pre_v4_hidden, hidden)
            hidden = pre_v4_hidden
            memory_feature = torch.cat((first_key.unsqueeze(dim=2), pre_key.unsqueeze(dim=2)), dim=2)

        matching_feature = self.memory(memory_feature, batch_list['k4e'])
        pre_mask = F.interpolate(pre_mask.unsqueeze(1), (p_r4.shape[-2], p_r4.shape[-1]), mode='bilinear',
                                 align_corners=False)
        # h, w = centerpoint(pre_mask.squeeze(1))

        # 匹配分支
        fg_feature = self.structure(first_key, batch_list['k4e'], first_masks, video_name, frame_index)

        # 传播分支
        global_feature = self.prop(pre_v4_hidden, batch_list['v4e'], hidden)
        # 传播分支后面接多尺度
        global_feature = self.aspp(global_feature)

        # # # # 方案二： 加入高频信息进行修正
        # high_fre = batch_list['v41'] - batch_list['k4e']
        # # # print(high_fre.shape)
        # f = (torch.mean(high_fre[0], dim=0) * 255.).cpu().numpy().astype(np.uint8)
        # # f = (high_fre[0].squeeze(0)* 255.).cpu().numpy().astype(np.uint8)
        # # print(f)
        # fore = Image.fromarray(f)
        #
        # seq_output_viz_path = '/data/jialewang/sigmoid/' + video_name
        #
        # if not os.path.exists(seq_output_viz_path):
        #     os.makedirs(seq_output_viz_path)
        # #
        # fore.save(os.path.join(seq_output_viz_path, 'f{}.jpg'.format(frame_index)))
        # x = matching_feature + global_feature
        # x = self.channel_attention_pool(x)
        # x = F.relu(self.channel_attention_conv1(x))
        # x = self.channel_attention_conv2(x)
        # x = x.view(x.shape[0], 2, 512, 1, 1)
        # x = F.softmax(x, 1)
        # x1 = matching_feature * x[:, 0, :, :, :]
        # x2 = global_feature * x[:, 1, :, :, :]
        # x  = x1 + x2
        # x = self.spatial_attention_conv(x)
        # x = F.softmax(x, 1)
        # x1 = x1* x[:, 0, :, :].unsqueeze(dim=1)
        # x2 = x2* x[:, 1, :, :].unsqueeze(dim=1)

        # high_fre = torch.sigmoid(high_fre)
        # high_feature = high_fre * batch_list['v41']
        logits = self.decoder(
            torch.cat((fg_feature, matching_feature, global_feature), dim=1),
            batch_list['r3e'], batch_list['r2e'])

        # logits = self.decoder(x, batch_list['r3e'], batch_list['r2e'])

        ps = F.softmax(logits, dim=1)  # no, h, w # 二分类
        ps = ps[:, 1]

        logit = self.soft_aggregation(ps, K, n_objects)  # 1, K, H, W

        if pad[2] + pad[3] > 0:
            logit = logit[:, :, pad[2]:-pad[3], :]

        if pad[0] + pad[1] > 0:
            logit = logit[:, :, :, pad[0]:-pad[1]]

        return logit, first_key, hidden

    def soft(self, ps):
        em = torch.prod(1 - ps, dim=0)  # bg prob
        em = torch.cat((ps, em.unsqueeze(dim=0)), dim=0)
        return em

    def forward(self, *args, **kwargs):
        if len(args) == 4:  # keys
            frames, masks, n_objects, epoch = args[0], args[1], args[2], args[3]
            pre_mask = masks[:, :, 0]
            first_key = None
            hidden = None
            est_masks_up = []
            for i in range(frames.shape[2] - 1):
                logit, first_key, hidden = self.segment(frames[:, :, i], frames[:, :, i + 1], pre_mask,
                                                        masks[:, :, 0],
                                                        n_objects,
                                                        i + 1, first_key, hidden, "")

                pre_mask = F.softmax(logit, dim=1).detach()
                est_masks_up.append(logit)
            return est_masks_up
        else:
            return self.Run_video(*args, **kwargs)
