from __future__ import division
import logging
import sys
# sys.path.append('/data/jialewang/MPMC')
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
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
import time

### My libs
from dataset.dataset import DAVIS_MO_Test
from module.model_high_frequency import STM

# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

from evaldavis2017.davis2017.davis import DAVIS
from evaldavis2017.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaldavis2017.davis2017 import utils
from evaldavis2017.davis2017.results import Results
from scipy.optimize import linear_sum_assignment
from demo import *

logger = logging.getLogger()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_random_seed(123)

def evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    for ii in range(all_gt_masks.shape[0]):
        if 'J' in metric:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        if 'F' in metric:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
    return j_metrics_res, f_metrics_res


def To_onehot(mask):
    M = np.zeros((6, mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for k in range(6):
        M[k] = (mask == k).astype(np.uint8)
    return M


# def crop(image, bounding_mask, margin_left, margin_right, margin_top, margin_bottom):
#     img_h, img_w = bounding_mask.shape
#     crop_box = np.array([0, 0, img_w, img_h], dtype=np.int32)
#
#     if not np.isclose(bounding_mask, 0).all():
#         x1, y1, box_w, box_h = cv2.boundingRect(bounding_mask.astype(np.uint8))
#         x2 = x1 + box_w
#         y2 = y1 + box_h
#
#         x1 -= box_w * margin_left
#         x2 += box_w * margin_right
#         y1 -= box_h * margin_top
#         y2 += box_h * margin_bottom
#
#         if x2 - x1 > 5 and y2 - y1 > 5:
#             crop_box = np.array([x1, y1, x2, y2], dtype=np.int32)
#
#             image = crop_image(image, crop_box)
#             bounding_mask = crop_image(bounding_mask, crop_box)
#
#     return image, bounding_mask, crop_box


def resize(image, bounding_mask, out_h, out_w):
    image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    bounding_mask = cv2.resize(bounding_mask, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    # bounding_mask = (bounding_mask > 0.5).astype(np.uint8)
    bounding_mask = np.around(bounding_mask).astype(np.uint8)
    return image, bounding_mask


def process_image_mask(image, bounding_mask, out_h, out_w):
    image, bounding_mask, crop_box = crop(image, bounding_mask, 0.5, 0.5,
                                          0.5, 0.5)
    image, bounding_mask = resize(image, bounding_mask, out_h, out_w)

    return image, bounding_mask, crop_box


def Run_video111(model, dataset, video, num_frames, num_objects):
    F_last1, M_last1 = dataset.load_single_image(video, 0)
    F_last, M_last, _ = process_image_mask(F_last1, M_last1, 384, 384)
    E_last = M_last1
    pred = torch.zeros((num_frames, M_last1.shape[0], M_last1.shape[1])).cuda()
    pred[0] = torch.from_numpy(M_last1)

    M_last = To_onehot(M_last)
    F_last = torch.from_numpy(np.transpose((F_last), (2, 0, 1))).float()
    M_last = torch.from_numpy(M_last).float()
    F_last = F_last.unsqueeze(0).cuda()
    M_last = M_last.unsqueeze(0).cuda()

    First_M = M_last
    all_Ms = []
    first_key = None
    hidden = None

    for t in range(1, num_frames):
        F_1, M_ = dataset.load_single_image(video, t)
        F_, E_last, crop_box = process_image_mask(
            F_1, E_last, 384, 384)
        # segment
        with torch.no_grad():
            E_last = To_onehot(E_last)
            M_ = To_onehot(M_)

            F_ = torch.from_numpy(np.transpose((F_), (2, 0, 1))).float()
            E_last = torch.from_numpy(E_last)
            M_ = torch.from_numpy(M_)

            F_ = F_.unsqueeze(0).cuda()
            M_ = M_.unsqueeze(0).unsqueeze(2).cuda()
            E_last = E_last.unsqueeze(0).cuda()

            all_Ms.append(M_)
            del M_
            logit, first_key, hidden = model(F_last, F_, E_last,
                                             First_M, num_objects, t, first_key, hidden)

        E = F.softmax(logit, dim=1)
        x1, y1, x2, y2 = crop_box
        box_w = x2 - x1
        box_h = y2 - y1
        E = F.interpolate(E, (box_h, box_w), mode='bilinear', align_corners=False)
        E1 = torch.argmax(E[0], dim=0)
        E = crop_image_back(E1.cpu(), crop_box, F_last1.shape[0], F_last1.shape[1])
        del logit
        pred[t] = torch.from_numpy(E.astype(np.uint8))
        E_last = E.astype(np.uint8)
        F_last = F_.cuda()

    Ms = torch.cat(all_Ms, dim=2).cuda()

    return pred, Ms


def evaluate(model, metric, epoch):
    # Containers
    Testloader = DAVIS_MO_Test('/data1/wangjiale/SSM-VOS/datasets/DAVIS2017/', resolution='480p',
                               imset='20{}/{}.txt'.format(17, 'val'), single_object=False)
    total_time = 0
    total_frame = 0
    # Testloader = DAVIS_MO_Test('/data/jialewang/SSM-VOS/datasets/YouTubeVOS/valid/')
    metrics_res = {}
    if 'J' in metric:
        metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metric:
        metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    for V in tqdm.tqdm(Testloader):
        num_objects, info = V
        seq_name = info['name']
        num_frames = info['num_frames']
        # print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))

        # pred, Ms, Ms1 = Run_video(model, Testloader, seq_name, num_frames, num_objects,"test")
        begin = time.time()
        pred, Ms, Ms1 = model(Testloader, seq_name, num_frames, num_objects,"test")
        end = time.time()
        total_time += (end-begin)
        total_frame += (num_frames-1)
        # demos(pred, Ms1,'./outputs_result_多帧','./vizs_多帧',num_objects, info)
        pred = pred.cpu().numpy()
        Ms = Ms.cpu().numpy()
        # all_res_masks = Es[0].cpu().numpy()[1:1+num_objects]
        all_res_masks = np.zeros((num_objects, pred.shape[0], pred.shape[1], pred.shape[2]))
        for i in range(1, num_objects + 1):
            all_res_masks[i - 1, :, :, :] = (pred == i).astype(np.uint8)
        all_res_masks = all_res_masks[:, 1:-1, :, :]
        all_gt_masks = Ms[0][1:1 + num_objects]
        all_gt_masks = all_gt_masks[:, :-1, :, :]
        j_metrics_res, f_metrics_res = evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                metrics_res['J']["M"].append(JM)
                metrics_res['J']["R"].append(JR)
                metrics_res['J']["D"].append(JD)
            if 'F' in metric:
                [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                metrics_res['F']["M"].append(FM)
                metrics_res['F']["R"].append(FR)
                metrics_res['F']["D"].append(FD)

    J, F = metrics_res['J'], metrics_res['F']
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    # return g_res
    return g_res

def init_logger(log_path):
    logger = logging.getLogger()
    format_str = '[%(asctime)s %(filename)s#%(lineno)3d] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter(format_str))
        logger.addHandler(fh)

if __name__ == "__main__":
    # init_logger('/home/davis2017_result.txt')
    torch.set_grad_enabled(False) # Volatile
    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc",  default='6')
        parser.add_argument("-s", type=str, help="set",  default='val')
        parser.add_argument("-y", type=int, help="year", default=17)
        parser.add_argument("-D", type=str, help="path to data",default='/data/jialewang/SSM-VOS/datasets/DAVIS2017/')
        parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18','resnest101']",default='resnet50')
        parser.add_argument("-p", type=str, help="path to weights",default='/data/jialewang/MPM/train_GC/weights_final_k=32/davis_youtube_resnet50_399999.pth')
        parser.add_argument("-output_mask_path", type=str, help="path to segmentation maps",default='./output_0.76')
        parser.add_argument("-output_viz_path", type=str, help="path to videos",default='./viz_0.76')
        return parser.parse_args()

    args = get_arguments()

    GPU = args.g
    YEAR = args.y
    SET = args.s
    DATA_ROOT = args.D

    # Model and version
    MODEL = 'STM'
    # print(MODEL, ': Testing on DAVIS')

    # os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    # if torch.cuda.is_available():
    #     logger.info('using Cuda devices, num:', torch.cuda.device_count())
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU

    # Testloader = DAVIS_MO_Test('/data/jialewang/SSM-VOS/datasets/DAVIS2017/', resolution='480p',
    #                            imset='20{}/{}.txt'.format(16, 'val'), single_object=True)
    model = nn.DataParallel(STM(args.backbone))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    pth = args.p

    model.load_state_dict(torch.load(pth))
    metric = ['J', 'F']
    a = evaluate(model, metric, 0)
    print(a)
    # logger = logging.getLogger()
    # logger.info(str(a))
    # logger.info(str(total_frame/total_time))
