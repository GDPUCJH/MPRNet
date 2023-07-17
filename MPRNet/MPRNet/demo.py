from __future__ import division
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


### My libs
from dataset.dataset import DAVIS_MO_Test
from module.model_high_frequency import STM
from utils.helpers import overlay_davis
# from train_gauss.eval_new import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from evaldavis2017.davis2017.davis import DAVIS
from evaldavis2017.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaldavis2017.davis2017 import utils
from evaldavis2017.davis2017.results import Results
from scipy.optimize import linear_sum_assignment


palette = [
    0, 0, 0,
    0.5020, 0, 0,
    0, 0.5020, 0,
    0.5020, 0.5020, 0,
    0, 0, 0.5020,
    0.5020, 0, 0.5020,
    0, 0.5020, 0.5020,
    0.5020, 0.5020, 0.5020,
    0.2510, 0, 0,
    0.7529, 0, 0,
    0.2510, 0.5020, 0,
    0.7529, 0.5020, 0,
    0.2510, 0, 0.5020,
    0.7529, 0, 0.5020,
    0.2510, 0.5020, 0.5020,
    0.7529, 0.5020, 0.5020,
    0, 0.2510, 0,
    0.5020, 0.2510, 0,
    0, 0.7529, 0,
    0.5020, 0.7529, 0,
    0, 0.2510, 0.5020,
    0.5020, 0.2510, 0.5020,
    0, 0.7529, 0.5020,
    0.5020, 0.7529, 0.5020,
    0.2510, 0.2510, 0]
# palette = [i * 0.9 for i in palette]
palette = (np.array(palette) * 255).astype('uint8')


def Run_video111(model, dataset, video, num_frames, num_objects):
    F_last1, M_last1 = dataset.load_single_image(video, 0)
    F_last, M_last, _ = process_image_mask(F_last1, M_last1, 384, 384)
    E_last = M_last1
    pred = torch.zeros((num_frames, M_last1.shape[0], M_last1.shape[1])).cuda()
    pred[0] = torch.from_numpy(M_last1)

    M_last = To_onehot(M_last)
    # print(M_last[:,150,100:200])
    F_last = torch.from_numpy(np.transpose((F_last), (2, 0, 1))).float()
    M_last = torch.from_numpy(M_last).float()
    F_last = F_last.unsqueeze(0).cuda()
    M_last = M_last.unsqueeze(0).cuda()

    First_M = M_last
    all_Ms = []
    all_Ms.append(torch.from_numpy(np.expand_dims(np.expand_dims(F_last1, 0), 1)))
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

            all_Ms.append(torch.from_numpy(np.expand_dims(np.expand_dims(F_1, 0), 1)))
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

    Ms = torch.cat(all_Ms, dim=1).cuda()

    return pred, Ms


def demos(pred, Ms, output_mask_path, output_viz_path, num_objects, info):
    seq_name = info['name']
    num_frames = info['num_frames']

    # pred, Ms = Run_video(model, Testloader, seq_name, num_frames, num_objects, "test")
    pred = pred.cpu().numpy()
    Ms = Ms.cpu().numpy()

    # Save results for quantitative eval ######################
    seq_output_mask_path = os.path.join(output_mask_path, seq_name)
    if not os.path.exists(seq_output_mask_path):
        os.makedirs(seq_output_mask_path)

    for f in range(num_frames):
        img_E = Image.fromarray(pred[f].astype(np.uint8))
        img_E.putpalette(palette)
        # img_E.save(os.path.join(seq_output_mask_path, names[f][-9:-4]+'.png'))
        img_E.save(os.path.join(seq_output_mask_path, '{:05d}.png'.format(f)))

    seq_output_viz_path = os.path.join(output_viz_path, seq_name)
    if not os.path.exists(seq_output_viz_path):
        os.makedirs(seq_output_viz_path)

    for f in range(num_frames):
        # F_, M_ = dataset.load_single_image(video, t)
        # print(Ms.shape)
        pF = (Ms[0, :, f].transpose(1, 2, 0) * 255.).astype(np.uint8)
        pE = pred[f].astype(np.uint8)
        canvas = overlay_davis(pF, pE, palette)
        canvas = Image.fromarray(canvas)
        canvas.save(os.path.join(seq_output_viz_path, 'f{}.jpg'.format(f)))

    vid_path = os.path.join(output_viz_path, '{}.mp4'.format(seq_name))
    frame_path = os.path.join(output_viz_path, seq_name, 'f%d.jpg')
    os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(
        frame_path, vid_path))

def demo1(model,Testloader,output_mask_path,output_viz_path, pred1):
    for V in tqdm.tqdm(Testloader):
        num_objects, info = V
        print(V)
        seq_name = info['name']
        num_frames = info['num_frames']

        pred, _, Ms = model(Testloader, seq_name, num_frames, num_objects, "test")
        pred = pred.cpu().numpy()
        Ms = Ms.cpu().numpy()

        # Save results for quantitative eval ######################
        seq_output_mask_path = os.path.join(output_mask_path,seq_name)
        if not os.path.exists(seq_output_mask_path):
            os.makedirs(seq_output_mask_path)

        for f in range(num_frames):
            img_E = Image.fromarray(pred[f].astype(np.uint8))
            img_E.putpalette(palette)
            img_E.save(os.path.join(seq_output_mask_path, '{:05d}.png'.format(f)))


        seq_output_viz_path = os.path.join(output_viz_path,seq_name)
        if not os.path.exists(seq_output_viz_path):
            os.makedirs(seq_output_viz_path)

        for f in range(num_frames):
            # F_, M_ = dataset.load_single_image(video, t)
            # print(Ms.shape)
            pF = (Ms[0,:,f].transpose(1,2,0) * 255.).astype(np.uint8)
            pE = pred[f].astype(np.uint8)
            canvas = overlay_davis(pF, pE, palette)
            canvas = Image.fromarray(canvas)
            canvas.save(os.path.join(seq_output_viz_path, 'f{}.jpg'.format(f)))

        vid_path = os.path.join(output_viz_path, '{}.mp4'.format(seq_name))
        frame_path = os.path.join(output_viz_path, seq_name, 'f%d.jpg')
        os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(frame_path, vid_path))



if __name__ == "__main__":
    torch.set_grad_enabled(False) # Volatile
    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc",  default='2')
        parser.add_argument("-s", type=str, help="set",  default='val')
        parser.add_argument("-y", type=int, help="year", default=17)
        parser.add_argument("-D", type=str, help="path to data",default='/data/jialewang/SSM-VOS/datasets/DAVIS2017/')
        parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18','resnest101']",default='resnet50')
        parser.add_argument("-p", type=str, help="path to weights",default='/data/jialewang/MPM/train_GC/weights_coco/davis_youtube_resnet50_119999.pth')
        parser.add_argument("-output_mask_path", type=str, help="path to segmentation maps",default='./output')
        parser.add_argument("-output_viz_path", type=str, help="path to videos",default='./viz')
        return parser.parse_args()

    args = get_arguments()

    GPU = args.g
    YEAR = args.y
    SET = args.s
    DATA_ROOT = args.D
    output_mask_path = args.output_mask_path
    output_viz_path = args.output_viz_path

    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)
    if not os.path.exists(output_viz_path):
        os.makedirs(output_viz_path)  

    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on DAVIS')

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    Testloader = DAVIS_MO_Test('/data/jialewang/SSM-VOS/datasets/DAVIS2017/', resolution='480p',
                               imset='20{}/{}.txt'.format(17, 'val'), single_object=False)

    pred = torch.zeros((num_frames, M_last.shape[3], M_last.shape[4])).cuda()

    F_, M_, _ = Testloader.load_single_image(video, t)

    model = nn.DataParallel(STM(args.backbone))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    pth = args.p

    model.load_state_dict(torch.load(pth))
    metric = ['J','F']
    demo1(model,Testloader,output_mask_path,output_viz_path, pred1)
    # pred, Ms1, './output_0.76', './viz_0.76', num_objects, info
