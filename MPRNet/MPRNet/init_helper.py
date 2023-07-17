import argparse
import logging
import random

import numpy as np
import torch


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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#
# def get_arguments():
#     parser = argparse.ArgumentParser(description="gauss")
#     parser.add_argument("-Ddavis", type=str, help="path to data", default='/data1/wangjiale/SSM-VOS/datasets/DAVIS2017/')
#     parser.add_argument("-Dyoutube", type=str, help="path to youtube-vos",
#                         default='/data1/wangjiale/SSM-VOS/datasets/YouTubeVOS/')
#     parser.add_argument("-batch", type=int, help="batch size", default=2)
#     parser.add_argument("-max_skip", type=int, help="max skip betwe en training frames", default=10)
#     parser.add_argument("-change_skip_step", type=int, help="change max skip per x iter", default=3000)
#     parser.add_argument("-total_iter", type=int, help="total iter num", default=400000)
#     parser.add_argument("-test_iter", type=int, help="evaluate per x iters", default=8000)
#     parser.add_argument("-log_iter", type=int, help="log per x iters", default=6)
#     parser.add_argument("-resume_path", type=str,
#                         default='/data1/wangjiale/weights/davis_youtube_resnet50_71999.pth')
#     parser.add_argument("-save", type=str, default='/data1/wangjiale/STM/train_gauss/weights1')
#     parser.add_argument("-sample_rate", type=float, default=0.08)
#     parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18']", default='resnet50')
#
#     return parser.parse_args()

# def get_arguments():
#     parser = argparse.ArgumentParser(description="GRU")
#     parser.add_argument("-Ddavis", type=str, help="path to data", default='/data1/wangjiale/SSM-VOS/datasets/DAVIS2017/')
#     parser.add_argument("-Dyoutube", type=str, help="path to youtube-vos",
#                         default='/data1/wangjiale/SSM-VOS/datasets/YouTubeVOS/')
#     parser.add_argument("-batch", type=int, help="batch size", default=2)
#     parser.add_argument("-max_skip", type=int, help="max skip betwe en training frames", default=10)
#     parser.add_argument("-change_skip_step", type=int, help="change max skip per x iter", default=3000)
#     parser.add_argument("-total_iter", type=int, help="total iter num", default=400000)
#     parser.add_argument("-test_iter", type=int, help="evaluate per x iters", default=8000)
#     parser.add_argument("-log_iter", type=int, help="log per x iters", default=6)
#     parser.add_argument("-resume_path", type=str,
#                         default='/data1/wangjiale/weights/davis_youtube_resnet50_7999.pth')
#     parser.add_argument("-save", type=str, default='/data1/wangjiale/STM/train_GRU/weights')
#     parser.add_argument("-sample_rate", type=float, default=0.08)
#     parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18']", default='resnet50')
#
#     return parser.parse_args()

def get_arguments():
    parser = argparse.ArgumentParser(description="GC")
    parser.add_argument("-Ddavis", type=str, help="path to data", default='/data1/wangjiale/SSM-VOS/datasets/DAVIS2017/')
    # parser.add_argument("-Ddavis", type=str, help="path to data", defaul t='/mnt/SSM-VOS/datasets/DAVIS2017/')
    parser.add_argument("-Dyoutube", type=str, help="path to youtube-vos",
                        default='/data1/wangjiale/SSM-VOS/datasets/YouTubeVOS/')
    parser.add_argument("-batch", type=int, help="batch size", default=4)
    parser.add_argument("-max_skip", type=int, help="max skip between training frames", default=4)
    parser.add_argument("-change_skip_step", type=int, help="change max skip per x iter", default=20000)
    parser.add_argument("-total_iter", type=int, help="total iter num", default=450000)
    parser.add_argument("-test_iter", type=int, help="evaluate per x iters", default=10000)
    parser.add_argument("-log_iter", type=int, help="log per x iters", default=12)
    parser.add_argument("-resume_path", type=str,
                        default='/data1/wangjiale/MPMC/weights_coco_短时传播/coco_pretrained_resnet50_199999.pth')
    #/data1/wangjiale/MPM/weights_coco_final
    parser.add_argument("-save", type=str, default='./weights_final_短时传播')
    parser.add_argument("-sample_rate", type=float, default=0.08)
    parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18']", default='resnet50')

    return parser.parse_args()