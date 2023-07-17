from __future__ import division
import logging
import sys
sys.path.append('/data1/wangjiale/MPRNet')
import torch
from torch.utils import data
import torch.nn as nn
import init_helper
import torch.nn.functional as F

# general libs
from PIL import Image
import os
import argparse
#####freeze_bn()
### My libs
from dataset.dataset import DAVIS_MO_Test
from dataset.coco import Coco_MO_Train
from module.model_high_frequency import STM
from train_gauss.eval_new import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-Ddavis", type=str, help="path to davis",
                        default='/data1/wangjiale/SSM-VOS/datasets/DAVIS2017/')
    parser.add_argument("-Dcoco", type=str, help="path to coco", default='/data1/wangjiale/STM/')
    parser.add_argument("-batch", type=int, help="batch size", default=4)
    parser.add_argument("-max_skip", type=int, help="max skip between training frames", default=4)
    parser.add_argument("-change_skip_step", type=int, help="change max skip per x iter", default=1000)
    parser.add_argument("-total_iter", type=int, help="total iter num", default=250000)
    parser.add_argument("-test_iter", type=int, help="evaluat per x iters", default=10000)
    parser.add_argument("-resume_path", type=str,
                        default='/home/weights_coco_final_no_mg/coco_pretrained_resnet50_399999.pth')
    parser.add_argument("-log_iter", type=int, help="log per x iters", default=12)
    parser.add_argument("-save", type=str, default='./weights_coco_短时传播')
    parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18']", default='resnet50')
    return parser.parse_args()


logger = logging.getLogger()


def main():
    args = get_arguments()
    init_helper.init_logger('./logs_coco_短时传播.txt')
    init_helper.set_random_seed(123)

    # DAVIS_ROOT = args.Ddavis
    COCO_ROOT = args.Dcoco
    # palette = Image.open(DAVIS_ROOT + '/Annotations/480p/blackswan/00000.png').getpalette()

    torch.backends.cudnn.benchmark = True

    Trainset1 = Coco_MO_Train('{}train2017'.format(COCO_ROOT),
                              '{}annotations/instances_train2017.json'.format(COCO_ROOT))
    Trainloader1 = data.DataLoader(Trainset1, batch_size=1, num_workers=1, shuffle=True, pin_memory=True)
    loader_iter1 = iter(Trainloader1)

    model = nn.DataParallel(STM(args.backbone))

    # pth_path = args.resume_path
    #
    # logger.info('Loading weights:'+str(pth_path))
    #
    # model.load_state_dict(torch.load(pth_path))

    if torch.cuda.is_available():
        model.cuda()
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm1d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm3d):
            module.eval()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, eps=1e-8, betas=(0.9, 0.999))

    def adjust_learning_rate(iteration, power=0.9):
        lr = 5e-5 * pow((1 - 1.0 * iteration / (args.total_iter)), power)
        logger.info("learning rate is "+ str(lr))
        return lr

    accumulation_step = args.batch
    save_step = args.test_iter
    log_iter = args.log_iter

    loss_momentum = 0
    loss_total = 0
    max_jf = 0

    for iter_ in range(0, args.total_iter):

        if (iter_ + 1) % 1000 == 0:
            lr = adjust_learning_rate(iter_)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        try:
            Fs, Ms, num_objects, info = next(loader_iter1)
        except:
            loader_iter1 = iter(Trainloader1)
            Fs, Ms, num_objects, info = next(loader_iter1)

        # seq_name = info['name'][0]
        # num_frames = info['num_frames'][0].item()
        # num_frames = 3

        # Es = torch.zeros_like(Ms)
        # Es[:,:,0] = Ms[:,:,0]
        logit_list = model(Fs, Ms, num_objects, int(iter_ / 10000))
        n2_label = torch.argmax(Ms[:, :, 1], dim=1).long().cuda()
        n2_loss = criterion(logit_list[0], n2_label)
        # n2_loss1 = criterion(logit_list1[0], n2_label)
        # print(Fs.shape)
        # print(Fs[-1,:,0])

        n3_label = torch.argmax(Ms[:, :, 2], dim=1).long().cuda()
        n3_loss = criterion(logit_list[1], n3_label)
        # n3_loss1 = criterion(logit_list1[1], n3_label)

        n4_label = torch.argmax(Ms[:, :, 3], dim=1).long().cuda()
        n4_loss = criterion(logit_list[2], n4_label)
        # n4_loss1 = criterion(logit_list1[2], n4_label)

        loss = n2_loss + n3_loss + n4_loss
        # loss = loss / accumulation_step
        loss.backward()
        loss_momentum += loss.cpu().data.numpy()
        loss_total += loss.cpu().data.numpy()

        if (iter_ + 1) % accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (iter_ + 1) % log_iter == 0:
            logger.info('iteration:{}, loss:{}'.format(iter_, loss_momentum / log_iter))
            loss_momentum = 0

        if (iter_ + 1) % save_step == 0:
            logger.info('iteration:{}, total_loss:{}'.format(iter_, loss_total / save_step))
            loss_total = 0
            if not os.path.exists(args.save):
                os.makedirs(args.save)
            torch.save(model.state_dict(),
                       os.path.join(args.save, 'coco_pretrained_{}_{}.pth'.format(args.backbone, iter_)))

            model.eval()

            logger.info('Evaluate at iter: ' + str(iter_))
            g_res = evaluate(model, ['J', 'F'],1)

            if g_res[0] > max_jf:
                max_jf = g_res[0]

            logger.info('J&F: ' + str(g_res[0]) + ' Max J&F: ' + str(max_jf))

            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()

        del Fs
        del Ms
        del num_objects
        del info
        # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
