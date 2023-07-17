from __future__ import division
import logging
import sys
sys.path.append('/data1/wangjiale/MPRNet')
import torch
from torch.utils import data
import torch.nn as nn

# general libs
from PIL import Image
import os
import argparse
import random
import init_helper

### My libs
from dataset.davis import DAVIS_MO_Train
from dataset.youtube import Youtube_MO_Train
from module.model_high_frequency import STM
# from module.model_gauss_map import STM
from train_gauss.eval_new import evaluate


os.environ["CUDA_VISIBLE_DEVICES"] = '1'


logger = logging.getLogger()

def main():

    args = init_helper.get_arguments()
    init_helper.init_logger('./logs_final_短时传播.txt')
    init_helper.set_random_seed(123)

    rate = args.sample_rate

    DATA_ROOT = args.Ddavis
    # palette = Image.open(DATA_ROOT + '/Annotations/480p/blackswan/00000.png').getpalette()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model = nn.DataParallel(STM(args.backbone))

    Trainset = DAVIS_MO_Train(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(17, 'train'),
                              single_object=False)
    Trainloader = data.DataLoader(Trainset, batch_size=1, num_workers=1, shuffle=True, drop_last=True)
    loader_iter = iter(Trainloader)

    YOUTUBE_ROOT = args.Dyoutube
    Trainset1 = Youtube_MO_Train('{}train/'.format(YOUTUBE_ROOT))
    Trainloader1 = data.DataLoader(Trainset1, batch_size=1, num_workers=1, shuffle=True, drop_last=True)
    loader_iter1 = iter(Trainloader1)

    pth_path = args.resume_path

    logger.info('Loading weights:'+str(pth_path))

    model.load_state_dict(torch.load(pth_path))

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
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, eps=1e-8, betas=[0.9, 0.999])
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-6)

    def adjust_learning_rate(iteration, power=0.9):
        lr = 2e-5 * pow((1 - 1.0 * iteration / (args.total_iter)), power)
        logger.info("learning rate is "+ str(lr))
        return lr

    accumulation_step = args.batch
    save_step = args.test_iter
    log_iter = args.log_iter

    loss_momentum = 0
    loss_total = 0
    change_skip_step = args.change_skip_step
    max_skip = 4
    skip_n = max_skip
    max_jf = 0
    # lambda1 = 1
    # lambda2 = 1
    # Trainset1.change_skip(skip_n)
    # loader_iter1 = iter(Trainloader1)
    # Trainset.change_skip(skip_n)
    # loader_iter = iter(Trainloader)

    for iter_ in range(0, args.total_iter):

        if (iter_ + 1) % 1000 == 0:
            lr = adjust_learning_rate(iter_)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if (iter_ + 1) % change_skip_step == 0:
            if skip_n > 0:
                skip_n -= 1
            Trainset1.change_skip(skip_n//2)
            loader_iter1 = iter(Trainloader1)
            Trainset.change_skip(skip_n)
            loader_iter = iter(Trainloader)

        # try:
        #     Fs, Ms, num_objects, info = next(loader_iter)
        # except:
        #     loader_iter = iter(Trainloader)
        #     Fs, Ms, num_objects, info = next(loader_iter)

        if random.random() < rate:
            try:
                Fs, Ms, num_objects, info = next(loader_iter)
            except:
                loader_iter = iter(Trainloader)
                Fs, Ms, num_objects, info = next(loader_iter)
        else:
            try:
                Fs, Ms, num_objects, info = next(loader_iter1)
            except:
                loader_iter1 = iter(Trainloader1)
                Fs, Ms, num_objects, info = next(loader_iter1)

        logit_list = model(Fs, Ms, num_objects, int(iter_ / 10000))
        # print(Ms.shape)
        n2_label = torch.argmax(Ms[:, :, 1], dim=1).long().cuda()
        n2_loss = criterion(logit_list[0], n2_label)
        # n2_loss1 = criterion(logit_list1[0], n2_label)

        n3_label = torch.argmax(Ms[:, :, 2], dim=1).long().cuda()
        n3_loss = criterion(logit_list[1], n3_label)
        # n3_loss1 = criterion(logit_list1[1], n3_label)

        n4_label = torch.argmax(Ms[:, :, 3], dim=1).long().cuda()
        n4_loss = criterion(logit_list[2], n4_label)
        # n4_loss1 = criterion(logit_list1[2], n4_label)

        # n5_label = torch.argmax(Ms[:, :, 4], dim=1).long().cuda()
        # n5_loss = criterion(logit_list[3], n5_label)

        loss = n2_loss + n3_loss + n4_loss
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
                       os.path.join(args.save, 'davis_youtube_{}_{}.pth'.format(args.backbone, str(iter_))))

            model.eval()

            logger.info('Evaluate at iter: ' + str(iter_))
            g_res = evaluate(model, ['J', 'F'], int((iter_+1) / 10000))

            if g_res[0] > max_jf:
                max_jf = g_res[0]

            logger.info('J&F: ' + str(g_res) + ' Max J&F: ' + str(max_jf))

            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()

        # torch.cuda.empty_cache()

        del Fs
        del Ms
        del num_objects
        del info

if __name__ == '__main__':
    main()


# nohup python3 train_davis.py &