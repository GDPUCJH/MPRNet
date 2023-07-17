# import os
# import os.path as osp
# import numpy as np
# from PIL import Image, ImageOps
#
# import torch
# import torchvision
# from torch.utils import data
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
#
# import glob
#
#
# class DAVIS_MO_Test(data.Dataset):
#     # for multi object, do shuffling
#
#     def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
#         self.root = root
#         self.mask_dir = os.path.join(root, 'Annotations', resolution)
#         self.mask480_dir = os.path.join(root, 'Annotations', '480p')
#         self.image_dir = os.path.join(root, 'JPEGImages', resolution)
#         _imset_dir = os.path.join(root, 'ImageSets')
#         _imset_f = os.path.join(_imset_dir, imset)
#
#         self.videos = []
#         self.num_frames = {}
#         self.num_objects = {}
#         self.shape = {}
#         self.size_480p = {}
#         with open(os.path.join(_imset_f), "r") as lines:
#             for line in lines:
#                 _video = line.rstrip('\n')
#                 self.videos.append(_video)
#                 self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
#                 _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
#                 self.num_objects[_video] = np.max(_mask)
#                 self.shape[_video] = np.shape(_mask)
#                 _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
#                 self.size_480p[_video] = np.shape(_mask480)
#
#         self.K = 8
#         self.single_object = single_object
#
#     def __len__(self):
#         return len(self.videos)
#
#     def To_onehot(self, mask):
#         M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
#         for k in range(self.K):
#             M[k] = (mask == k).astype(np.uint8)
#         return M
#
#     def All_to_onehot(self, masks):
#         Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
#         for n in range(masks.shape[0]):
#             Ms[:, n] = self.To_onehot(masks[n])
#         return Ms
#
#     def __getitem__(self, index):
#         video = self.videos[index]
#         info = {}
#         info['name'] = video
#
#         info['num_frames'] = self.num_frames[video]
#         info['size_480p'] = self.size_480p[video]
#
#         N_frames = np.empty((self.num_frames[video],) + self.shape[video] + (3,), dtype=np.float32)
#         N_masks = np.empty((self.num_frames[video],) + self.shape[video], dtype=np.uint8)
#         num_objects = torch.LongTensor([int(self.num_objects[video])])
#         return num_objects, info
#
#     def load_single_image(self, video, f):
#         N_frames = np.empty((1,) + (self.shape[video][0],) + (self.shape[video][1],) + (3,), dtype=np.float32)
#         N_masks = np.empty((1,) + self.shape[video], dtype=np.uint8)
#
#         img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
#         # mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))
#         # b = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
#         # im = Image.fromarray(b)
#         # im.save("out.png")
#
#         N_frames[0] = np.array(ImageOps.exif_transpose(Image.open(img_file)).convert('RGB')) / 255.
#         try:
#             mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))
#             N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
#         except:
#             N_masks[0] = 255
#         Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
#         if self.single_object:
#             N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
#             Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
#             num_objects = torch.LongTensor([int(1)])
#             return Fs, Ms, num_objects, info
#         else:
#             Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
#             # num_objects = torch.LongTensor([int(self.num_objects[video])])
#             return F.interpolate(Fs, (1024,1024), mode='bilinear', align_corners=False), F.interpolate(Ms, (1024,1024), mode='bilinear', align_corners=False)
#
#
# if __name__ == '__main__':
#     pass

import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob


class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations')
        self.mask480_dir = os.path.join(root, 'Annotations')
        self.image_dir = os.path.join(root, 'JPEGImages')
        # _imset_dir = os.path.join(root, 'ImageSets')
        # _imset_f = os.path.join(_imset_dir, imset)
        self.youtube_videos = [i.split('/')[-1] for i in glob.glob(os.path.join(self.image_dir, '*'))]

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        # with open(os.path.join(_imset_f), "r") as lines:
        for line in self.youtube_videos:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                first_name = sorted(glob.glob(os.path.join(self.mask_dir, _video, '*.png')))[0]
                # print(first_name)
                _mask = np.array(Image.open(first_name).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, first_name)).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 6

    def __len__(self):
        return len(self.videos)

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:, n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        print(video)
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((self.num_frames[video],) + self.shape[video] + (3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],) + self.shape[video], dtype=np.uint8)
        num_objects = torch.LongTensor([int(self.num_objects[video])])
        return num_objects, info

    def load_single_image(self, video, f):
        N_frames = np.empty((1,) + self.shape[video] + (3,), dtype=np.float32)
        N_masks = np.empty((1,) + self.shape[video], dtype=np.uint8)

        # img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
        img_files = glob.glob(os.path.join(self.image_dir, video, '*.jpg'))
        img_files.sort()
        # print(f)
        # print(img_files[f])

        N_frames[0] = np.array(Image.open(img_files[f]).convert('RGB')) / 255.
        try:
            # mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))
            mask_files = glob.glob(os.path.join(self.mask_dir, video, '*.png'))
            mask_files.sort()
            # print(mask_files[f])
            N_masks[0] = np.array(Image.open(mask_files[f]).convert('P'), dtype=np.uint8)
        except:
            N_masks[0] = 255
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        # Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
        return Fs, torch.from_numpy(N_masks), img_files


if __name__ == '__main__':
    pass

