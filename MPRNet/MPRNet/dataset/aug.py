import numpy as np
import imgaug.augmenters as iaa
import random
import cv2
from PIL import Image


class Flip(object):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, images, labels):
        if random.random() < self.rate:
            for i in range(4):
                # Random flipping
                images[i] = np.fliplr(images[i]).copy()  # HWC
                labels[i] = np.fliplr(labels[i]).copy()  # HW
        return images, labels

# class Flip1(object):
#     def __init__(self, rate):
#         self.rate = rate
#
#     def __call__(self, images, labels):
#         newimages = list(reversed(images))
#         newlabels = list(reversed(labels))
#         del images
#         del labels
#         return newimages, newlabels


class RandomSizedCrop(object):
    def __init__(self, scale, crop_size):
        self.scale = scale
        self.crop_size = crop_size

    def __call__(self, images, labels):

        scale_factor = random.uniform(self.scale[0], self.scale[1])
        for i in range(4):
            h, w = labels[i].shape
            h, w = (max(384, int(h * scale_factor)), max(384, int(w * scale_factor)))
            images[i] = (cv2.resize(images[i], (w, h), interpolation=cv2.INTER_LINEAR))
            labels[i] = Image.fromarray(labels[i]).resize((w, h), resample=Image.NEAREST)
            labels[i] = np.asarray(labels[i], dtype=np.int8)
        ob_loc = ((labels[0] + labels[1] + labels[2] + labels[3]) > 0).astype(np.uint8)
        box = cv2.boundingRect(ob_loc)

        x_min = box[0]
        x_max = box[0] + box[2]
        y_min = box[1]
        y_max = box[1] + box[3]

        if x_max - x_min > 384:
            start_w = random.randint(x_min, x_max - 384)
        elif x_max - x_min == 384:
            start_w = x_min
        else:
            start_w = random.randint(max(0, x_max - 384), min(x_min, w - 384))

        if y_max - y_min > 384:
            start_h = random.randint(y_min, y_max - 384)
        elif y_max - y_min == 384:
            start_h = y_min
        else:
            start_h = random.randint(max(0, y_max - 384), min(y_min, h - 384))
        # Cropping

        start_h = random.randint(start_h - 20, start_h + 20)
        start_h = max(0, start_h)
        start_h = min(h - 384, start_h)
        start_w = random.randint(start_w - 20, start_w + 20)
        start_w = max(0, start_w)
        start_w = min(w - 384, start_w)
        end_h = start_h + 384
        end_w = start_w + 384

        for i in range(4):
            images[i] = images[i][start_h:end_h, start_w:end_w] / 255.
            labels[i] = labels[i][start_h:end_h, start_w:end_w]
        return images, labels


class aug_heavy(object):
    def __init__(self):
        self.crop = RandomSizedCrop([0.80, 1.1], 384)
        self.flip = Flip(0.5)
        # self.flip1 = Flip1(0.5)

    def __call__(self, images, labels):
        images, labels = self.flip(images, labels)
        # images, labels = self.flip1(images, labels)
        r = random.randint(-30, 30)
        self.rotate = iaa.Sometimes(
            1,
            iaa.Affine(rotate=r)
        )
        s = random.randint(-15, 15)
        self.shear = iaa.Sometimes(
            1,
            iaa.Affine(shear=s)
        )
        x1 = random.randint(-15, 15)
        y1 = random.randint(-15, 15)
        self.trans = iaa.Sometimes(
            1,
            iaa.Affine(translate_px={"x": x1, "y": y1})
        )
        x2 = random.uniform(0.8, 1.2)
        y2 = random.uniform(0.8, 1.2)
        self.scale = iaa.Sometimes(
            1,
            iaa.Affine(scale={"x": x2, "y": y2})
        )

        for i in range(4):
            # print("hhhhhhhhhhhhhh")
            # print(images[i].shape)
            # print(labels[i].shape)
            if random.random() < 0.5:
                images[i], labels[i] = self.rotate(image=images[i],
                                                   segmentation_maps=labels[i][np.newaxis, :, :, np.newaxis])
                labels[i] = labels[i][0, :, :, 0]
            if random.random() < 0.5:
                images[i], labels[i] = self.shear(image=images[i],
                                                  segmentation_maps=labels[i][np.newaxis, :, :, np.newaxis])
                labels[i] = labels[i][0, :, :, 0]
            if random.random() < 0.5:
                images[i], labels[i] = self.trans(image=images[i],
                                                  segmentation_maps=labels[i][np.newaxis, :, :, np.newaxis])
                labels[i] = labels[i][0, :, :, 0]
            if random.random() < 0.5:
                images[i], labels[i] = self.scale(image=images[i],
                                                  segmentation_maps=labels[i][np.newaxis, :, :, np.newaxis])
                labels[i] = labels[i][0, :, :, 0]
        images, labels = self.crop(images, labels)
        return images, labels
