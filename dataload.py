from os import listdir
from os.path import join
import numpy as np
import random
import cv2
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
matplotlib.use('TkAgg')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])

def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-8)
    y[y > 1] = 1
    y[y < 0] = 0
    return y

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:           # 旋转 90°
        return np.rot90(img)
    elif mode == 1:         # 旋转 180°
        return np.rot90(img, k=2)
    elif mode == 2:         # 旋转 270°
        return np.rot90(img, k=3)
    elif mode == 3:         # 垂直翻转
        return np.flipud(img)
    elif mode == 4:         # 水平翻转
        return np.fliplr(img)
    elif mode >= 5:
        return img


class DatasetFromFolder_train(data.Dataset):
    def __init__(self, image_dir, patch_size):
        super(DatasetFromFolder_train, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.patch_size = patch_size
        transform_list = [transforms.ToTensor()]
                          # transforms.Normalize(0.5, 0.5)]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a_0 = Image.open(join(self.a_path, self.image_filenames[index]))
        a_0 = np.asarray(a_0, dtype=np.float32)

        b_0 = Image.open(join(self.b_path, self.image_filenames[index]))
        b_0 = np.asarray(b_0, dtype=np.float32)

        # # Data preprocess with uint8
        # a_0 = a_0 / 255
        # b_0 = b_0 / 255

        # # Data preprocess with uint16
        # a_0[a_0 < 20] = 0           # GT 中存在伪影，通过调整阈值去除
        # a_0 = a_0 / 65535
        # b_0 = (b_0 + 20) / 65535    # Input 中存在负值，调整到 0 以上

        a_0 = prctile_norm(a_0)
        b_0 = prctile_norm(b_0)

        # --------------------------------
        # randomly crop the patch
        # --------------------------------
        # w = np.random.randint(0, b_0.shape[0]-self.patch_size)
        # h = np.random.randint(0, b_0.shape[1]-self.patch_size)
        # Y = a_0[w:w+self.patch_size, h:h+self.patch_size]
        # X = b_0[w:w + self.patch_size, h:h + self.patch_size]

        # # --------------------------------
        # # augmentation - flip and/or rotate
        # # --------------------------------
        # mode = random.randint(0, 10)
        # X, Y = augment_img(X, mode=mode), augment_img(Y, mode=mode)

        # 将像素值转换为PyTorch张量
        Y, X = a_0, b_0
        a = self.transform(Y.copy())
        b = self.transform(X.copy())

        # # 加入 DPM 通道，增强不同深度下的区分度
        # DPM = torch.ones_like(a)
        # frame_name = self.image_filenames[index]
        # depth = float(frame_name[-7:-4])
        # DPM = DPM * depth * 10
        #
        # a = torch.cat((a, DPM), dim=0)
        # b = torch.cat((b, DPM), dim=0)

        return b, a

    def __len__(self):
        return len(self.image_filenames)



class DatasetFromFolder_test(data.Dataset):
    def __init__(self, image_dir, patch_size):
        super(DatasetFromFolder_test, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.patch_size = patch_size
        transform_list = [transforms.ToTensor()]
                          # transforms.Normalize(0.5, 0.5)]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a_0 = Image.open(join(self.a_path, self.image_filenames[index]))
        a_0 = np.asarray(a_0, dtype=np.float32)

        b_0 = Image.open(join(self.b_path, self.image_filenames[index]))
        b_0 = np.asarray(b_0, dtype=np.float32)

        # # Data preprocess with uint8
        # a_0 = a_0 / 255
        # b_0 = b_0 / 255

        # # Data preprocess with uint16
        # a_0[a_0 < 20] = 0           # GT 中存在伪影，通过调整阈值去除
        # a_0 = a_0 / 65535
        # b_0 = (b_0 + 20) / 65535    # Input 中存在负值，调整到 0 以上

        a_0 = prctile_norm(a_0)
        b_0 = prctile_norm(b_0)

        # # --------------------------------
        # # randomly crop the patch
        # # --------------------------------
        # w = np.random.randint(0, b_0.shape[0]-self.patch_size)
        # h = np.random.randint(0, b_0.shape[1]-self.patch_size)
        # X = b_0[w:w+self.patch_size, h:h+self.patch_size]
        # Y = a_0[w:w+self.patch_size, h:h+self.patch_size]

        # 将像素值转换为PyTorch张量
        Y, X = a_0, b_0
        a = self.transform(Y.copy())
        b = self.transform(X.copy())

        # # 加入 DPM 通道，增强不同深度下的区分度
        # DPM = torch.ones_like(a)
        # frame_name = self.image_filenames[index]
        # depth = float(frame_name[-7:-4])
        # DPM = DPM * depth * 10
        #
        # a = torch.cat((a, DPM), dim=0)
        # b = torch.cat((b, DPM), dim=0)

        return b, a, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)

# Show pictures
# plt.figure(figsize=(8, 8))
# plt.subplot(121)
# plt.title('fliped image')
# plt.imshow(a, cmap='gray')
# plt.subplot(122)
# plt.title('shift image')
# plt.imshow(b, cmap='gray')
# plt.show(block=True)


