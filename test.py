from __future__ import print_function
import argparse
import os
from os import listdir
from os.path import join
import numpy as np
from PIL import Image
import cv2
import mahotas as mh
from skimage import filters
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchsummary import summary
import datetime

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])

def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-8)
    y[y > 1] = 1
    y[y < 0] = 0
    return y

class DatasetFromFolder_test(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder_test, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames_a = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.image_filenames_b = [x for x in listdir(self.b_path) if is_image_file(x)]
        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a_0 = Image.open(join(self.a_path, self.image_filenames_a[index]))
        a_0 = np.asarray(a_0, dtype=np.float32)

        b_0 = Image.open(join(self.b_path, self.image_filenames_b[index]))
        b_0 = np.asarray(b_0, dtype=np.float32)

        a_0 = prctile_norm(a_0)
        b_0 = prctile_norm(b_0)

        Y, X = a_0, b_0
        a = self.transform(Y.copy())
        b = self.transform(X.copy())
        return b, a, self.image_filenames_b[index]

    def __len__(self):
        return len(self.image_filenames_b)

def sample_images(input, target, prediction, filename):
    input_0 = input.cpu().numpy()
    target_0 = target.cpu().numpy()
    prediction_0 = prediction.detach().cpu().numpy()

    input = np.transpose(input_0, (0, 2, 3, 1))
    input = (input - np.min(input)) / (np.max(input) - np.min(input))
    input = np.asarray(input[0, :, :, 0] * 255, dtype=np.uint8)

    target = np.transpose(target_0, (0, 2, 3, 1))
    target = (target - np.min(target)) / (np.max(target) - np.min(target))
    target = np.asarray(target[0, :, :, 0] * 255, dtype=np.uint8)

    prediction = np.transpose(prediction_0, (0, 2, 3, 1))
    prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
    prediction = np.asarray(prediction[0, :, :, 0] * 255, dtype=np.uint8)

    plt.subplot(1, 3, 1)
    plt.imshow(input, cmap='hot')
    plt.axis('off')
    plt.title('Original')

    plt.subplot(1, 3, 2)
    plt.imshow(target, cmap='hot')
    plt.axis('off')
    plt.title('Condition')

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='hot')
    plt.axis('off')
    plt.title('Generated')

    plt.savefig(filename)
    # plt.show()
    plt.close()


# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', type=str, default="Vasculature")
parser.add_argument('--cuda', type=str, default=True, help='use cuda?')
parser.add_argument('--test_result', type=str, default="test_result")

opt = parser.parse_args()
print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/DA-net/danet.pth"
net_g = torch.load(model_path).to(device)
 
# summary(net_g, (1, 512, 512))

root_path = "dataset/"
test_dir = join(root_path + opt.dataset, "")
test_set = DatasetFromFolder_test(test_dir)

testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

start_time = datetime.datetime.now()
with torch.no_grad():
    for iteration, batch in enumerate(testing_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)
        file_name = batch[2][0]
        file_name = file_name[0:-4]
        prediction = net_g(input)

        # Save comparison
        if not os.path.exists(os.path.join("result", opt.dataset)):
            os.makedirs(os.path.join("result", opt.dataset))
        sample_images(input, target, prediction, "result/{}/{}.png".format(opt.dataset, file_name))

        # # Save single image
        pred = prediction.detach().cpu().numpy()
        pred = np.transpose(pred, (0, 2, 3, 1))
        # pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
        # pred = np.asarray(pred * 255, dtype=np.uint8)

        pred = np.maximum(pred, 0)
        pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-8)

        pixel_values = pred.flatten()
        sorted_pixel_values = np.sort(pixel_values)
        threshold_index = int(0.01 * len(sorted_pixel_values))
        threshold_value = sorted_pixel_values[threshold_index]
        pred = pred - threshold_value
        pred = np.maximum(pred, 0)  # prediction[prediction < 0] = 0

        prediction = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-8)
        prediction = np.asarray(prediction[0, :, :, 0] * 255, dtype=np.uint8)

        if not os.path.exists('result/{}/pred/'.format(opt.dataset)):
            os.makedirs('result/{}/pred/'.format(opt.dataset))
        cv2.imwrite("result/{}/pred/{}.tiff".format(opt.dataset, file_name), prediction)
elapsed_time = datetime.datetime.now() - start_time
print(elapsed_time)

