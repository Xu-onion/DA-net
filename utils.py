import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torch.optim import lr_scheduler

def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    a = image_numpy.squeeze(2)
    cv2.imwrite(filename[0:-3] + 'tiff', a)

    print("Image saved as {}".format(filename))


def sample_images(input, target, prediction, filename):
    input_0 = input.cpu().numpy()
    target_0 = target.cpu().numpy()
    prediction_0 = prediction.detach().cpu().numpy()

    input = np.transpose(input_0, (0, 2, 3, 1))
    input = (input - np.min(input)) / (np.max(input) - np.min(input))
    input = np.asarray(input * 255, dtype=np.uint8)

    target = np.transpose(target_0, (0, 2, 3, 1))
    target = (target - np.min(target)) / (np.max(target) - np.min(target))
    target = np.asarray(target * 255, dtype=np.uint8)

    prediction = np.transpose(prediction_0, (0, 2, 3, 1))
    prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
    prediction = np.asarray(prediction * 255, dtype=np.uint8)

    gen_imgs = [input, target, prediction]
    titles = ['Original', 'Condition', 'Generated']
    r, c = 4, 3
    fig, axs = plt.subplots(r, c, figsize=(8, 9))
    # r, c = 7, 3
    # fig, axs = plt.subplots(r, c, figsize=(10, 16))
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[j][i], cmap='hot')
            axs[i, j].set_title(titles[j])
            axs[i, j].axis('off')
    fig.savefig(filename)
    # plt.show()
    plt.close()

def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y

def diffxy(img, order=3):
    for _ in range(order):
        img = prctile_norm(img)
        d = np.zeros_like(img)
        dx = (img[1:-1, 0:-2] + img[1:-1, 2:]) / 2
        dy = (img[0:-2, 1:-1] + img[2:, 1:-1]) / 2
        d[1:-1, 1:-1] = img[1:-1, 1:-1] - (dx + dy) / 2
        d[d < 0] = 0
        img = d
    return img


def rm_outliers(img, order=3, thresh=0.2):
    img_diff = diffxy(img, order)
    mask = img_diff > thresh
    img_rm_outliers = img
    img_mean = np.zeros_like(img)
    for i in [-1, 1]:
        for a in range(0, 2):
            img_mean = img_mean + np.roll(img, i, axis=a)
    img_mean = img_mean / 4
    img_rm_outliers[mask] = img_mean[mask]
    img_rm_outliers = prctile_norm(img_rm_outliers)
    return img_rm_outliers

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.nEpochs) / float((opt.nEpochs/2) + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nEpochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.9f' % lr)