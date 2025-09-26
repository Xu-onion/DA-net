from __future__ import print_function
import argparse
import os
import numpy as np
import pandas as pd
from math import log10
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import pytorch_ssim
import mahotas as mh
from skimage import filters
from network_unet import UnetRCAB_ViT, init_net
from dataload import DatasetFromFolder_train, DatasetFromFolder_test
from utils import sample_images, get_scheduler, update_learning_rate
import warnings
warnings.filterwarnings('ignore')


# Training settings
def parse_opts():
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    # 加载恢复：继续断点训练/迁移学习； 如要迁移模型请设置为 True，从头训练则设置为 False
    parser.add_argument('--RESUME', type=str, default=False, help='use transfer weight?')
    parser.add_argument('--dataset', type=str, default="Scatter_Vas")
    parser.add_argument('--train_out', type=str, default="train_out")
    parser.add_argument('--patch_size', type=int, default=512, help='input image size')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
    parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
    parser.add_argument('--nEpochs', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--generatorLR', type=float, default=1e-4, help='initial learning rate for generator')
    parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--cuda', type=str, default=True, help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    args = parser.parse_args()
    return args

# Li loss
def segment_Li(input):
    input = (input - np.min(input)) / (np.max(input) - np.min(input) + 1e-8)
    input = np.asarray(input * 255, dtype=np.uint8)
    bw_input = (input > filters.threshold_li(input))
    return bw_input

def dice_coeff(pred, target):
    # Segment prediction and target
    pred = pred.detach().cpu().numpy().squeeze(1)
    target = target.detach().cpu().numpy().squeeze(1)

    bw_pred = np.zeros_like(pred, dtype=bool)
    bw_target = np.zeros_like(target, dtype=bool)
    # Segmentation
    for channel in range(pred.shape[0]):
        bw_pred[channel, :, :] = segment_Li(pred[channel, :, :])
        bw_target[channel, :, :] = segment_Li(target[channel, :, :])
    bw_pred = torch.tensor(bw_pred)
    bw_target = torch.tensor(bw_target)

    # Cal Dice
    smooth = 1e-5
    num = bw_pred.size(0)
    m1 = bw_pred.view(num, -1)             # Flatten
    m2 = bw_target.view(num, -1)           # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

if __name__ == "__main__":
    opt = parse_opts()
    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    cudnn.benchmark = True
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    print('===> Loading datasets')
    root_dir = './dataset/'
    train_set = DatasetFromFolder_train(root_dir + opt.dataset + '/train', opt.patch_size)
    test_set = DatasetFromFolder_test(root_dir + opt.dataset + '/test', opt.patch_size)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)

    print('===> Building models')

# -------------------- Build Networks --------------------
    net_g = UnetRCAB_ViT(nb=2)
    net_g = init_net(net_g, init_type='normal', init_gain=0.02)
    net_g = net_g.to(device)

    # loss function
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    ssim_loss = pytorch_ssim.SSIM(window_size=11)

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.generatorLR, betas=(0.9, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)

    # History
    history = pd.DataFrame()
    G_loss = []
    VAL_mae = []
    VAL_mse = []
    VAL_dice = []
    VAL_ssim = []
    VAL_psnr = []
    start_time = datetime.datetime.now()

    opt.dataset = opt.dataset + '(UnetRCAB_ViT_Seg)'
    for epoch in range(opt.epoch_count, opt.nEpochs):
        # train------
        mean_mae_loss = 0.0
        mean_ssim_loss = 0.0
        mean_dice_loss = 0.0
        G_epoch_loss = 0.0
        count = len(training_data_loader)

        net_g.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a, real_b = batch[0].to(device), batch[1].to(device)

            fake_b = net_g(real_a)
            optimizer_g.zero_grad()

            # calculate loss
            loss_g_l1 = criterionL1(fake_b, real_b)
            mean_mae_loss += loss_g_l1

            # loss_g_l2 = criterionMSE(fake_b, real_b)
            loss_g_ssim = 1 - ssim_loss(fake_b, real_b)
            mean_ssim_loss += loss_g_ssim

            # Dice loss
            loss_g_dice = 1 - dice_coeff(fake_b, real_b)
            mean_dice_loss += loss_g_dice

            loss_g = (0.4 * loss_g_l1) + (0.1 * loss_g_ssim) + (0.5 * loss_g_dice)
            loss_g.backward()
            optimizer_g.step()

            G_epoch_loss += loss_g.item()

        update_learning_rate(net_g_scheduler, optimizer_g)


        # 求平均损失，生成图像以查看效果
        net_g.eval()
        with torch.no_grad():
            G_loss.append(G_epoch_loss / count)
            # 训练完一个Epoch,打印提示并绘制生成的图片
            elapsed_time = datetime.datetime.now() - start_time
            print("===> TRAIN: Epoch[{}]: Loss: {:.6f}  MAE: {:.6f}  Dice: {:.6f}  SSIM: {:.6f}  time: {}".
                  format(epoch+1, G_loss[-1], mean_mae_loss/count, 1-(mean_dice_loss/count), 1-(mean_ssim_loss/count), elapsed_time))

            # test
            all_mae = 0.0
            all_mse = 0.0
            all_dice = 0.0
            all_ssim = 0.0
            all_psnr = 0.0
            count = len(testing_data_loader)
            record = 5
            for iteration, batch in enumerate(testing_data_loader, 1):
                input, target = batch[0].to(device), batch[1].to(device)
                prediction = net_g(input)

                mae_value = criterionL1(prediction, target)
                mse_value = criterionMSE(prediction, target)
                dice_value = dice_coeff(prediction, target)
                ssim_value = ssim_loss(prediction, target)
                psnr_value = 20 * log10(1 / mse_value.item())

                all_mae += mae_value.item()
                all_mse += mse_value.item()
                all_dice += dice_value.item()
                all_ssim += ssim_value.item()
                all_psnr += psnr_value

                # If at save interval => save generated image samples
                if iteration + 1 == record:
                    trainout = opt.train_out + '('+opt.dataset+')'
                    if not os.path.exists(os.path.join("results", trainout)):
                        os.makedirs(os.path.join("results", trainout))
                    sample_images(input, target, prediction, "results/{}/{}.png".format(trainout, epoch+1))


            VAL_mae.append(all_mae / count)
            VAL_mse.append(all_mse / count)
            VAL_dice.append(all_dice / count)
            VAL_ssim.append(all_ssim / count)
            VAL_psnr.append(all_psnr / count)
            elapsed_time = datetime.datetime.now() - start_time
            print("===> VAL:   Epoch[{}]: MAE: {:.6f}  MSE: {:.6f}  DICE: {:.6f}  SSIM: {:.6f}  PSNR: {:.6f}dB  time: {}".
                  format(epoch+1, VAL_mae[-1], VAL_mse[-1], VAL_dice[-1], VAL_ssim[-1], VAL_psnr[-1], elapsed_time))

        # Save history
        epoch_result = dict(epoch=epoch+1, G_loss=round(float(G_loss[-1]), 6),
                            avg_mae=round(float(VAL_mae[-1]), 6), avg_mse=round(float(VAL_mse[-1]), 6), avg_dice=round(float(VAL_dice[-1]), 6),
                            avg_ssim=round(float(VAL_ssim[-1]), 6), avg_psnr=round(float(VAL_psnr[-1]), 6),
                            )
        history = history._append(epoch_result, ignore_index=True)
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
        history_path = "checkpoint/{}/history.csv".format(opt.dataset)
        history.to_csv(history_path, index=False)

        # Save model
        if (epoch+1) % 1 == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
                os.mkdir(os.path.join("checkpoint", opt.dataset))
            net_g_model_out_path = "checkpoint/{}/netG_epoch_{}.pth".format(opt.dataset, epoch+1)
            # net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
            torch.save(net_g, net_g_model_out_path)
            # torch.save(net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

            # Save loss
            x_ = range(opt.epoch_count, epoch + 1)
            plt.plot(x_, G_loss, label='Generator Losses')
            plt.legend()
            loss_image_path = "checkpoint/{}/loss.png".format(opt.dataset)
            plt.savefig(loss_image_path)
            plt.close()

    print('---------------Training is finished!!!---------------')















