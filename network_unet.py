import math
import torch
import torch.nn as nn
from torch.nn import init
from torchsummary import summary
import networks.basicblock as B
import numpy as np

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, gain=init_gain)
    return net


# --------------- UnetRCAB_ViT ---------------
class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)

        ''' use DropKey as regularizer '''
        m_r = torch.ones_like(attn) * 0.2
        attn = attn + torch.bernoulli(m_r) * -1e-12

        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class UnetRCAB_ViT(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512, 512], nb=4, act_mode='L', downsample_mode='strideconv',
                 upsample_mode='convtranspose', bias=True):
        super(UnetRCAB_ViT, self).__init__()

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        # self.m_head = B.conv(in_nc, nc[0], kernel_size=3, stride=2, padding=1, mode='C' + 'B' + act_mode)
        self.m_head = downsample_block(in_nc, nc[0], bias=bias, mode='2')

        self.m_down1 = B.sequential(
            *[B.RCABlock(nc[0], nc[0], bias=bias, mode='C' + 'B' + act_mode) for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=bias, mode='2'))
        self.m_down2 = B.sequential(
            *[B.RCABlock(nc[1], nc[1], bias=bias, mode='C' + 'B' + act_mode) for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=bias, mode='2'))
        self.m_down3 = B.sequential(
            *[B.RCABlock(nc[2], nc[2], bias=bias, mode='C' + 'B' + act_mode) for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=bias, mode='2'))
        self.m_down4 = B.sequential(
            *[B.RCABlock(nc[3], nc[3], bias=bias, mode='C' + 'B' + act_mode) for _ in range(nb)],
            downsample_block(nc[3], nc[4], bias=bias, mode='2'))

        self.m_body = B.sequential(*[B.RCABlock(nc[4], nc[4], bias=bias, mode='C' + 'B' + act_mode) for _ in range(nb)])
        self.attn = SelfAttention(nc[3], n_head=8, norm_groups=32)

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up4 = B.sequential(upsample_block(nc[4], nc[3], bias=bias, mode='2'),
                                  *[B.RCABlock(nc[3], nc[3], bias=bias, mode='C' + 'B' + act_mode) for _ in range(nb)])
        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'),
                                  *[B.RCABlock(nc[2], nc[2], bias=bias, mode='C' + 'B' + act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'),
                                  *[B.RCABlock(nc[1], nc[1], bias=bias, mode='C' + 'B' + act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'),
                                  *[B.RCABlock(nc[0], nc[0], bias=bias, mode='C' + 'B' + act_mode) for _ in range(nb)])

        # self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')
        self.m_tail = upsample_block(nc[0], out_nc, bias=bias, mode='2')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)

        x = self.m_body(x4)
        x = self.attn(x)

        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = nn.Tanh()(x)
        return x


if __name__ == '__main__':
    x = torch.rand(1, 1, 512, 512).to('cuda:0')
    net = UnetRCAB_ViT(nb=2)
    net = init_net(net, init_type='normal', init_gain=0.02)
    net = net.to('cuda:0')

    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())
    print('Parameters number is ', sum(param.numel() for param in net.parameters()))

    # summary(net, (1, 512, 512))         # 模型框架

