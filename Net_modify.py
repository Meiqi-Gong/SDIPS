import os
import sys
import torch.nn as nn
import torch.optim as optim
from base_modify import *
from torchvision.transforms import *
import numpy as np
import torch.nn.functional as F

class MS_encoder(nn.Module):
    def __init__(self, num_channels=4, n_feat=28, reduction=8, bias=False, norm=None):
        super(MS_encoder, self).__init__()
        act = nn.PReLU()
        self.up = nn.Upsample(scale_factor=4, mode='bicubic')
        self.body1_1 = nn.Sequential(Depthwise_ConvBlock(num_channels, n_feat, 3, 1, padding=1, activation='prelu', norm=norm),
                                     Depthwise_ConvBlock(n_feat, n_feat, 3, 1, padding=1, activation='prelu', norm=norm),
                                     )
        self.body1_2 = nn.Sequential(Depthwise_ConvBlock(n_feat*7, n_feat * 4, 3, 1, padding=1, activation='prelu',norm=norm),
                                     Depthwise_ConvBlock(n_feat * 4, n_feat * 2, 3, 1, padding=1, activation='prelu',norm=norm))
        self.body1_3 = nn.Sequential(Depthwise_ConvBlock(n_feat * 2, num_channels*2, 3, 1, padding=1, activation='prelu', norm=norm),
                                     Depthwise_ConvBlock(num_channels*2, n_feat, 3, 1, padding=1, activation='prelu', norm=norm))
        self.encoder = Encoder(n_feat, 3, bias, act, n_feat, norm)
        self.decoder = Decoder(n_feat, 3, bias, act, n_feat, norm)
        self.dilated1 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=1, dilation=1), act)
        self.dilated2 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=2, dilation=2), act)
        self.dilated3 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=4, dilation=4), act)
    def forward(self, ms):
        feat_ms1 = self.body1_1(self.up(ms))
        feat_1 = self.encoder(feat_ms1)
        feat_2 = self.decoder(feat_1[0], feat_1[1], feat_1[2])
        feat1 = torch.cat((feat_2[0], self.dilated1(feat_2[0]), self.dilated2(feat_2[0]),
                           self.dilated3(feat_2[0])), 1)
        feat_ms = self.body1_2(feat1)
        feat_ms = self.body1_3(feat_ms)+feat_ms1
        return feat_ms

class PAN_encoder(nn.Module):
    def __init__(self, n_feat=28, kernel_size=3, scale_feat=28, reduction=8, bias=False, norm=None):
        super(PAN_encoder, self).__init__()
        act = nn.PReLU()
        self.body2_1 = nn.Sequential(Depthwise_ConvBlock(1, n_feat, 3, 1, padding=1, activation='prelu', norm=norm),
                                     Depthwise_ConvBlock(n_feat, n_feat, 3, 1, padding=1, activation='prelu', norm=norm),
                                     )
        self.body2_2 = nn.Sequential(Depthwise_ConvBlock(n_feat*7, n_feat * 4, 3, 1, padding=1, activation='prelu',norm=norm),
                                     Depthwise_ConvBlock(n_feat * 4, n_feat * 2, 3, 1, padding=1, activation='prelu',norm=norm))
        self.body2_3 = nn.Sequential(Depthwise_ConvBlock(n_feat * 2, n_feat, 3, 1, padding=1, activation='prelu', norm=norm),
                                     Depthwise_ConvBlock(n_feat, n_feat, 3, 1, padding=1, activation='prelu', norm=norm))
        self.encoder = Encoder(n_feat, kernel_size, bias, act, scale_feat, norm)
        self.decoder = Decoder(n_feat, kernel_size, bias, act, scale_feat, norm)
        self.dilated1 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=1, dilation=1), act)
        self.dilated2 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=2, dilation=2), act)
        self.dilated3 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=4, dilation=4), act)

    def forward(self, pan):
        feat_pan1 = self.body2_1(pan)

        feat_1 = self.encoder(feat_pan1)
        feat_2 = self.decoder(feat_1[0], feat_1[1], feat_1[2])
        feat1 = torch.cat((feat_2[0], self.dilated1(feat_2[0]), self.dilated2(feat_2[0]),
                           self.dilated3(feat_2[0])), 1)
        feat_pan = self.body2_2(feat1)
        feat_pan = self.body2_3(feat_pan)+feat_pan1

        return feat_pan

class PS_decoder(nn.Module):
    def __init__(self, num_channels=4, n_feat=28,
                 scale_feat=28, kernel_size=3, reduction=8, bias=False):
        super(PS_decoder, self).__init__()
        act = nn.PReLU()
        self.up4 = nn.Upsample(scale_factor=4, mode='bicubic')
        self.shallow_feat2 = nn.Sequential(conv(n_feat*2, n_feat, kernel_size, bias=bias), act)
        self.encoder2 = Encoder(n_feat, kernel_size, bias, act, scale_feat, norm=None)
        self.decoder2 = Decoder(n_feat, kernel_size, bias, act, scale_feat, norm=None)
        self.dilated1 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=1, dilation=1), act)
        self.dilated2 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=2, dilation=2), act)
        self.dilated3 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=4, dilation=4), act)
        self.body = nn.Sequential(Depthwise_ConvBlock(n_feat * 7, n_feat*4, 3, 1, padding=1, activation='prelu', norm=None),
                                     Depthwise_ConvBlock(n_feat*4, n_feat, 3, 1, padding=1, activation='prelu', norm=None))
        self.conv1_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv1_2 = nn.Sequential(conv(n_feat, n_feat*2, kernel_size, bias=bias), act)
        self.conv1_3 = nn.Sequential(conv(n_feat*2, n_feat*3, kernel_size, bias=bias), act)

        self.conv2_1 = nn.Sequential(conv(n_feat * 3, n_feat * 2, kernel_size, bias=bias), act)
        self.conv2_2 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias), act)
        self.conv2_3 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())
        self.conv_tail3_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv_tail3_2 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())


    def forward(self, feat_ms, feat_pan):
        xfeat = self.shallow_feat2(torch.cat((feat_ms, feat_pan), 1))
        _,_,h,w=xfeat.shape
        feat2 = self.encoder2(xfeat)
        resx = self.decoder2(feat2[0], feat2[1], feat2[2])
        feat1 = torch.cat((resx[0], self.dilated1(resx[0]), self.dilated2(resx[0]),
                           self.dilated3(resx[0])), 1)
        feat = self.body(feat1)
        x1 = self.conv1_1(feat) + xfeat
        x2 = self.conv1_2(x1)
        x3 = self.conv1_3(x2)

        x4 = self.conv2_1(x3) + x2
        x5 = self.conv2_2(x4) + x1
        x6 = self.conv2_3(x5)
        img3 = x6 / 2 + 0.5

        return img3
#
class MS_decoder(nn.Module):
    def __init__(self, num_channels=4, n_feat=28,
                 scale_feat=28, kernel_size=3, reduction=8, bias=False):
        super(MS_decoder, self).__init__()
        act = nn.PReLU()
        self.down = nn.Upsample(scale_factor=0.25, mode='bicubic')
        self.shallow_feat1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.encoder2 = Encoder(n_feat, kernel_size, bias, act, scale_feat, norm=None)
        self.decoder2 = Decoder(n_feat, kernel_size, bias, act, scale_feat, norm=None)
        self.dilated1 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=1, dilation=1), act)
        self.dilated2 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=2, dilation=2), act)
        self.dilated3 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=4, dilation=4), act)
        self.body = nn.Sequential(Depthwise_ConvBlock(n_feat * 7, n_feat*4, 3, 1, padding=1, activation='prelu', norm=None),
                                     Depthwise_ConvBlock(n_feat*4, n_feat, 3, 1, padding=1, activation='prelu', norm=None))
        self.conv1_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv1_2 = nn.Sequential(conv(n_feat, n_feat*2, kernel_size, bias=bias), act)
        self.conv1_3 = nn.Sequential(conv(n_feat*2, n_feat*3, kernel_size, bias=bias), act)

        self.conv2_1 = nn.Sequential(conv(n_feat * 3, n_feat * 2, kernel_size, bias=bias), act)
        self.conv2_2 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias), act)
        self.conv2_3 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())
    def forward(self, feat_ms):
        xfeat = self.shallow_feat1(self.down(feat_ms))
        _, _, h, w = xfeat.shape
        feat2 = self.encoder2(xfeat)
        resx = self.decoder2(feat2[0], feat2[1], feat2[2])
        feat1 = torch.cat((resx[0], self.dilated1(resx[0]), self.dilated2(resx[0]),
                           self.dilated3(resx[0])), 1)
        feat = self.body(feat1)
        x1 = self.conv1_1(feat) + xfeat
        x2 = self.conv1_2(x1)
        x3 = self.conv1_3(x2)

        x4 = self.conv2_1(x3) + x2
        x5 = self.conv2_2(x4) + x1
        x6 = self.conv2_3(x5)
        img3 = x6 / 2 + 0.5
        return img3

class PAN_decoder(nn.Module):
    def __init__(self, num_channels=1, n_feat=28,
                 scale_feat=28, kernel_size=3, reduction=8, bias=False):
        super(PAN_decoder, self).__init__()
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)

        self.encoder2 = Encoder(n_feat, kernel_size, bias, act, scale_feat, norm=None)
        self.decoder2 = Decoder(n_feat, kernel_size, bias, act, scale_feat, norm=None)
        self.dilated1 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=1, dilation=1), act)
        self.dilated2 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=2, dilation=2), act)
        self.dilated3 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, padding=4, dilation=4), act)
        self.body = nn.Sequential(Depthwise_ConvBlock(n_feat * 7, n_feat*4, 3, 1, padding=1, activation='prelu', norm=None),
                                     Depthwise_ConvBlock(n_feat*4, n_feat, 3, 1, padding=1, activation='prelu', norm=None))
        self.conv_tail3_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv_tail3_2 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())
        self.conv1_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv1_2 = nn.Sequential(conv(n_feat, n_feat*2, kernel_size, bias=bias), act)
        self.conv1_3 = nn.Sequential(conv(n_feat*2, n_feat*3, kernel_size, bias=bias), act)

        self.conv2_1 = nn.Sequential(conv(n_feat * 3, n_feat * 2, kernel_size, bias=bias), act)
        self.conv2_2 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias), act)
        self.conv2_3 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())

    def forward(self, feat_pan):
        # print(feat_pan.shape)
        xfeat = self.shallow_feat1(feat_pan)
        _, _, h, w = xfeat.shape
        feat2 = self.encoder2(xfeat)
        resx = self.decoder2(feat2[0], feat2[1], feat2[2])
        feat1 = torch.cat((resx[0], self.dilated1(resx[0]), self.dilated2(resx[0]),
                           self.dilated3(resx[0])), 1)
        feat = self.body(feat1)
        x1 = self.conv1_1(feat) + xfeat
        x2 = self.conv1_2(x1)
        x3 = self.conv1_3(x2)

        x4 = self.conv2_1(x3) + x2
        x5 = self.conv2_2(x4) + x1
        x6 = self.conv2_3(x5)
        img3 = x6 / 2 + 0.5
        return img3


class DS(nn.Module):
    def __init__(self, n_feats=56):
        super(DS, self).__init__()
        self.down1 = Depthwise_ConvBlock(n_feats, n_feats, 4, 2, padding=1, activation='prelu', norm=None)
        self.down2 = Depthwise_ConvBlock(n_feats, n_feats, 4, 2, padding=1, activation='prelu', norm=None)
        act = nn.PReLU()
        self.conv1 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1), act)
        self.conv2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1), act)

    def forward(self, x):
        x = self.down1(x)
        x = self.conv1(x)
        x = self.down2(x)
        x = self.conv2(x)
        return x