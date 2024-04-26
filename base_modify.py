import sys

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'prelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out
class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=False, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Depthwise_ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, bias=True, activation='prelu',
                 norm=None):
        super(Depthwise_ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'prelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Depthwise_DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(Depthwise_DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=True):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y = self.conv_du(y1+y2)
        return x * y

class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu'):
        super(UpBlock, self).__init__()
        self.up_conv1 = Depthwise_DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv3 = Depthwise_DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, base, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu'):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter-base, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter-base, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter-base, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0



class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, base, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu'):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter+base, kernel_size, stride, padding, activation, norm='batch')
        self.down_conv2 = DeconvBlock(num_filter+base, num_filter, kernel_size, stride, padding, activation, norm='batch')
        self.down_conv3 = ConvBlock(num_filter, num_filter+base, kernel_size, stride, padding, activation, norm='batch')

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class enc(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, act):
        super(enc, self).__init__()
        modules_body = []
        modules_body.append(ConvBlock(n_feat, n_feat//2, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                         bias=bias, activation='prelu', norm='batch'))
        # modules_body.append(act)
        modules_body.append(ConvBlock(n_feat//2, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                         bias=bias, activation='prelu', norm='batch'))
        # modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, act, scale_feat, norm):
        super(Encoder, self).__init__()
        self.encoder_level1 = [ConvBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                         bias=bias, activation='prelu', norm=norm),
                               enc(n_feat, kernel_size, bias, act)]
        self.encoder_level2 = [enc(n_feat+scale_feat, kernel_size, bias, act)]
        self.encoder_level3 = [enc(n_feat+scale_feat*2, kernel_size, bias, act)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = nn.Sequential(ConvBlock(n_feat, n_feat, 2, 2, padding=0, activation='prelu', norm=norm),
                                    ConvBlock(n_feat, n_feat+scale_feat, 1, 1, padding=0, activation='prelu', norm=norm))
        self.down23 = nn.Sequential(ConvBlock(n_feat+scale_feat, n_feat+scale_feat, 2, 2, padding=0, activation='prelu', norm=norm),
                                    ConvBlock(n_feat+scale_feat, n_feat+2*scale_feat, 1, 1, padding=0, activation='prelu', norm=norm))


    def forward(self, x):#, f1, f4):
            enc1 = self.encoder_level1(x)# + self.trans_enc1(f4))
            x = self.down12(enc1)

            enc2 = self.encoder_level2(x)
            x = self.down23(enc2)

            enc3 = self.encoder_level3(x)# + self.trans_enc3(f1))
            # print(enc3.shape)
            return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, act, scale_feat, norm):
        super(Decoder, self).__init__()
        self.decoder_level1 = [enc(n_feat, kernel_size, bias, act)]
        self.decoder_level2 = [enc(n_feat+scale_feat, kernel_size, bias, act)]
        self.decoder_level3 = [enc(n_feat+scale_feat*2, kernel_size, bias, act)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = enc(n_feat, kernel_size, bias, act)
        self.skip_attn2 = enc(n_feat+scale_feat, kernel_size, bias, act)

        self.up32 = nn.Sequential(Depthwise_DeconvBlock(n_feat+scale_feat*2, n_feat+scale_feat*2, 2, 2, padding=0, activation='prelu', norm=norm),
                                    ConvBlock(n_feat+scale_feat*2, n_feat+scale_feat, 1, 1, padding=0, activation='prelu', norm=norm))
        self.up21 = nn.Sequential(Depthwise_DeconvBlock(n_feat+scale_feat, n_feat+scale_feat, 2, 2, padding=0, activation='prelu', norm=norm),
                                    ConvBlock(n_feat+scale_feat, n_feat, 1, 1, padding=0, activation='prelu', norm=norm))

    def forward(self, enc1, enc2, enc3):
        # enc1, enc2, enc3 = feat
        dec3 = self.decoder_level3(enc3)

        # print(self.skip_attn2(enc2).shape)
        x = self.up32(dec3)
        # print(self.skip_attn2(enc2).shape, self.decoder_level2(x).shape)
        dec2 = self.decoder_level2(x) + self.skip_attn2(enc2)
        # print(dec2.shape)
        # sys.exit(0)

        x = self.up21(dec2)
        dec1 = self.decoder_level1(x) + enc1# + self.skip_attn1(enc1)

        return [dec1, dec2, dec3]
