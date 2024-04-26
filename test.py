from __future__ import print_function
import argparse
import sys

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
# import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set, get_fulltraining_set
from Net_modify import MS_encoder, PAN_encoder, PS_decoder, MS_decoder, PAN_decoder
from MMD import mmd_rbf
import socket
import time
import socket
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import xlwt
from scipy.misc import imresize
import scipy.io as scio

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--save_dir', default='results/QB/', help='Location to save checkpoint models')
parser.add_argument('--model_dir', default='weights/QB/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def load_mat(filepath):
    img = scio.loadmat(filepath)['i']
    return img

def ERGAS(hr_mul, label):
    """
    calc ergas.
    """
    h = 264
    l = 66
    channels = hr_mul.shape[2]

    inner_sum = 0
    for channel in range(channels):
        band_img1 = hr_mul[:, :, channel]
        band_img2 = label[:, :, channel]
        rmse_value = np.square(np.sqrt(np.mean(np.square(band_img1 - band_img2))) / np.mean(band_img1))
        inner_sum += rmse_value
    ergas = 100/(h/l)*np.sqrt(inner_sum/channels)

    return ergas

def save_img(img, name, epoch):
    save_dir = opt.save_dir+'reduced/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = save_dir + name[25:-4]
    print(name)
    scio.savemat(save_fn + '.mat', {'i': img})

def test(epoch):
    e1=0
    e2=0
    Encoder_ms.eval()
    Encoder_pan.eval()
    Decoder_ps.eval()
    t=[]
    with torch.no_grad():
        for iteration, batch in enumerate(testing_data_loader, 1):
            ms_label, pan_label, ms, pan, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), batch[4]
            if cuda:
                pan = pan.cuda(gpus_list[0])
                ms = ms.cuda(gpus_list[0])
                pan_label = pan_label.cuda(gpus_list[0])
                ms_label = ms_label.cuda(gpus_list[0])
            t0 = time.time()
            # print(name[0][25:-4])

            if int(name[0][25:-4])<=200 and int(name[0][25:-4])>=0:
                feat_ms = Encoder_ms(ms)
                feat_pan = Encoder_pan(pan)
                # feat_ms = Encoder_ms(ms_label)
                # feat_pan = Encoder_pan(pan_label)

                HRMS3 = Decoder_ps(feat_ms, feat_pan)
                result = HRMS3.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()

                t1 = time.time()
                t.append(t1-t0)
                print(name[0][25:-4], t1-t0)
                save_img(result, name[0], epoch)
                gt = ms_label.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
                erg = ERGAS(result, gt)
                e1+=erg
                if iteration % 20 == 0:
                    print("===> Epoch[{}]({}/{}):ERGAS: {:.4f}".format(epoch, iteration, len(testing_data_loader), erg))
    print("===> Eval Complete: Avg. ERGAS: {:.4f}".format(e1 / len(testing_data_loader)))

test_set = get_test_set('QB/TestFolder/ms_label', 'QB/TestFolder/pan_label',
                        'QB/TestFolder/ms', 'QB/TestFolder/pan')
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

Encoder_ms = MS_encoder()
Encoder_pan = PAN_encoder()
Decoder_ps = PS_decoder()

if cuda:
    Encoder_ms = Encoder_ms.cuda(gpus_list[0])
    Encoder_pan = Encoder_pan.cuda(gpus_list[0])
    Decoder_ps = Decoder_ps.cuda(gpus_list[0])

for epoch3 in range(42, 43):
    print(epoch3)
    Encoder_ms_name = opt.model_dir+"Encoder_ms_epoch_{}.pth".format(epoch3)
    Encoder_pan_name = opt.model_dir+"Encoder_pan_epoch_{}.pth".format(epoch3)
    Decoder_ps_name = opt.model_dir+"Decoder_ps_epoch_{}.pth".format(epoch3)

    Encoder_ms.load_state_dict(torch.load(Encoder_ms_name, map_location=lambda storage, loc: storage),False)
    Encoder_pan.load_state_dict(torch.load(Encoder_pan_name, map_location=lambda storage, loc: storage),False)
    Decoder_ps.load_state_dict(torch.load(Decoder_ps_name, map_location=lambda storage, loc: storage),False)
    test(epoch3)
