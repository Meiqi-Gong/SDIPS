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
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Net_modify import MS_encoder, PAN_encoder, PS_decoder, MS_decoder, PAN_decoder, DS
from data import get_training_set, get_test_set, get_fulltraining_set
import socket
import time
import socket
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import xlwt
from scipy.misc import imresize
from MMD import mmd_rbf, SSIM_loss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize1', type=int, default=8, help='training batch size')
parser.add_argument('--batchSize2', type=int, default=4, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate. Default=0.01')#init:5e-4
parser.add_argument('--lrs', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)
writer = SummaryWriter("logs")

def erggg(hr, tar):
    hr = hr.permute(0,2,3,1)
    tar = tar.permute(0,2,3,1)

    channels = hr.shape[3]
    inner_sum = 0
    for c in range(channels):
        band_img1 = hr[:, :, :, c]
        band_img2 = tar[:, :, :, c]
        rmse_value = torch.square(torch.sqrt(torch.mean(torch.square(band_img1 - band_img2))) / torch.mean(band_img1))
        inner_sum += rmse_value
    ergas = 100 / 4 * torch.sqrt(inner_sum / channels)
    return ergas

def trainPS(epoch):
    print('Training MS!')
    e = 0
    print('*****************************New epoch!******************************')
    print('Begin eval!')
    Decoder_ps.eval()
    Encoder_pan.eval()
    Encoder_ms.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(testing_data_loader, 1):
            ms_label, pan_label, ms, pan = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
            if cuda:
                pan = pan.cuda(gpus_list[0])
                ms = ms.cuda(gpus_list[0])
                ms_label = ms_label.cuda(gpus_list[0])

            ms = ms.type(torch.cuda.FloatTensor)
            pan = pan.type(torch.cuda.FloatTensor)
            ms_label = ms_label.type(torch.cuda.FloatTensor)

            feat_ms = Encoder_ms(ms)
            feat_pan = Encoder_pan(pan)
            HRMS3 = Decoder_ps(feat_ms, feat_pan)

            HRMS = HRMS3.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
            tar = ms_label.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
            erg = ERGAS(HRMS, tar)
            e += erg
            if iteration % 20 == 0:
                print("===> Epoch[{}]({}/{}):ERGAS: {:.4f}".format(epoch, iteration, len(testing_data_loader),
                                                                                        erg))
    print("===> Eval Complete: Avg. ERGAS: {:.4f}".format(e/len(testing_data_loader)))
    
    
    print('############Train pansharpening!##############')
    epoch_loss1 = 0
    epoch_losseg2 = 0
    Decoder_ps.train()
    Encoder_pan.train()
    Encoder_ms.train()
    DS.train()
    ##train PS
    for iteration, batch in enumerate(training_data_loader, 1):
        ms_label, pan_label, ms, pan = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
        if cuda:
            pan = pan.cuda(gpus_list[0])
            ms = ms.cuda(gpus_list[0])
            pan_label = pan_label.cuda(gpus_list[0])
            ms_label = ms_label.cuda(gpus_list[0])
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        ##pansharpening
        feat_ms = Encoder_ms(ms)
        feat_pan = Encoder_pan(pan)
        HRMS3 = Decoder_ps(feat_ms, feat_pan)
        _, _, h, w = ms_label.shape
        loss_1 = SSIM_loss(HRMS3, ms_label) + criterion(HRMS3, ms_label)
        loss_eg2 = erggg(HRMS3, ms_label)
        epoch_loss1 += loss_1.data
        epoch_losseg2 += loss_eg2.data
        loss = loss_1
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}):Loss_1fs: {:.4f}  Loss_erg1: {:.4f}".format(epoch, iteration,
                                                                                     len(training_data_loader), loss_1.data, loss_eg2.data))
    print("===> RR pansharpening: Epoch {} Complete: Avg. Loss_1fs: {:.4f} Avg. Loss_erg1: {:.4f}"
        .format(epoch, epoch_loss1 / len(training_data_loader), epoch_losseg2 / len(training_data_loader)))
    print('\n')

def trainDS(epoch):
    print('Training DS!')
    print('############Train pansharpening!##############')
    epoch_loss1 = 0
    DS.train()
    ##train DS
    for iteration, batch in enumerate(training_data_loader, 1):
        ms_label, pan_label, ms, pan = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
        if cuda:
            pan = pan.cuda(gpus_list[0])
            ms = ms.cuda(gpus_list[0])
            pan_label = pan_label.cuda(gpus_list[0])
            ms_label = ms_label.cuda(gpus_list[0])
        optimizer6.zero_grad()
        feat_ms = Encoder_ms(ms)
        feat_pan = Encoder_pan(pan)
        feat = torch.cat((feat_ms, feat_pan), 1)

        feat_msl = Encoder_ms(ms_label)
        feat_panl = Encoder_pan(pan_label)
        featl = torch.cat((feat_msl, feat_panl), 1)
        dfeatl = DS(featl)

        loss_1 = MSE(feat.detach(), dfeatl)
        epoch_loss1 += loss_1.data
        loss = loss_1
        loss.backward()
        optimizer6.step()

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}):Loss_1fs: {:.4f}".format(epoch, iteration, len(training_data_loader), loss_1.data))
    print("===> RR pansharpening: Epoch {} Complete: Avg. Loss_1fs: {:.4f}".format(epoch, epoch_loss1 / len(training_data_loader)))
    print('\n')

def trainRC(epoch, optim):
    Decoder_ps.eval()
    Encoder_pan.eval()
    Encoder_ms.eval()
    e=0
    with torch.no_grad():
        for iteration, batch in enumerate(testing_data_loader, 1):
            ms_label, pan_label, ms, pan = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
            if cuda:
                pan = pan.cuda(gpus_list[0])
                ms = ms.cuda(gpus_list[0])
                ms_label = ms_label.cuda(gpus_list[0])
                pan_label = pan_label.cuda(gpus_list[0])

            ms = ms.type(torch.cuda.FloatTensor)
            pan = pan.type(torch.cuda.FloatTensor)
            ms_label = ms_label.type(torch.cuda.FloatTensor)
            pan_label = pan_label.type(torch.cuda.FloatTensor)

            feat_ms = Encoder_ms(ms)
            feat_pan = Encoder_pan(pan)
            HRMS3 = Decoder_ps(feat_ms, feat_pan)

            HRMS = HRMS3.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
            tar = ms_label.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
            erg = ERGAS(HRMS, tar)
            e += erg
            if iteration % 20 == 0:
                print("===> Epoch[{}]({}/{}):ERGAS: {:.4f}".format(epoch, iteration, len(testing_data_loader),
                                                                                        erg))
    print("===> Eval Complete: Avg. ERGAS: {:.4f}".format(e/len(testing_data_loader)))
    
    
    Decoder_ms.train()
    Decoder_pan.train()
    Encoder_pan.train()
    Encoder_ms.train()
    Decoder_ps.train()
    DS.train()
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_loss3 = 0
    epoch_loss4 = 0
    print('############Train reconstruction!##############')
    ##FR reconstruction
    for iteration, batch in enumerate(training_data_loader, 1):
        ms_label, pan_label, ms, pan = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
        if cuda:
            pan = pan.cuda(gpus_list[0])
            ms = ms.cuda(gpus_list[0])
            pan_label = pan_label.cuda(gpus_list[0])
            ms_label = ms_label.cuda(gpus_list[0])

        if optim=='all':
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            optimizer5.zero_grad()
            optimizer6.zero_grad()
        elif optim=='finetune':
            optimizer1s.zero_grad()
            optimizer2s.zero_grad()
            optimizer3s.zero_grad()
            optimizer4s.zero_grad()
            optimizer5s.zero_grad()
            optimizer6s.zero_grad()
        ##reconstruction
        feat_ms = Encoder_ms(ms)
        feat_pan = Encoder_pan(pan)
        HRMS3 = Decoder_ps(feat_ms, feat_pan)
        ms_hat3 = Decoder_ms(feat_ms)
        pan_hat3 = Decoder_pan(feat_pan)

        feat_msl = Encoder_ms(ms_label)
        feat_panl = Encoder_pan(pan_label)
        feat = torch.cat((feat_ms, feat_pan), 1)
        featl = torch.cat((feat_msl, feat_panl), 1)
        dfeatl = DS(featl)
        _, _, h, w = ms_label.shape
        msl_hat3 = Decoder_ms(feat_msl)
        panl_hat3 = Decoder_pan(feat_panl)
        # w = -0.01 * (epoch - 1) + 1
        w=0.5
        
        loss_1 = SSIM_loss(HRMS3, ms_label) + criterion(HRMS3, ms_label)
        loss_4 = MSE(feat.detach(), dfeatl) + w * (mmd_rbf(feat_ms, feat_msl)+mmd_rbf(feat_pan, feat_panl))
        loss_2 = criterion(ms_hat3, ms) + SSIM_loss(ms_hat3, ms) + criterion(msl_hat3, ms_label) + SSIM_loss(msl_hat3,
                                                                                                             ms_label)
        loss_3 = criterion(pan_hat3, pan) + SSIM_loss(pan_hat3, pan) + criterion(panl_hat3, pan_label) + SSIM_loss(
            panl_hat3, pan_label)

        loss = 1 * (1 * loss_2 + 1 * loss_3 + 1 * loss_1) + 0.1 * loss_4

        epoch_loss2 += loss_2.data
        epoch_loss3 += loss_3.data
        epoch_loss1 += loss_1.data
        epoch_loss4 += loss_4.data
        loss.backward()
        if optim=='all':
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            optimizer6.step()
        elif optim=='finetune':
            optimizer1s.step()
            optimizer2s.step()
            optimizer3s.step()
            optimizer4s.step()
            optimizer5s.step()
            optimizer6s.step()

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}):Loss_ps: {:.4f} Loss_ms: {:.4f} Loss_pan: {:.4f} Loss_mmd: {:.4f}".
                  format(epoch, iteration, len(training_data_loader), loss_1.data, loss_2.data, loss_3.data, loss_4.data))
    print("===> FR reconstruction: Epoch {} Complete: Avg. Loss_ms: {:.4f} Avg. Loss_pan: {:.4f} Loss_ps: {:.4f} Loss_mmd: {:.4f}"
          .format(epoch, epoch_loss2 / len(training_data_loader), epoch_loss3 / len(training_data_loader), epoch_loss1 / len(training_data_loader),
                  epoch_loss4 / len(training_data_loader)))
    print('\n')

def checkpoint(name, epoch):
    if name == 'all':
        Encoder_ms_path = opt.save_folder + "Encoder_ms_epoch_{}.pth".format(epoch)
        torch.save(Encoder_ms.state_dict(), Encoder_ms_path)
        Encoder_pan_path = opt.save_folder + "Encoder_pan_epoch_{}.pth".format(epoch)
        torch.save(Encoder_pan.state_dict(), Encoder_pan_path)
        Decoder_ps_path = opt.save_folder + "Decoder_ps_epoch_{}.pth".format(epoch)
        torch.save(Decoder_ps.state_dict(), Decoder_ps_path)
        Decoder_ms_path = opt.save_folder + "Decoder_ms_epoch_{}.pth".format(epoch)
        torch.save(Decoder_ms.state_dict(), Decoder_ms_path)
        Decoder_pan_path = opt.save_folder + "Decoder_pan_epoch_{}.pth".format(epoch)
        torch.save(Decoder_pan.state_dict(), Decoder_pan_path)
        DS_path = opt.save_folder + "DS_epoch_{}.pth".format(epoch)
        torch.save(DS.state_dict(), DS_path)
        print("Checkpoint saved to {}".format(Decoder_ps_path))
    elif name == 'PS':
        Encoder_ms_path = opt.save_folder + "Encoder_ms_epoch_{}.pth".format(epoch)
        torch.save(Encoder_ms.state_dict(), Encoder_ms_path)
        Encoder_pan_path = opt.save_folder + "Encoder_pan_epoch_{}.pth".format(epoch)
        torch.save(Encoder_pan.state_dict(), Encoder_pan_path)
        Decoder_ps_path = opt.save_folder + "Decoder_ps_epoch_{}.pth".format(epoch)
        torch.save(Decoder_ps.state_dict(), Decoder_ps_path)
        print("Checkpoint saved to {}".format(Decoder_ps_path))
    elif name == 'DS':
        DS_path = opt.save_folder + "DS_epoch_{}.pth".format(epoch)
        torch.save(DS.state_dict(), DS_path)
        print("Checkpoint saved to {}".format(DS_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set('QB/TrainFolder/ms_label',
                             'QB/TrainFolder/pan_label',
                             'QB/TrainFolder/ms',
                             'QB/TrainFolder/pan')
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize2, shuffle=True)
test_set = get_test_set('QB/TestFolder/ms_label', 'QB/TestFolder/pan_label',
                        'QB/TestFolder/ms', 'QB/TestFolder/pan')
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

Encoder_ms = MS_encoder()
Encoder_pan = PAN_encoder()
Decoder_ps = PS_decoder()
Decoder_ms = MS_decoder()
Decoder_pan = PAN_decoder()
DS = DS()

criterion = nn.L1Loss()
MSE = nn.MSELoss()

if cuda:
    Encoder_ms = Encoder_ms.cuda(gpus_list[0])
    Encoder_pan = Encoder_pan.cuda(gpus_list[0])
    Decoder_ps = Decoder_ps.cuda(gpus_list[0])
    Decoder_ms = Decoder_ms.cuda(gpus_list[0])
    Decoder_pan = Decoder_pan.cuda(gpus_list[0])
    DS = DS.cuda(gpus_list[0])

optimizer1 = optim.Adam(Encoder_ms.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
optimizer2 = optim.Adam(Encoder_pan.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
optimizer3 = optim.Adam(Decoder_ps.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
optimizer4 = optim.Adam(Decoder_ms.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
optimizer5 = optim.Adam(Decoder_pan.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
optimizer6 = optim.Adam(DS.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

optimizer1s = optim.Adam(Encoder_ms.parameters(), lr=opt.lrs, betas=(0.9, 0.999), eps=1e-8)
optimizer2s = optim.Adam(Encoder_pan.parameters(), lr=opt.lrs, betas=(0.9, 0.999), eps=1e-8)
optimizer3s = optim.Adam(Decoder_ps.parameters(), lr=opt.lrs, betas=(0.9, 0.999), eps=1e-8)
optimizer4s = optim.Adam(Decoder_ms.parameters(), lr=opt.lrs, betas=(0.9, 0.999), eps=1e-8)
optimizer5s = optim.Adam(Decoder_pan.parameters(), lr=opt.lrs, betas=(0.9, 0.999), eps=1e-8)
optimizer6s = optim.Adam(DS.parameters(), lr=opt.lrs, betas=(0.9, 0.999), eps=1e-8)


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

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    if epoch > 1 and epoch % 5 == 0:
        for optimizer in [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5, optimizer6]:
            for param in optimizer.param_groups:
                param['lr'] *= 0.95
    if epoch > 40:
        for optimizer in [optimizer1s, optimizer2s, optimizer3s, optimizer4s, optimizer5s, optimizer6s]:
            for param in optimizer.param_groups:
                param['lr'] *= 0.95


    if epoch<=40:
        print('Learning rate decay now: lr={}'.format(optimizer2.param_groups[0]['lr']))
        trainPS(epoch)
        checkpoint('PS', epoch)
    else:
        print('Learning rate decay now: lrs={}'.format(optimizer2s.param_groups[0]['lr']))
        trainRC(epoch, 'finetune')
        checkpoint('all', epoch)