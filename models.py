import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random


def calculate_conv_output(W, K, P, S):
    return int((W-K+2*P)/S)+1

def calculate_flat_input(dim_1, dim_2, dim_3):
    return int(dim_1*dim_2*dim_3)


class ClassifierNet(nn.Module):

    def __init__(self, img_rows, img_cols, dropout, dropout_rate):

        super(ClassifierNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=0)
        self.conv4 = nn.Conv2d(
            in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=0)
        self.conv5 = nn.Conv2d(
            in_channels=192, out_channels=350, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=0)
        """ CONV DIMENSIONS CALCULATIONS """
        self.conv_output_H = img_rows
        self.conv_output_W = img_cols

        for _ in range(4):
            """ CONV DIMENSIONS CALCULATIONS """
            self.conv_output_H = calculate_conv_output(
                self.conv_output_H, 3, 1, 1)
            self.conv_output_W = calculate_conv_output(
                self.conv_output_W, 3, 1, 1)
            """ POOLING DIMENSIONS CALCULATIONS """
            self.conv_output_H = calculate_conv_output(
                self.conv_output_H, 2, 0, 2)
            self.conv_output_W = calculate_conv_output(
                self.conv_output_W, 2, 0, 2)

        self.conv_output_H = calculate_conv_output(self.conv_output_H, 3, 1, 1)
        self.conv_output_W = calculate_conv_output(self.conv_output_W, 3, 1, 1)
        """ POOLING DIMENSIONS CALCULATIONS """
        self.conv_output_H = calculate_conv_output(self.conv_output_H, 2, 0, 2)
        self.conv_output_W = calculate_conv_output(self.conv_output_W, 2, 0, 2)
        print("Convolutional output: " + str(self.conv_output_H) +
              "X" + str(self.conv_output_W))

    
        self.linear = nn.Sequential(
            torch.nn.Linear(calculate_flat_input(350, self.conv_output_H, self.conv_output_W),  1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 5),
            nn.Softmax(dim=1)
            )
        if torch.cuda.is_available():
            self.cuda()
        else:
            print("NO CUDA to activate")

    # Ovveride the forward function in nn.Module
    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = self.pool1(x)
        x = F.relu(self.conv2(x.float()))
        x = self.pool2(x)
        x = F.relu(self.conv3(x.float()))
        x = self.pool3(x)
        x = F.relu(self.conv4(x.float()))
        x = self.pool4(x)
        x = F.relu(self.conv5(x.float()))
        x=self.pool5(x)
        x = x.view(-1, calculate_flat_input(1, self.conv_output_H, self.conv_output_W)*350)
        x= self.linear(x)
        return x

""" TAKEN FROM WEB: https://github.com/jindongwang/Pytorch-CapsuleNet/blob/master/capsnet.py?fbclid=IwAR30r7IwgFTkAvvM9aNB9GTvN2vrdF5Uc1fnsvYEf-hUDt1ckMAkfFed9Wk """ 

USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 42 * 42):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        print(u.size())
        u = u.view(x.size(0), self.num_routes, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=5, num_routes=32 * 42 * 42, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, input_width=100, input_height=100, input_channel=3):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_height * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        
        classes = F.softmax(classes, dim=1)
        

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(5))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self, config=None):
        super(CapsNet, self).__init__()
        if config:
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels,
                                                config.pc_kernel_size, config.pc_num_routes)
            self.digit_capsules = DigitCaps(config.dc_num_capsules, config.dc_num_routes, config.dc_in_channels,
                                            config.dc_out_channels)
            self.decoder = Decoder(config.input_width, config.input_height, config.cnn_in_channels)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()
        if USE_CUDA:
            self.cuda()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005

