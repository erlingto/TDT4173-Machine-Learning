import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random

USE_CUDA = True if torch.cuda.is_available() else False
#USE_CUDA = False

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
        if USE_CUDA: # torch.cuda.is_available():
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

