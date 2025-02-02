"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(predicted_H_4pt: np.ndarray, ground_truth_H_4pt: np.ndarray):
    diff = predicted_H_4pt - ground_truth_H_4pt
    loss = np.linalg.norm(diff)
    return loss * 0.5


class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = Net()

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch):
        homographies, labels = batch
        delta = self.model(homographies)
        loss = LossFn(delta, labels)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Net(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=(3,3), stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()
        self.mp3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1)
        self.bn8 = nn.BatchNorm2d(64)
        self.relu8 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(32768, 1024)
        self.relu9 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 8)

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        # TODO: fix shape and try training
        # TODO: Remake data with 128 shape
        # TODO: fix datatypes...
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.bn1(x)
        print(x.shape)
        x = self.relu1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.mp1(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.mp2(x)

        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.mp2(x)

        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.relu8(self.bn8(self.conv8(x)))
        x = self.mp2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu9(self.fc1(x))
        x = self.fc2(x)
        return x
