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
import Utilities
import cv2
import copy
# Don't generate pyc codes
sys.dont_write_bytecode = True

# TODO: check if size (64,8) works as expected for this loss func
# TODO: what if multiplying output by 10 gives us something that is meaningful?
# TODO: manually calculate MSE for each h4pt instead of getting element-wise mean...
def LossFn_sup(predicted_H_4pt: torch.Tensor, ground_truth_H_4pt: torch.Tensor):
    ground_truth_H_4pt = torch.reshape(ground_truth_H_4pt, predicted_H_4pt.shape)
    return torch.nn.functional.mse_loss(predicted_H_4pt, ground_truth_H_4pt)

def LossFn_unsup(x, ground_truth_patches):
    patches_b = ground_truth_patches[:, 0:1, :, :]

    # num_patches_shown = 5
    # for i in range(num_patches_shown):
    #     patch = patches_b[i,:,:,:].numpy()
    #     estim = x[i,:,:,:].detach().numpy()
    #     patch = np.squeeze(np.transpose(patch, axes=(1,2,0)))
    #     estim = np.squeeze(np.transpose(estim, axes=(1,2,0)))
    #     patch_row = patch if i == 0 else np.hstack((patch_row, patch))
    #     estim_row = estim if i == 0 else np.hstack((estim_row, estim))
    #     black_bar = np.zeros((128,2))
    #     patch_row = np.hstack((patch_row, black_bar))
    #     estim_row = np.hstack((estim_row, black_bar))
    
    # hori_black_bar = np.zeros((2, patch_row.shape[1]))
    # full = np.vstack((patch_row,hori_black_bar, estim_row))
    # cv2.imshow('TOP: original. BOTTOM: estimation', np.uint8(full))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return F.l1_loss(patches_b,x)

class HomographyModel(pl.LightningModule):
    def __init__(self, ModelType):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.ModelType = ModelType
        self.model = Net(ModelType)

    def forward(self, x, corners=None, image_idx=None):
        return self.model(x, corners, image_idx)

    def validation_step(self, batch):
        homographies, labels, corners, image_idx = batch
        delta, __ = self.model(homographies, corners, image_idx, is_train=False)
        loss = LossFn_sup(delta, labels) if self.ModelType=='Sup' else LossFn_unsup(delta, homographies)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Net(nn.Module):
    def __init__(self, ModelType):
        self.ModelType = ModelType
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(3,3), padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU()
        self.mp3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU()
        
        self.flatten = nn.Flatten() # TODO: is this correct. Should we make it 64, 1, 32768 instead of 64, 32768
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.fc1 = nn.Linear(32768, 1024)
        self.relu9 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 8)

        if(self.ModelType == 'Unsup'):
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
            )

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )

            self.localization = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
            )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x, image):
        "Spatial transformer network forward function"
        x = torch.from_numpy(np.array([x]))
        xs = self.localization(x.float())
        xs = xs.view(-1, 64 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x, corners, idx, is_train=True):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.mp1(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.mp2(x)

        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.mp3(x)

        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.relu8(self.bn8(self.conv8(x)))
        
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.relu9(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        if self.ModelType == 'Sup':
            return x
        else: 
            h_4pt = copy.deepcopy(x)

        
        # print(x[1, :])

        # tensor DLT
        x = Utilities.tensor_dlt(x, corners)
        # stn
        image = Utilities.get_image_from_idx(idx, is_train)
        x = Utilities.spacial_transform_layer(x, image, corners)
        # x = self.stn(x, image)
        # (64, 128, 128)
        x = x.to(device=torch.device("cuda"))
        return x, h_4pt
