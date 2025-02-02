#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
# import torchvision
from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms
from torch.optim import AdamW, SGD
from Network.Network import HomographyModel, LossFn
import cv2
import sys
import os
import numpy as np
import random
# import skimage
# import PIL
import os
import glob
import random
# from skimage import data, exposure, img_as_float
# import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
# from torchvision.transforms import ToTensor
import argparse
import shutil
import string
# from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Utilities import read_data, NUM_DATA 


def GenerateBatch(BasePath, DirNamesTrain, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinates corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    labels - Batch of coordinates
    """
    I1Batch = []
    labels = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(1, NUM_DATA)
        image, label = read_data(BasePath+DirNamesTrain, RandIdx)
        ImageNum += 1

        # ##########################################################
        # # Add any standardization or data augmentation here!
        # ##########################################################

        # # Append All Images and Mask
        I1Batch.append(torch.from_numpy(image))
        labels.append(torch.tensor(label))

    return torch.stack(I1Batch), torch.stack(labels)

def Generate_Val_Batch(BasePath, DirNamesVal, NumValSamples):
    I1Batch = []
    labels = []

    for ImageNum in range(1,NumValSamples):
        image, label = read_data(BasePath+DirNamesVal, ImageNum)
        I1Batch.append(torch.from_numpy(image))
        labels.append(torch.tensor(label))
        
    return torch.stack(I1Batch), torch.stack(labels)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    DirNamesVal,
    NumTrainSamples,
    NumValSamples,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel()

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)

    # Tensorboard
    # Create a summary to monitor loss tensor
    # Writer = SummaryWriter(LogsPath)


    mps = torch.device("mps")

    # if LatestFile is not None:
    #     CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
    #     # Extract only numbers from the name
    #     StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
    #     model.load_state_dict(CheckPoint["model_state_dict"])
    #     print("Loaded latest checkpoint with the name " + LatestFile + "....")
    # else:
    #     StartEpoch = 0
    #     print("New model initialized....")

    StartEpoch = 0
    # model.to(mps)

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        epoch_loss = 0
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, labels = GenerateBatch(BasePath, DirNamesTrain, MiniBatchSize)

            # I1Batch.to(mps)
            # labels.to(mps)

            # Predict output with forward pass
            PredicatedCoordinatesBatch = model(I1Batch)
            LossThisBatch = LossFn(PredicatedCoordinatesBatch, labels)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()
            epoch_loss += LossThisBatch

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            # result = model.validation_step(GenerateBatch(BasePath, DirNamesVal, MiniBatchSize))
            
            # Tensorboard
            # Writer.add_scalar(
            #     "LossEveryIter",
            #     result["val_loss"],
            #     Epochs * NumIterationsPerEpoch + PerEpochCounter,
            # )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            # Writer.flush()
        

        with torch.no_grad():
            val_ims, val_labels = GenerateBatch(BasePath, DirNamesVal, NumValSamples)
            # val_ims.to(mps)
            # val_labels.to(mps)
            result = model.validation_step((val_ims, val_labels))
        
        print(f"Validation Loss: {result["val_loss"]}, Training Loss: {epoch_loss}")
        
        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="Phase2/Data/Data_Generation/",
        help="Base path of images, Default:Phase2/Data/Data_Generation",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=512,
        help="Size of the MiniBatch to use, Default:32",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType
    DirNamesTrain = "Train/"
    DirNamesVal = "Val/"

    # # Setup all needed parameters including file reading
    # (
    #     DirNamesTrain,
    #     SaveCheckPoint,
    #     ImageSize,
    #     NumTrainSamples,
    #     TrainCoordinates,
    #     NumClasses,
    # ) = SetupAll(BasePath, CheckPointPath)

    

    NumTrainSamples = next(os.walk(BasePath+"Train/Homographies"))[2]
    NumTrainSamples = len(NumTrainSamples)

    NumValSamples = next(os.walk(BasePath+"Val/Homographies"))[2] #directory is your directory path as string
    NumValSamples = int(len(NumValSamples) * 0.1)
    
    SaveCheckPoint = 100

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        DirNamesVal,
        NumTrainSamples,
        NumValSamples,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType
    )


if __name__ == "__main__":
    main()
