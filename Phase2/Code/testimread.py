import pandas as pd
import numpy as np
import cv2
from Utilities import read_data

def main():

    
    image, label = read_data("Phase2/Data/Data_Generation/Train/", 1)
    print(image.shape)
    cv2.imshow('first', image[0:3, :, :])


if __name__ == '__main__':
    main()
