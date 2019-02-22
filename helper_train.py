import argparse
from os.path import join
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import distance as spdist
from enum import Enum



##########################################################
#                      Load Arguments                    #
##########################################################

def load_args():
    """
    [Training]
    Load arguments from user command input [attributes for training]
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize",
                        help="the batch size of the dataloader",
                        type=int,
                        default=8,  ###
                        required=True)
    parser.add_argument("--validsize",
                        help="percentage that training set split into validation set",
                        type=float,
                        default=1/6,
                        required=False)
    parser.add_argument("--input-prefix",
                        help="file for saving the data (.gz file)",
                        type=str,
                        default="../mnistPC",
                        required=False)
    parser.add_argument("--epochs",
                        help="number of epochs",
                        type=int,
                        default=20,
                        required=False)
    parser.add_argument("--learning-rate",
                        help="learning rate of the model",
                        type=float,
                        default=5e-3,
                        required=False)
    parser.add_argument("--cuda",
                        help="cuda number to use, if use cpu, please enter -1",
                        type=int,
                        required=True)
    args = parser.parse_args()
    return args