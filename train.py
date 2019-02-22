import torch.nn as nn
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import torch.nn.functional as F
import gzip
import pickle
import os
import argparse
from helper_train import *
from logger_setup import *
from torch.utils.data import DataLoader
import pdb


def main():
    logger = setup_logger('Training')
    args = load_args()
    logger.info("call with args: \n{}".format(args))

    ##########################################################
    #                      Load Dataset                      #
    ##########################################################

    logger.info("Loading MNIST Dataset")



if __name__ == '__main__':
    main()