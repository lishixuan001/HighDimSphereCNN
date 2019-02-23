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
    #                        Load Data                       #
    ##########################################################

    logger.info("==> Loading Configurations")

    num_epochs = args.epochs
    learning_rate = args.learning_rate
    cuda_id = args.cuda
    validation_size = args.validsize
    batch_size = args.batchsize


    if args.test:
        input_file_path = join(args.input_prefix, "test.gz")
    else:
        input_file_path = join(args.input_prefix, "train.gz")

    """ Data Size (num of images to be loaded)"""
    if args.demo:
        input_file_path = "demo_" + input_file_path

    logger.info("Loading MNIST Dataset")

    with gzip.open(os.path.join(args.input_prefix, input_file_path), 'rb') as f:
        dataset = pickle.load(f)

    model = ManifoldNet(10, 15)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)



if __name__ == '__main__':
    main()