import argparse
import sys
import torch
from os.path import join
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import distance as spdist
from enum import Enum



##########################################################
#                       Data Loader                      #
##########################################################

class DatasetConstructor(Dataset):
    def __init__(self, tensor_data, tensor_labels, adjacent_matrix):
        self.data = tensor_data  # (data_size, num_points, grid_size^3)
        self.labels = tensor_labels  # (data_size,)
        self.adjacent_matrix = adjacent_matrix # (data_size, num_points, num_points)

    def __len__(self):
        return self.data.shape[0] # data_size

    def __getitem__(self, index):
        data = self.data[index] # (num_points, grid_size^3)
        label = self.labels[index] # label
        element = dict(
            data=data,
            label=label
        )
        return element

    def get_adj_matrix(self):
        return self.adjacent_matrix


##########################################################
#                      Load Arguments                    #
##########################################################

def load_args():
    """
    [Data Generation]
    Load arguments from user command input [attributes for data management]
    :return: parsed arguments
    """
    mnist_data_path = '../mnistPC'
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path",
                        help="the path of train file",
                        type=str,
                        default=join(mnist_data_path, "train.hdf5"),
                        required=False)
    parser.add_argument("--test-file-path",
                        help="the path of test file",
                        type=str,
                        default=join(mnist_data_path, "test.hdf5"),
                        required=False)
    parser.add_argument("--batch_size",
                        help="the batch size of the dataloader",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--demo",
                        help="if demo is true, then only load small number of images",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--test",
                        help="if test is true, then load data from test dataset",
                        type=bool,
                        default=True,
                        required=False)
    parser.add_argument("--grid_size",
                        help="set the grid size",
                        type=int,
                        default=20,
                        required=False)
    parser.add_argument("--sigma",
                        help="set the sigma for mapping",
                        type=int,
                        default=10,
                        required=False)
    parser.add_argument("--output-prefix",
                        help="file for saving the data output (.gz file)",
                        type=str,
                        default="../mnistPC",
                        required=False)
    args = parser.parse_args()
    return args


##########################################################
#                 Raw Data Normalization                 #
##########################################################

def raw_data_normalization(tensor_dataset):
    tensor_dataset_transpose = tensor_dataset.transpose(0, 2) # (3, num_points, data_size)

    for i in range(tensor_dataset_transpose.size()[0]):
        data_dimension = tensor_dataset_transpose[i]
        minimum = int(torch.min(data_dimension))
        maximum = int(torch.max(data_dimension))

        if maximum != minimum:
            frac = 2.0 / (maximum - minimum)
            subt = -1.0 * (minimum + 1.0 / frac)
            tensor_dataset_transpose[i] = torch.add(tensor_dataset_transpose[i], subt)
            tensor_dataset_transpose[i] = torch.mul(tensor_dataset_transpose[i], frac)

    tensor_dataset = tensor_dataset_transpose.transpose(0, 2)
    return tensor_dataset


##########################################################
#                     Grid Generation                    #
##########################################################

def grid_generation(grid_size):
    linspace = np.linspace(-1, 1, grid_size)
    grid = np.meshgrid(linspace, linspace, linspace)  # (3, grid_size, grid_size, grid_size)
    grid = torch.from_numpy(np.array(grid))
    grid = grid.reshape(grid.size()[0], -1).float()  # (3, grid_size^3)
    return grid


##########################################################
#                      Progress Bar                     #
##########################################################

def progress(count, total):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    sys.stdout.flush()
    

##########################################################
#                         Mapping                        #
##########################################################

def map_and_norm(tensor_dataset, grid, sigma):
    """
    Mapping the tensor_dataset and do normalization
    :param tensor_dataset: dataset in tensor type
    :param grid: grid dataset is mapped to
    :param sigma: sigma parameter in mapping computation
    :return: mapped and normalized tensor type dataset
    """
    """ Mapping """
    tensor_dataset_spread = tensor_dataset.unsqueeze(-1)  # (data_size, num_points, 3, 1)
    tensor_dataset_spread = tensor_dataset_spread.repeat(
        (1, 1, 1, grid.size()[-1]))  # (data_size, num_points, 3, grid_size^3)
    grid_spread = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, grid_size^3)
    tensor_dataset_spread = tensor_dataset_spread - grid_spread  # (data_size, num_points, 3, grid_size^3)
    tensor_dataset_spread_transpose = tensor_dataset_spread.transpose(2, 3)  # (data_size, num_points, grid_size^3, 3)
    tensor_dataset_spread_transpose_norms = torch.norm(tensor_dataset_spread_transpose, dim=3, p=2,
                                                       keepdim=True)  # (data_size, num_points, grid_size^3, 1)

    tensor_dataset = torch.div(tensor_dataset_spread_transpose_norms,
                               -2.0 * np.power(sigma, 2))  # (data_size, num_points, grid_size^3, 1)
    tensor_dataset = torch.exp(tensor_dataset)  # (data_size, num_points, grid_size^3, 1)
    tensor_dataset = tensor_dataset.squeeze(-1)  # (data_size, num_points, grid_size^3)

    """ Normalization (Mapping) """
    tensor_dataset_norms = torch.norm(tensor_dataset, dim=2, p=2, keepdim=True)  # (data_size, num_points, 1)
    tensor_dataset_norms = tensor_dataset_norms.repeat(1, 1, grid.size()[-1])  # (data_size, num_points, grid_size^3)
    tensor_dataset = tensor_dataset / tensor_dataset_norms  # (data_size, num_points, grid_size^3)

    return tensor_dataset
