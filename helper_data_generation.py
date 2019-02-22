import argparse
from os.path import join
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import distance as spdist
from enum import Enum



##########################################################
#                       Data Loader                      #
##########################################################

class MNIST(Dataset):
    def __init__(self, tensor_data, tensor_labels):
        self.data = tensor_data  # (data_size, num_points, grid_size^3)
        self.labels = tensor_labels  # (data_size,)

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
    parser.add_argument("--bandwidth",
                        help="the bandwidth of the S2 signal",
                        type=int,
                        default=10,
                        required=False)
    parser.add_argument("--batch_size",
                        help="the batch size of the dataloader",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--demo",
                        help="if demo is true, then only load small number of images",
                        type=bool,
                        default=False,
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

def get_min_distance(point_cloud):
    """
    calculate the minimal distance of the given point cloud
    :param point_cloud: tensor (train_size, num_points, num_dims)
    :return: half of the minimal distance between 2 points, which is assigned as the radius of sphere
    """
    num_points_in_cloud = point_cloud.shape[0]
    min_distance = np.inf
    for index in range(num_points_in_cloud):
        points_coords = point_cloud[index]
        pairwise_point_distances = spdist.pdist(points_coords)
        min_distance = min(pairwise_point_distances.min(), min_distance)
    return min_distance


##########################################################
#                         Utility                        #
##########################################################

class UtilityTypes(Enum):
    Gaussian = "Gaussian"
    Potential = "Potential"

class UtilityError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

