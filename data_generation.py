import h5py
import gzip
import pickle
import os
import torch
import utils
from logger_setup import *
from helper_data_generation import *



def main():
    logger = setup_logger('Data_Generation')
    args = load_args()
    logger.info("Received Args: \n{}".format(args))


    ##########################################################
    #                      Data Loading                      #
    ##########################################################

    logger.info("Start Loading MNIST Data and Configurations")

    f_train = h5py.File(args.train_file_path)
    f_test = h5py.File(args.test_file_path)

    logger.info("==> Loading Configurations")
    if args.test:
        f_data = f_test
        output_file_name = "test_mnist.gz"
    else:
        f_data = f_train
        output_file_name = "train_mnist.gz"

    """ Data Size (num of images to be loaded)"""
    if args.demo:
        data_size = 2 # Number of images to be loaded
        output_file_name = "demo_" + output_file_name
    else:
        data_size = f_data['data'].shape[0]

    """ Number of Points (MNIST => 512)"""
    num_points = f_data['data'].shape[1]

    """ Load Labels """
    logger.info("==> Loading Labels")
    tensor_labels = np.array(f_data['labels'])[0:data_size] # [1, 2, 3]

    """ Load Raw Data Set """
    logger.info("==> Loading Data Set")
    np_dataset = np.array(f_data['data'])[0:data_size] # train_size * 512 * 2
    tensor_dataset = torch.from_numpy(np_dataset).float() # convert from numpy.ndarray to torch.Tensor

    """ Adjust Data Dimension """
    if tensor_dataset.shape[-1] == 2:
        """ 
        if deal with 2D point set, have to add one dimension as z dimension
        z dimension should be padded with 0, since point is ranged from -1 to 1, 0 is the average value
        """
        # (train_size * num_points, 3) -> z-dimension additionally padded by 0 -> (x, y, 0)
        logger.info("==> Data Dimension Adjustment Operated")
        zero_padding = torch.zeros((data_size, num_points, 1), dtype=tensor_dataset.dtype)
        tensor_dataset = torch.cat((tensor_dataset, zero_padding), -1) # (data_size, num_points, 3)

    logger.info("Finish Loading MNIST Data and Basic Configuration")


    ##########################################################
    #                     Data Generation                    #
    ##########################################################

    logger.info("Start Computing Dataset Generation")

    """ Normalization (Raw) """
    # TODO: Try normalize by norm, but divide-by-zero problem cannot fix
    logger.info("==> Normalizing Raw Data")
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

    """ Grid """
    logger.info("==> Constructing Grid")
    grid_size = args.grid_size
    linspace = np.linspace(-1, 1, grid_size)
    grid = np.meshgrid(linspace, linspace, linspace) # (3, grid_size, grid_size, grid_size)
    grid = torch.from_numpy(np.array(grid))
    grid = grid.reshape(grid.size()[0], -1).float() # (3, grid_size^3)

    """ Mapping """
    logger.info("==> Calculating Mapping")
    sigma = args.sigma
    tensor_dataset_spread = tensor_dataset.unsqueeze(-1) # (data_size, num_points, 3, 1)
    tensor_dataset_spread = tensor_dataset_spread.repeat((1, 1, 1, grid.size()[-1])) # (data_size, num_points, 3, grid_size^3)
    grid_spread = grid.unsqueeze(0).unsqueeze(0) # (1, 1, 3, grid_size^3)
    tensor_dataset_spread = tensor_dataset_spread - grid_spread # (data_size, num_points, 3, grid_size^3)
    tensor_dataset_spread_transpose = tensor_dataset_spread.transpose(2, 3) # (data_size, num_points, grid_size^3, 3)
    tensor_dataset_spread_transpose_norms = torch.norm(tensor_dataset_spread_transpose, dim=3, p=2, keepdim=True) # (data_size, num_points, grid_size^3, 1)

    tensor_dataset = torch.div(tensor_dataset_spread_transpose_norms, -2.0 * np.power(sigma, 2)) # (data_size, num_points, grid_size^3, 1)
    tensor_dataset = torch.exp(tensor_dataset) # (data_size, num_points, grid_size^3, 1)
    tensor_dataset = tensor_dataset.squeeze(-1) # (data_size, num_points, grid_size^3)

    """ Normalization (Mapping) """
    logger.info("==> Normalizing Mapping Data")
    tensor_dataset_norms = torch.norm(tensor_dataset, dim=2, p=2, keepdim=True)  # (data_size, num_points, 1)
    tensor_dataset_norms = tensor_dataset_norms.repeat(1, 1, grid.size()[-1])  # (data_size, num_points, grid_size^3)
    tensor_dataset = tensor_dataset / tensor_dataset_norms  # (data_size, num_points, grid_size^3)

    logger.info("Finish Dataset Generation Processes")


    ##########################################################
    #                       Data Saving                      #
    ##########################################################

    logger.info("Start Saving Dataset")
    with gzip.open(os.path.join(args.output_prefix, output_file_name), 'wb') as file:
        pickle.dump(MNIST(tensor_dataset, tensor_labels), file)
    logger.info("Finish Saving Dataset")

if __name__ == '__main__':
    main()
