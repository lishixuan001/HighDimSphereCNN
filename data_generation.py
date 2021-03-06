import h5py
import gzip
import pickle
import os
import torch
import utils
from logger import *
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

    logger.info("--> Loading Configurations")
    if args.test:
        f_data = f_test
        output_file_name = "test.gz"
        output_path = join(args.output_prefix, "test")
    else:
        f_data = f_train
        output_file_name = "train.gz"
        output_path = join(args.output_prefix, "train")

    """ Data Size (num of images to be loaded)"""
    if args.demo:
        data_size = args.demo # Number of images to be loaded
        output_file_name = "demo_" + output_file_name
        output_path = join(args.output_prefix, "demo")
    else:
        data_size = f_data['data'].shape[0]
    
    """ Number of Points (MNIST => 512)"""
    num_points = f_data['data'].shape[1]

    """ Load Labels """
    logger.info("--> Loading Labels")
    np_labels = np.array(f_data['labels'])[0:data_size] # [1, 2, 3]
    tensor_labels = torch.from_numpy(np_labels)

    """ Load Raw Data Set """
    logger.info("--> Loading Data Set")
    np_dataset = np.array(f_data['data'])[0:data_size] # (data_size, 512, 2)
    tensor_dataset = torch.from_numpy(np_dataset).float() # convert from numpy.ndarray to torch.Tensor

    """ Adjust Data Dimension """
    if tensor_dataset.shape[-1] == 2:
        """ 
        if deal with 2D point set, have to add one dimension as z dimension
        z dimension should be padded with 0, since point is ranged from -1 to 1, 0 is the average value
        """
        # (train_size * num_points, 3) -> z-dimension additionally padded by 0 -> (x, y, 0)
        logger.info("--> Data Dimension Adjustment Operated")
        zero_padding = torch.zeros((data_size, num_points, 1), dtype=tensor_dataset.dtype)
        tensor_dataset = torch.cat((tensor_dataset, zero_padding), -1) # (data_size, num_points, 3)

    logger.info("Finish Loading MNIST Data and Basic Configuration")

    ##########################################################
    #                     Adjacent Matrix                    #
    ##########################################################

    logger.info("Start Computing Adjacent Matrix")

    """ Adjacent Matrix """
    batch_size = args.batch_size
    start, end = 0, args.batch_size

    total_count = tensor_dataset.size()[0]
    adj_tensor_datasets = []
    while end < tensor_dataset.size()[0]:
        tensor_dataset_subset = tensor_dataset[start : end]
        tensor_dataset_subset_adj = utils.pairwise_distance(tensor_dataset_subset)
        adj_tensor_datasets.append(tensor_dataset_subset_adj)
        progress(end, total_count)
        start += batch_size
        end += batch_size
    tensor_dataset_subset = tensor_dataset[start: tensor_dataset.size()[0]]
    tensor_dataset_subset_adj = utils.pairwise_distance(tensor_dataset_subset)
    adj_tensor_datasets.append(tensor_dataset_subset_adj)
    progress(total_count, total_count)

    adjacent_matrix = torch.cat(tuple(adj_tensor_datasets), dim=0)

    logger.info("Finish Computing Adjacent Matrix".ljust(60))

    ##########################################################
    #                     Data Generation                    #
    ##########################################################

    logger.info("Start Computing Dataset Generation")

    """ Normalization (Raw) """
    logger.info("--> Normalizing Raw Data")

    tensor_dataset = raw_data_normalization(tensor_dataset)

    """ Grid """
    logger.info("--> Constructing Grid")

    grid_size = args.grid_size
    grid = grid_generation(grid_size)

    """ Mapping and Normalization """
    logger.info("--> Computing Mapping and Normalization")
    sigma = args.sigma
    batch_size = args.batch_size
    start, end = 0, args.batch_size
#     index, save_batch = 0, args.save_batch
#     save_start = 0

    total_count = tensor_dataset.size()[0]
    mapped_tensor_datasets = list()
    while end < tensor_dataset.size()[0]:
        tensor_dataset_subset = tensor_dataset[start : end] # (batch_size, num_points, 3)
        tensor_dataset_mapped_norm = map_and_norm(tensor_dataset_subset, grid, sigma) # (batch_size, num_points, grid_size^3)
        mapped_tensor_datasets.append(tensor_dataset_mapped_norm)
        
#         """ Saving Processes """
#         if len(mapped_tensor_datasets) >= save_batch:
#             save_file_name = "part{index}_{filename}".format(index=index, filename=output_file_name)
#             tensor_dataset_part = torch.cat(tuple(mapped_tensor_datasets), dim=0)
#             tensor_labels_part = tensor_labels[save_start : end]
#             adjacent_matrix_part = adjacent_matrix[save_start : end]
            
#             print(tensor_dataset_part.size())
#             print(tensor_labels_part.size())
#             print(adjacent_matrix_part.size())
            
#             dataConstructor = DatasetConstructor(tensor_dataset_part, tensor_labels_part, adjacent_matrix_part)
#             print(sys.getsizeof(object))
            
#             with gzip.open(os.path.join(output_path, save_file_name), 'wb') as file:
#                 pickle.dump(dataConstructor, file)
            
#             mapped_tensor_datasets = list()
#             save_start = end
#             index += 1
        
        progress(end, total_count)
        start += batch_size
        end += batch_size
        
    tensor_dataset_subset = tensor_dataset[start: tensor_dataset.size()[0]]
    tensor_dataset_mapped_norm = map_and_norm(tensor_dataset_subset, grid, sigma)
    mapped_tensor_datasets.append(tensor_dataset_mapped_norm)
    tensor_dataset = torch.cat(tuple(mapped_tensor_datasets), dim=0)
    
#     print(tensor_dataset_part.size())
#     print(tensor_labels_part.size())
#     print(adjacent_matrix_part.size())

#     save_file_name = "part{index}_{filename}".format(index=index, filename=output_file_name)
#     tensor_dataset_part = torch.cat(tuple(mapped_tensor_datasets), dim=0)
#     tensor_labels_part = tensor_labels[save_start : tensor_dataset.size()[0]]
#     adjacent_matrix_part = adjacent_matrix[save_start : tensor_dataset.size()[0]]
    
#     with gzip.open(os.path.join(output_path, save_file_name), 'wb') as file:
#         pickle.dump(DatasetConstructor(tensor_dataset_part, tensor_labels_part, adjacent_matrix_part), file)
    
    progress(total_count, total_count)

    logger.info("Finish Dataset Generation Processes".ljust(60))

    ##########################################################
    #                       Data Saving                      #
    ##########################################################

    logger.info("Start Saving Dataset")
    
    total_count = tensor_dataset.size()[0]
    index, save_batch = 0, args.save_batch
    start, end = 0, args.save_batch
    while end < tensor_dataset.size()[0]:
        output_file_name = "part{index}_{filename}".format(index=index, filename=output_file_name)
        tensor_dataset_part = tensor_dataset[start : end]
        tensor_labels_part = tensor_labels[start : end]
        adjacent_matrix_part = adjacent_matrix[start : end]
        with gzip.open(os.path.join(output_path, output_file_name), 'wb') as file:
            pickle.dump(DatasetConstructor(tensor_dataset_part, tensor_labels_part, adjacent_matrix_part), file)
        progress(end, total_count)
        index += 1
        start += save_batch
        end += save_batch
        
    logger.info("Finish Saving Dataset")

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# END
