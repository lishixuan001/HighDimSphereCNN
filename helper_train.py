import argparse
import sys
import torch
import numpy as np
from torch.autograd import Variable



##########################################################
#                      Load Arguments                    #
##########################################################

def load_args():
    """
    [Training]
    Load arguments from user command input [attributes for training]
    :return: parsed arguments
    """
    mnist_data_path = '../mnistPC'
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize",
                        help="the batch size of the dataloader",
                        type=int,
                        default=2,
                        required=False)
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
    parser.add_argument("--epochs",
                        help="number of epochs",
                        type=int,
                        default=5,
                        required=False)
    parser.add_argument("--learning_rate",
                        help="learning rate of the model",
                        type=float,
                        default=1e-2,
                        required=False)
    parser.add_argument("--num_classes",
                        help="number of classes for classification",
                        type=int,
                        default=10,
                        required=False)
    parser.add_argument("--num_neighbors",
                        help="num of neighbors for the network",
                        type=int,
                        default=15,
                        required=False)
    parser.add_argument("--cudas",
                        help="cuda numbera to use, if use cpu, please enter -1",
                        type=str,
                        default="0/1/2/3",
                        required=False)
    args = parser.parse_args()
    return args


##########################################################
#                       Load Dataset                     #
##########################################################

def load_data(dataset_class, batch_size, shuffle=True, num_workers=4):
    """
    Load dataset into torch's DataLoader format
    :param dataset_class: class.data = (data_size, num_points, grid_size^3)
                          class.labels = (data_size, )
                          class.adjacent_matrix = (data_size, num_points, num_points)
    :param batch_size: batch size for loading data
    :param shuffle: boolean for if shuffle the loaded data or not
    :param num_workers: number of subprocesses to use for data loading
    :return: type => torch.utils.data.DataLoader
    """
    # FIXME : Labels be long type ?
    loader = torch.utils.data.TensorDataset(dataset_class.data.float(),
                                            dataset_class.labels,
                                            dataset_class.adjacent_matrix.float())
    loader_dataset = torch.utils.data.DataLoader(loader,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers)
    return loader_dataset


##########################################################
#                    Train Evaluation                    #
##########################################################

def evaluate(loader_dataset, model):
    accuracies = list()
    for batch_index, (dataset, labels, adjacent_matrix) in enumerate(loader_dataset):
        adjacent_matrix = adjacent_matrix.cuda()
        dataset = Variable(dataset).cuda()
        predictions = model(dataset, adjacent_matrix)
        prediction = torch.argmax(predictions, dim=-1)
        accuracy = np.mean(prediction.detach().cpu().numpy() == labels.numpy())
        accuracies.append(accuracy)
    return np.mean(accuracies)


##########################################################
#                      Progress Bar                    #
##########################################################

def progress(count, total):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('\r[%s] %s%s' % (bar, percents, '%'), end='')
    # sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    sys.stdout.flush()