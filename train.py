import gzip
import cPickle as pickle
import os
from helper_train import *
from logger import *
from model import *
from torch.autograd import Variable
from os.path import join
from pdb import *


def main():
    logger = setup_logger('Training')
    args = load_args()
    logger.info("call with args: \n{}".format(args))

    ##########################################################
    #                        Load Data                       #
    ##########################################################

    logger.info("==> Loading Configurations")

    validation_size = args.validsize
    batch_size = args.batchsize
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    num_classes = args.num_classes
    num_neighbors = args.num_neighbors
    cudas = args.cudas.split("/") # [0, 1, 2, 3]



    if args.test:
        input_file_name = "test.gz"
    else:
        input_file_name = "train.gz"

    """ Data Size (num of images to be loaded)"""
    if args.demo:
        input_file_name = "demo_" + input_file_name

    input_file_path = join(args.input_prefix, input_file_name)

    logger.info("Loading Dataset")
    logger.info("==> Loading DatasetConstructor")

    # dataset_class => class.data = (data_size, num_points, grid_size^3)
    #                  class.labels = (data_size, )
    #                  class.adjacent_matrix = (data_size, num_points, num_points)
    with gzip.open(os.path.join(args.input_prefix, input_file_path), 'rb') as f:
        dataset_class = pickle.load(f)

    logger.info("==> Creating Dataloader")
    loader_dataset = load_data(dataset_class=dataset_class,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=len(cudas)) # FIXME: euqals number of cudas ?
    num_batchs = len(loader_dataset)

    logger.info("Finish Loading Dataset")

    ##########################################################
    #                       Data Train                       #
    ##########################################################

    logger.info("Start Training")

    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(cudas) # "0, 1, 2, 3"

    model = ManifoldNet(num_classes, num_neighbors).cuda()
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in num_epochs:
        running_loss = list()
        cls_criterion = torch.nn.CrossEntropyLoss().cuda()

        for batch_index, (dataset, labels, adjacent_matrix) in enumerate(loader_dataset):
            dataset, labels = Variable(dataset).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(dataset, adjacent_matrix) # (data_size, num_classes)

            loss = cls_criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

            logger.info("[epoch] {epoch}/{num_epochs} | [batch] {batch}/{num_batchs} | [loss] {loss}".format(epoch=epoch,
                                                                                                             num_epochs=num_epochs,
                                                                                                             batch=batch_index,
                                                                                                             num_batchs=num_batchs,
                                                                                                             loss=loss.item()))
        accuracy = evaluate(loader_dataset, model)
        mean_loss = np.mean(running_loss)

        logger.info("==EPOCH [{epoch}/{num_epochs}]== Loss: [{mean_loss}] | Accuracy: [{accuracy}]".format(epoch=epoch,
                                                                                                           num_epochs=num_epochs,
                                                                                                           mean_loss=mean_loss,
                                                                                                           accuracy=accuracy))


    logger.info("Finish Training")

if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# END
