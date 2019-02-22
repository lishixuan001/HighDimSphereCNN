from model import ManifoldNet
import dataloader
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import utils
import h5py
from pdb import set_trace as st
import argparse

#trainloader = dataloader.getLoader("./mnistPC/train.hdf5", 80, 'train')

#optim = torch.optim.SGD(model.parameters(), lr=1e-6)
    
def eval(test_iterator, model, grid, sigma):
    acc_all = []
    for (inputs, labels) in (test_iterator):
        inputs = Variable(inputs).cuda()
        pred = model(utils.sdt(inputs, grid, sigma))
        pred = torch.argmax(pred, dim=-1)
        acc_all.append(np.mean(pred.detach().cpu().numpy() == labels.numpy()))
    return np.mean(acc_all)
        


def train(train_data_dir, test_data_dir, train_iter, log_interval, grid, sigma, batch_size):
    model = ManifoldNet(10, 15)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    test_iterator = utils.load_data(train_data_dir, batch_size=40)
    train_iterator = utils.load_data(test_data_dir, batch_size=batch_size)
    for epoch in range(train_iter):  # loop over the dataset multiple times
        running_loss = []
        cls_criterion = torch.nn.CrossEntropyLoss()
        for i, (inputs, labels) in enumerate(train_iterator):
            # get the inputs
            #inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            # zero the parameter gradients
            optim.zero_grad()

            # forward + backward + optimize
            outputs = model(utils.sdt(inputs, grid, sigma))
            #print(1 in torch.isnan(outputs).numpy())
            #print(labels.squeeze())
            #loss = F.cross_entropy(outputs, labels.squeeze())
            loss = cls_criterion(outputs, labels.squeeze())
            loss.backward(retain_graph=True)
            optim.step()

            # print statistics
            running_loss.append( loss.item() )
            if i % log_interval == 0:
                #acc = eval(test_iterator, model)
                #print(acc*100 +"%", np.mean(running_loss) )
                print(np.mean(running_loss))

        #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))
        print("Epoch"+str(epoch))
        print(np.mean(running_loss) )
        running_loss = []

    print('Finished Training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HighDimSphere Train')
    parser.add_argument('--data_path', default='./mnistPC',  type=str, metavar='XXX', help='Path to the model')
    parser.add_argument('--batch_size', default=80 , type=int, metavar='N', help='Batch size of test set')
    parser.add_argument('--max_epoch', default=200 , type=int, metavar='N', help='Epoch to run')
    parser.add_argument('--log_interval', default=10 , type=int, metavar='N', help='log_interval')
    parser.add_argument('--grid', default=10 , type=int, metavar='N', help='grid of sdt')
    parser.add_argument('--sigma', default=0.01 , type=float, metavar='N', help='sigma of sdt')
    
    args = parser.parse_args()
    test_data_dir = os.path.join(args.data_path, "test.hdf5")
    train_data_dir = os.path.join(args.data_path, "train.hdf5")
    train(train_data_dir, test_data_dir, args.max_epoch, args.log_interval, args.grid, args.sigma, args.batch_size)