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
from pdb import set_trace as st

train_iter = 1000
trainloader = dataloader.getLoader("./mnistPC/train.hdf5", 10, 'train')
grid = 20
sigma = 10
model = ManifoldNet(10, 15)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
#optim = torch.optim.SGD(model.parameters(), lr=1e-6)

for epoch in range(train_iter):  # loop over the dataset multiple times

    running_loss = 0.0
    cls_criterion = torch.nn.CrossEntropyLoss()
    for i, (inputs, labels) in enumerate(trainloader):
        # get the inputs
        #inputs, labels = data
        adj = utils.pairwise_distance(inputs)
        x = inputs
        dim = x.shape[2]
        num_point = x.shape[1]
        linspace = np.linspace(-1,1,grid)
        mesh = linspace
        for i in range(dim-1):
            mesh = np.meshgrid(mesh, linspace)
        mesh = torch.from_numpy(np.array(mesh))#.cuda()
        mesh = mesh.reshape(mesh.shape[0], -1).float()
        
        temp = x.unsqueeze(-1).repeat( 1,1,1,mesh.shape[-1])
        temp = temp - mesh.unsqueeze(0).unsqueeze(0)#torch.from_numpy(np.expand_dims(np.expand_dims(mesh, 0),0)).cuda()
        out = torch.sum(temp**2, -2)
        out = torch.exp(-out/(2*sigma))
        norms = torch.norm(out, dim = 2, keepdim=True)
        inputs = (out/norms).unsqueeze(-1)
        # zero the parameter gradients
        optim.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, adj)
        print(1 in torch.isnan(outputs).numpy())
        #print(labels.squeeze())
        #loss = F.cross_entropy(outputs, labels.squeeze())
        loss = cls_criterion(outputs, labels.squeeze())
        loss.backward()
        optim.step()
        print("loss is "+str(loss.item()))

        # print statistics
        running_loss += loss.item()
    #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))
    running_loss = 0.0

print('Finished Training')