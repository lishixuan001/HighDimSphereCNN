import torch 
import time
import torch.nn as nn
import torch.nn.functional as F
import wFM
import utils

class ManifoldNet(nn.Module):
    def __init__(self, num_classes, num_neighbor):
        super(ManifoldNet, self).__init__()
        self.wFM1 = wFM.wFMLayer(1, 10, num_neighbor)
        self.wFM2 = wFM.wFMLayer(10, 20, num_neighbor)
        self.wFM3 = wFM.wFMLayer(20, 30, num_neighbor)
        self.last = wFM.Last(10, num_classes)
    
    def forward(self, x):
        adj = utils.pairwise_distance(x)
        return self.last(self.wFM3(self.wFM2(self.wFM1(x, neighborhood_matrix), neighborhood_matrix), neighborhood_matrix))
        #return self.last(self.wFM1(x, adj))

