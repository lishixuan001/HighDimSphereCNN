<<<<<<< HEAD
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import wFM
import utils
from pdb import set_trace as st


class ManifoldNet(nn.Module):
    def __init__(self, num_classes, num_neighbor):
        super(ManifoldNet, self).__init__()
        self.wFM1 = wFM.wFMLayer(1, 10, num_neighbor).cuda()
        self.wFM2 = wFM.wFMLayer(10, 20, num_neighbor).cuda()
        self.wFM3 = wFM.wFMLayer(20, 30, num_neighbor).cuda()
        self.last = wFM.Last(30, num_classes).cuda()

    def forward(self, x, neighborhood_matrix):
        return self.last(
            self.wFM3(self.wFM2(self.wFM1(x, neighborhood_matrix), neighborhood_matrix), neighborhood_matrix))
=======
import torch 
import time
import torch.nn as nn
import torch.nn.functional as F
import wFM
import utils
from pdb import set_trace as st

class ManifoldNet(nn.Module):
    def __init__(self, num_classes, num_neighbor):
        super(ManifoldNet, self).__init__()
        self.wFM1 = wFM.wFMLayer(1, 10, num_neighbor).cuda()
        self.wFM2 = wFM.wFMLayer(10, 20, num_neighbor).cuda()
        self.wFM3 = wFM.wFMLayer(20, 30, num_neighbor).cuda()
        self.last = wFM.Last(30, num_classes).cuda()
    
    def forward(self, x, neighborhood_matrix):
        return self.last(self.wFM3(self.wFM2(self.wFM1(x, neighborhood_matrix), neighborhood_matrix), neighborhood_matrix))
        #return self.last(self.wFM1(x, adj))

>>>>>>> 53123aaf1be83633dff0ba85523c13718e2e8f88
