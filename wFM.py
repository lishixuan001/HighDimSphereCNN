import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from pdb import *


def weightNormalize(weights_in):
    weights = weights_in ** 2
    weights = weights / torch.sum(weights, dim=1, keepdim=True)
    return weights  # torch.stack(out_all)


class wFMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbor):
        super(wFMLayer, self).__init__()
        # Initial input is B * N * D * C ----> B * N1 * D * C'
        # dont forget to normalize w in dim 0
        self.w1 = nn.Parameter(torch.randn(in_channels, num_neighbor))
        self.w2 = nn.Parameter(torch.randn(out_channels, in_channels))
        self.neighbors = num_neighbor
        self.out_channels = out_channels

    # Initial input is B * N * C * d ----> B * N1 * C * m
    def wFM_on_sphere(self, point_set, adj_mtr):
        # Input is B*N*D*C where B is batch size, N is number of points, D is dimension of each point, and C is input channel
        B, N, D, C = point_set.shape

        # assert(N >= neighbor_size) #make sure we can get
        k = self.neighbors  # Tis is number of neighbors

        idx = torch.arange(B) * N  # IDs for later processing, used because we flatten the tensor
        idx = idx.view((B, 1, 1))  # reshape to be added to knn indices
        k2 = knn(adj_mtr, k=k)  # B*N*k
        k2 = k2 + idx

        ptcld = point_set.view(B * N, D, C)  # reshape pointset to BN * DC
        ptcld = ptcld.view(B * N, D * C)
        # st()
        gathered = ptcld[k2]  # get matrix of dimension B*N*K*(D*C)
        # print(gathered.shape)
        gathered = gathered.view(B, N, k, D, C)
        north_pole_cos = torch.zeros(gathered.shape).cuda()
        theta = torch.acos(torch.clamp(gathered[:, :, :, 0, :], -1, 1))  # this is of shape B*N*K*C
        eps = (torch.ones(theta.shape) * 0.0001).cuda()
        theta_sin = theta / (torch.sin(theta) + eps)  # theta/sin(theta) B*N*K*D*C
        north_pole_cos[:, :, :, 0, :] = torch.cos(theta)  # cos(theta)
        q_p = gathered - north_pole_cos  # q-cos(theta)
        theta_sin = theta_sin.repeat(1, 1, 1, D)  # should be of shape B*N*K*D*C
        theta_sin = theta_sin.view(B, N, k, D, C)
        q_p_s = torch.mul(q_p, theta_sin)  # B*N*K*D*C

        q_p_s = torch.transpose(q_p_s, 2, 3)
        q_p_s = torch.transpose(q_p_s, 3, 4)  # Reshape to B*N*D*C*k

        # print(1 in torch.isnan(q_p_s).numpy())
        transformed_w1 = weightNormalize(self.w1)
        transformed_w2 = weightNormalize(self.w2).transpose(1, 0)
        # print(transformed_w1.shape)
        m = self.out_channels
        # print(self.weight)

        # if (1 in torch.isnan(self.weight).numpy()):
        # st()
        # q_p_s = q_p_s.repeat(1, 1, 1, 1, m)
        # q_p_s = q_p_s.view(B, N, D, C, k, m)

        # print(self.weight.shape)
        # print(q_p_s.shape)

        weighted = q_p_s * transformed_w1
        weighted = torch.mean(weighted, dim=-1)
        #       print(weighted.shape)
        #       print(transformed_w2.shape)
        weighted_sum = torch.matmul(weighted, transformed_w2)

        # print(weighted_sum.shape)

        # torch.matmul(q_p_s, self.weight) #q_p_s * self.weight\

        # weighted_sum = torch.mean(weighted, 4)
        # weighted_sum = torch.mean(weighted_sum, 3) #B*N*D*M
        # weighted_sum = torch.mean(weighted, -2)
        # print(1 in torch.isnan(weighted_sum).numpy())
        v_mag = torch.norm(weighted_sum, dim=2)
        north_pole_cos_vmag = torch.zeros(weighted_sum.shape).cuda()
        north_pole_cos_vmag[:, :, 0, :] = torch.cos(v_mag)
        normed_w = F.normalize(weighted_sum, p=2, dim=2)
        sin_vmag = torch.sin(v_mag).repeat(1, 1, D).view(B, N, D, m)
        out = north_pole_cos_vmag + sin_vmag * normed_w
        # print(1 in torch.isnan(v_mag).numpy())
        # print(self.weight)
        # print(1 in torch.isnan(self.weight).numpy())
        return out

    ## to do: implement inverse exponential mapping
    def forward(self, x, adj_mtr):
        return self.wFM_on_sphere(x, adj_mtr)


class Last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Last, self).__init__()
        # Initial input is B * N * D * C ----> B * N1 * D * C'
        self.linear = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    # Initial input is B * N * C * d ----> B * N1 * C * m
    def wFM_on_sphere(self, point_set):
        # Input is B*N*D*C where B is batch size, N is number of points, D is dimension of each point, and C is input channel
        B, N, D, C = point_set.shape

        north_pole_cos = torch.zeros(point_set.shape).cuda()  # B*N*D*C
        theta = torch.acos(torch.clamp(point_set[:, :, 0, :], -1, 1))  # this is of shape B*N*D*C
        eps = (torch.ones(theta.shape) * 0.0001).cuda()
        theta_sin = theta / (torch.sin(theta) + eps)  # theta/sin(theta) B*N*K*D*C
        north_pole_cos[:, :, 0, :] = torch.cos(theta)  # cos(theta)
        q_p = point_set - north_pole_cos  # q-cos(theta)
        theta_sin = theta_sin.repeat(1, 1, D)  # should be of shape B*N*K*D*C
        theta_sin = theta_sin.view(B, N, D, C)

        q_p_s = torch.mul(q_p, theta_sin)  # B*N*D*C

        unweighted_sum = torch.mean(q_p_s, 3)  # B*N*D

        # distance in terms of cosine
        # for each channel compute distance from mean to get B*N*C reshape to -> B*NC (can also do global maxpool)
        # print(1 in torch.isnan(unweighted_sum).numpy())

        v_mag = torch.norm(unweighted_sum, dim=2)
        north_pole_cos_vmag = torch.zeros(unweighted_sum.shape).cuda()
        north_pole_cos_vmag[:, :, 0] = torch.cos(v_mag)
        normed_w = F.normalize(unweighted_sum, p=2, dim=2)
        sin_vmag = torch.sin(v_mag).repeat(1, D).view(B, N, D)
        out = north_pole_cos_vmag + sin_vmag * normed_w

        out = out.unsqueeze(-1)
        x_ = torch.transpose(point_set, 2, 3)
        # print(point_set.shape)
        res = torch.matmul(x_, out).squeeze(-1)
        # print(res.shape)
        res = torch.acos(torch.clamp(res, -1, 1))
        # print("last layer "+str(1 in torch.isnan(res).numpy()))
        return torch.mean(res, dim=1)  # res.view(B, N*C)

    ## to do: implement inverse exponential mapping
    def forward(self, x):
        # print(self.wFM_on_sphere(x))
        return self.linear2(self.wFM_on_sphere(x))
