import torch 
import time
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import h5py 
from pdb import set_trace as st

def weightNormalize(weights_in):
    weights = weights_in**2
    weights = weights/ torch.sum(weights, dim = 1, keepdim= True)
    return weights #torch.stack(out_all)

class wFMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbor):
        super(wFMLayer, self).__init__()
        #Initial input is B * N * D * C ----> B * N1 * D * C'
        #dont forget to normalize w in dim 0
        self.w1 = nn.Parameter(torch.randn(in_channels, num_neighbor))
        self.w2 = nn.Parameter(torch.randn(out_channels, in_channels))
        self.neighbors = num_neighbor
        self.out_channels = out_channels


    #Initial input is B * N * C * d ----> B * N1 * C * m
    def wFM_on_sphere(self, point_set, adj_mtr):
      #Input is B*N*D*C where B is batch size, N is number of points, D is dimension of each point, and C is input channel
      B, N, D, C = point_set.shape

      #assert(N >= neighbor_size) #make sure we can get 
      k=self.neighbors #Tis is number of neighbors
      
      idx = torch.arange(B)*N #IDs for later processing, used because we flatten the tensor
      idx = idx.view((B, 1, 1)) #reshape to be added to knn indices
      k2 = knn(adj_mtr, k=k) #B*N*k
      k2 = k2+idx
      
      ptcld = point_set.view(B*N, D, C) #reshape pointset to BN * DC
      ptcld = ptcld.view(B*N, D*C)
      gathered=ptcld[k2] #get matrix of dimension B*N*K*(D*C)

      gathered = gathered.view(B, N, k, D, C)
      north_pole_cos = torch.zeros(gathered.shape)
      theta = torch.acos(torch.clamp(gathered[:, :, :, 0, :], -1, 1)) #this is of shape B*N*K*C
      eps = torch.ones(theta.shape)*0.0001
      theta_sin = theta / (torch.sin(theta) + eps) #theta/sin(theta) B*N*K*D*C
      north_pole_cos[:, :, :, 0, :] = torch.cos(theta) #cos(theta)
      q_p = gathered - north_pole_cos #q-cos(theta)
      theta_sin = theta_sin.repeat(1, 1, 1, D) #should be of shape B*N*K*D*C
      theta_sin = theta_sin.view(B, N, k, D, C)
      q_p_s = torch.mul(q_p, theta_sin) #B*N*K*D*C

      q_p_s = torch.transpose(q_p_s, 2, 3)
      q_p_s = torch.transpose(q_p_s, 3, 4) #Reshape to B*N*D*C*k

      #print(1 in torch.isnan(q_p_s).numpy())
      transformed_w1 = weightNormalize(self.w1)
      transformed_w2 = weightNormalize(self.w2).transpose(1, 0)
      #print(transformed_w1.shape)
      m=self.out_channels
      #print(self.weight)

      #if (1 in torch.isnan(self.weight).numpy()):
          #st()
      # q_p_s = q_p_s.repeat(1, 1, 1, 1, m)
      # q_p_s = q_p_s.view(B, N, D, C, k, m)

      # print(self.weight.shape)
      # print(q_p_s.shape)

      weighted = q_p_s * transformed_w1
      weighted = torch.mean(weighted, dim = -1)
#       print(weighted.shape)
#       print(transformed_w2.shape)
      weighted_sum = torch.matmul(weighted, transformed_w2)

      
      #print(weighted_sum.shape)

      #torch.matmul(q_p_s, self.weight) #q_p_s * self.weight\

      # weighted_sum = torch.mean(weighted, 4)
      # weighted_sum = torch.mean(weighted_sum, 3) #B*N*D*M
      #weighted_sum = torch.mean(weighted, -2)
      #print(1 in torch.isnan(weighted_sum).numpy())
      v_mag = torch.norm(weighted_sum, dim=2)
      north_pole_cos_vmag = torch.zeros(weighted_sum.shape)
      north_pole_cos_vmag[:, :, 0, :] = torch.cos(v_mag)
      normed_w = F.normalize(weighted_sum, p=2, dim=2)
      sin_vmag = torch.sin(v_mag).repeat(1, 1, D).view(B, N, D, m)
      out = north_pole_cos_vmag + sin_vmag*normed_w
      #print(1 in torch.isnan(v_mag).numpy())
      # print(self.weight)
      # print(1 in torch.isnan(self.weight).numpy())
      return out
    
    ## to do: implement inverse exponential mapping
    def forward(self, x, adj_mtr):
        return self.wFM_on_sphere(x, adj_mtr)

class Last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Last, self).__init__()
        #Initial input is B * N * D * C ----> B * N1 * D * C'
        self.linear = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )


    #Initial input is B * N * C * d ----> B * N1 * C * m
    def wFM_on_sphere(self, point_set):
      #Input is B*N*D*C where B is batch size, N is number of points, D is dimension of each point, and C is input channel
      B, N, D, C = point_set.shape

      north_pole_cos = torch.zeros(point_set.shape) #B*N*D*C
      theta = torch.acos(torch.clamp(point_set[:, :, 0, :], -1, 1)) #this is of shape B*N*D*C
      eps = torch.ones(theta.shape)*0.0001
      theta_sin = theta / (torch.sin(theta) + eps) #theta/sin(theta) B*N*K*D*C
      north_pole_cos[:, :, 0, :] = torch.cos(theta) #cos(theta)
      q_p = point_set - north_pole_cos #q-cos(theta)
      theta_sin = theta_sin.repeat(1, 1, D) #should be of shape B*N*K*D*C
      theta_sin = theta_sin.view(B, N, D, C)

      q_p_s = torch.mul(q_p, theta_sin) #B*N*D*C

      unweighted_sum = torch.mean(q_p_s, 3) #B*N*D

      #distance in terms of cosine
      #for each channel compute distance from mean to get B*N*C reshape to -> B*NC (can also do global maxpool)
      #print(1 in torch.isnan(unweighted_sum).numpy())

      v_mag = torch.norm(unweighted_sum, dim=2)
      north_pole_cos_vmag = torch.zeros(unweighted_sum.shape)
      north_pole_cos_vmag[:, :, 0] = torch.cos(v_mag)
      normed_w = F.normalize(unweighted_sum, p=2, dim=2)
      sin_vmag = torch.sin(v_mag).repeat(1, D).view(B, N, D)
      out = north_pole_cos_vmag + sin_vmag*normed_w
      
      out = out.unsqueeze(-1)
      x_ = torch.transpose(point_set, 2, 3)
      # print(point_set.shape)
      res = torch.matmul(x_, out).squeeze(-1)
      #print(res.shape)
      res = torch.acos(torch.clamp(res, -1, 1))
      #print("last layer "+str(1 in torch.isnan(res).numpy()))
      return torch.mean(res, dim = 1) #res.view(B, N*C)
    
    ## to do: implement inverse exponential mapping
    def forward(self, x):
        # print(self.wFM_on_sphere(x))
        return self.linear2(self.wFM_on_sphere(x))

def sdt(x, grid = 20, sigma = 1):
   dim = x.shape[2]
   num_point = x.shape[1]
   out = np.zeros((x.shape[0],x.shape[1],grid**dim,1))
   linspace = np.linspace(0,1,grid)
   mesh = linspace
   for i in range(dim-1):
       mesh = np.meshgrid(mesh, linspace)
   mesh = np.array(mesh)
   mesh = mesh.reshape(mesh.shape[0], -1)
   for batch_id in range(x.shape[0]):
       for id_, var in enumerate(mesh.T):
           var = var.reshape((1, -1))
           core_dis = np.sum( (np.squeeze(x[batch_id, ...]) -  np.repeat(var, num_point, axis = 0) ) **2, axis =1) *1.0 /(2*sigma)
           out[batch_id, :, id_,0] = np.exp( -core_dis)
   return out

# train = h5py.File("./mnistPC/train.hdf5", "r")
# print(train["data"].shape)
# train_ = sdt(train["data"])
# print(train_)

# def wFM_on_sphere(point_set, weight, num_filters=4, stride=1, neighbor_size=3, subset=9):
#   #Input is B*N*D*C where B is batch size, N is number of points, D is dimension of each point, and C is input channel
#   B, N, D, C = point_set.shape

#   adj_mtr = pairwise_distance(point_set)
#   assert(N >= neighbor_size) #make sure we can get 
#   k=neighbor_size #This is number of neighbors
  
#   idx = torch.arange(B)*N #IDs for later processing, used because we flatten the tensor
#   idx = idx.view((B, 1, 1)) #reshape to be added to knn indices
#   k2 = knn(adj_mtr, k=k) #B*N*k
#   k2 = k2+idx
  
#   ptcld = point_set.view(B*N, D, C) #reshape pointset to BN * DC
#   ptcld = ptcld.view(B*N, D*C)
#   gathered=ptcld[k2] #get matrix of dimension B*N*K*(D*C)

#   gathered = gathered.view(B, N, k, D, C)
#   north_pole_cos = torch.zeros(gathered.shape)
#   theta = torch.acos(torch.clamp(gathered[:, :, :, 0, :], -1, 1)) #this is of shape B*N*K*C
#   eps = torch.ones(theta.shape)*0.00001
#   theta_sin = theta / (torch.sin(theta) + eps) #theta/sin(theta) B*N*K*D*C
#   north_pole_cos[:, :, :, 0, :] = torch.cos(theta) #cos(theta)
#   q_p = gathered - north_pole_cos #q-cos(theta)
#   theta_sin = theta_sin.repeat(1, 1, 1, D) #should be of shape B*N*K*D*C
#   theta_sin = theta_sin.view(B, N, k, D, C)

#   q_p_s = torch.mul(q_p, theta_sin) #B*N*K*D*C

#   q_p_s = torch.transpose(q_p_s, 2, 3)
#   q_p_s = torch.transpose(q_p_s, 3, 4) #Reshape to B*N*D*C*k
 
#   m=8
#   q_p_s = q_p_s.repeat(1, 1, 1, 1, m)
#   q_p_s = q_p_s.view(B, N, D, C, k, m)

#   weighted = q_p_s * weight

#   weighted_sum = torch.mean(weighted, 4)
#   weighted_sum = torch.mean(weighted_sum, 3) #B*N*D*M

#   v_mag = torch.norm(weighted_sum, dim=2)
#   north_pole_cos_vmag = torch.zeros(weighted_sum.shape)
#   north_pole_cos_vmag[:, :, 0, :] = torch.cos(v_mag)
#   normed_w = F.normalize(weighted_sum, p=2, dim=2)
#   sin_vmag = torch.sin(v_mag).repeat(1, 1, D).view(B, N, D, m)
#   out = north_pole_cos_vmag + sin_vmag*normed_w

  # ##Initialize north pole
  # north_pole_cos_vmag = torch.zeros(gathered.shape)
  

  # theta =1 #dimension B*N*

  # ##Todo: implement exponential mapping
  # v_mag = torch.norm(point_set, dim=1)
  # north_pole_cos_vmag[:, 0] = torch.cos(v_mag)
  # sin_vmag = torch.transpose(torch.sin(v_mag).expand((C, N)), 0, 1)
  # tangent_mapped_points = sin_vmag * F.normalize(point_set, p=2, dim=1) + north_pole_cos_vmag 

  # ##Transform the weights, and do weighted mean on tangent plane
  # weights_transformed = torch.transpose(weights.expand((C, N)), 0, 1)
  # weighted_mean = torch.mean(tangent_mapped_points * weights_transformed, dim=0)

  # ##Apply inverse log map to get approximation of wFM
  # theta = torch.acos(torch.clamp(weighted_mean[0], -1, 1))
  # #print(theta)
  # north_pole_cos=torch.zeros(C)
  # north_pole_cos[0] = torch.cos(theta)
  # out = theta / torch.sin(theta) * (weighted_mean - north_pole_cos)
  # return out


# def pairwise_distance(point_cloud):
#     """Compute pairwise distance of a point cloud.
#     Args:
#       point_cloud: tensor (batch_size, num_points, num_dims, m)
#     Returns:
#       pairwise distance: (batch_size, num_points, num_points)
#     """
#     og_batch_size = point_cloud.shape[0] #point_cloud.get_shape().as_list()[0]
#     point_cloud = torch.squeeze(point_cloud)
#     if og_batch_size == 1:
#         point_cloud = point_cloud.unsqueeze(0) #torch.expand_dims(point_cloud, 0)
#     print(point_cloud.shape)
#     point_cloud_transpose = point_cloud.permute(0, 2, 3, 1)
#     print(point_cloud_transpose.shape)
#     #torch.transpose(point_cloud, perm=[0, 2, 1])
#     point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
#     point_cloud_inner = -2*point_cloud_inner
#     point_cloud_square = torch.sum( point_cloud**2, dim=-1, keepdim = True)
#     point_cloud_square_tranpose = point_cloud_square.permute(0, 3, 2, 1) #torch.transpose(point_cloud_square, perm=[0, 2, 1])
#     return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


#Example 
# kernel = 2
# indim = 1
# #st()
# data = torch.rand(60)
# data = data.view(2, 6, 5, 1) #This is ---> B*d^3*N*1// B*N*D*C
# # u = nn.Unfold(kernel_size = (kernel, 1))
# a = pairwise_distance(data)
# #print(a)
# # u = u(data).view(data.shape[0], data.shape[1], kernel, -1, indim) #This is B*d^3*S*N'*1
# # u = torch.transpose(u, 2, 4)
# # u = torch.transpose(u, 3, 2)
# # print(u.shape)

# weights = torch.rand(24)
# weights = weights.view(1, 3, 8) #m*k*n
# # print("data is of shape "+str(data))
# # print("weights is of shape "+str(weights.shape))
# a = wFM_on_sphere(data, weights)
# print(a)

