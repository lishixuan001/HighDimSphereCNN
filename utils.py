import numpy as np
import torch
from pdb import set_trace as st

def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.shape[0] #point_cloud.get_shape().as_list()[0]
    point_cloud = torch.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = point_cloud.unsqueeze(0) #torch.expand_dims(point_cloud, 0)
    
    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    #torch.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2*point_cloud_inner
    point_cloud_square = torch.sum( point_cloud**2, dim=-1, keepdim = True)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1) #torch.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int
    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    nn_idx = np.argsort(adj_matrix, axis=-1)[:,:,1:k+1] #torch.nn.top_k(neg_adj, k=k)
    return nn_idx