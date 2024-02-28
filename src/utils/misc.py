from scipy.spatial import KDTree
import torch
from torch import Tensor
import numpy as np
import math
from torch.nn import functional as F

# The following function computes the radius graph of the input nodes positions using KDTrees and the box size
def pbc_radius_graph(pos: Tensor, r: float, box_size: Tensor, batch=None):
    # KDTree generation for batched positions input
    edge_index = []
    
    for batch_idx in range(box_size.shape[0]):
        kdtree = KDTree(pos[batch == batch_idx].cpu().numpy(), boxsize=box_size[batch_idx].cpu().numpy())
        pairs = kdtree.query_pairs(r, output_type='ndarray').swapaxes(0,1)
        edge_index.append(pairs)
    edge_index = np.array(edge_index).swapaxes(0,1).reshape(2, -1)
    return torch.tensor(edge_index, dtype=torch.long).contiguous()


# Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py 
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb   


def sigmoid_scheduler(beta_0:float, beta_T:float, k: int=10, N: int=1000):
    """
    Sigmoid scheduler function for beta(t).

    Parameters:
        T (int): Total number of steps.
        beta_0 (float): Initial value of beta.
        beta_T (float): Final value of beta.

    Returns:
        list: Scheduled values of beta from time step 0 to T-1.
    """
    # Sigmoid function parameters
    x = torch.arange(N, dtype=torch.float32)
    x0 = 0.5 * N  # Midpoint of the sigmoid
   
    return beta_0 + (beta_T - beta_0) / (1 + torch.exp(-k * (x - x0) / N)), x


