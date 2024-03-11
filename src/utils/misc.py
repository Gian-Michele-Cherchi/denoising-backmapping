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
        pos[batch == batch_idx] -= torch.floor(pos[batch == batch_idx] / box_size[batch_idx]) * box_size[batch_idx]
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

def get_beta_scheduler(beta_start:float, beta_end:float, diffusion_timesteps: int):
    
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    
    betas = torch.linspace(-10, 10, steps=diffusion_timesteps)
    betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    assert betas.shape == (diffusion_timesteps,)
    return betas




