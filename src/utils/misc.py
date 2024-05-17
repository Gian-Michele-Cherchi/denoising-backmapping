from scipy.spatial import KDTree
import torch
from torch import Tensor
import numpy as np
import math
from torch.nn import functional as F

# The following function computes the radius graph of the input nodes positions using KDTrees and the box size
def pbc_radius_graph(perb_pos: Tensor, r: float, d: Tensor, batch=None):
    # KDTree generation for batched positions input
   
    edge_index = torch.tensor([[],[]]).to(d)
    for batch_idx in range(d.shape[0]):
        
        perb_pos[batch == batch_idx] -= torch.floor(perb_pos[batch == batch_idx] / d[batch_idx]) * d[batch_idx]
        kdtree = KDTree(perb_pos[batch == batch_idx].cpu().numpy(), boxsize=d[batch_idx].cpu().numpy())
        pairs = np.array(kdtree.query_pairs(r, output_type='ndarray').swapaxes(0,1)) + batch_idx * perb_pos[batch == batch_idx].size(0)
        
        edge_index = torch.concatenate([edge_index, 
                                        torch.tensor(pairs.copy()).to(d), 
                                        torch.tensor(np.flip(pairs, axis=0).copy()).to(d)], axis=1)
        del kdtree
        del pairs
    return edge_index


# Function below is ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
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



def write_lammps_traj(filename, positions, atom_type='C'):
    n_atoms = positions.shape[0]
    with open(filename, 'w') as f:
        for i in range(n_atoms):
            f.write('ITEM: TIMESTEP\n')
            f.write(f'{i}\n')
            f.write('ITEM: NUMBER OF ATOMS\n')
            f.write(f'{n_atoms}\n')
            f.write('ITEM: BOX BOUNDS pp pp pp\n')
            f.write('0.0 1.0\n'*3)
            f.write('ITEM: ATOMS id type x y z\n')
            for j in range(n_atoms):
                x, y, z = positions[j]
                f.write(f'{j+1} {atom_type} {x} {y} {z}\n')