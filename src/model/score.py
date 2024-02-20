import math 

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
#from torch_cluster import radius_graph
from torch_scatter import scatter
import numpy as np 
from e3nn.nn import BatchNorm
from scipy.spatial import KDTree
from torch import Tensor
from joblib import Parallel, delayed

#The following class implements a Tensor Product Convolutional layer 
class TensorProductConvLayer(nn.Module):
    def __init__(self, 
                 in_irreps, 
                 sh_irreps, 
                 out_irreps, 
                 n_edge_features, 
                 residual: bool=True,
                 batch_norm: bool=True):
        super(TensorProductConvLayer, self).__init__()
        
        self.in_irreps = in_irreps,
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        
        self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)
        
        # n_edge_features is the number of features per edge in the subgraph (in the classic example would be the embedding of the radial basis representation of the distance)
        #the following layer is used to transform the edge features to the correct dimension for the tensor product layer
        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Linear(n_edge_features, self.tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None
        
    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce="mean"):
        
        edge_src, edge_dst = edge_index
        #tensor product weights are shared across all edges
        weigths = self.fc(edge_attr) # n_edges x weight_numel
        
        #The node attributes are lifted up to the edge representation
        #The lifted node attributes are then transformed by the tensor product layer only on the paths allowed by the spherical harmonics
        summand = self.tp(node_attr[edge_src], edge_sh, weigths)
        
        out_nodes = out_nodes or node_attr.shape[0]
        out_tp = scatter(summand, edge_dst, dim=0, dim_size=out_nodes, reduce=reduce) # n_edges x out_dims 
        if self.residual:
            padded = F.pad(node_attr, (0, out_tp.shape[-1] - node_attr.shape[-1]))
            out = out_tp + padded

        if self.batch_norm:
            out = self.batch_norm(out)      

        return out
    
    
#The following class implements a tensor product score model for predicting the score of atom displacements with respect to a coarse-grained representation
# The score is parametrized with the output parity equivariant node features of order l=1 
class TensorProductScoreModel(nn.Module):
    def __init__(self, in_node_features: int=74, in_edge_features: int=4, sigma_embed_dim: int=32, sigma_min: float=0.01 * np.pi,
                 sigma_max: float=np.pi, sh_lmax: int=2, ns: int=32, nv: int=8, num_conv_layers: int=4, max_radius=5, radius_embed_dim: int=50,
                 scale_by_sigma: bool=True, second_order_repr: bool=True, batch_norm: bool=True , residual: bool=True
                 ): 
        super(TensorProductScoreModel, self).__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.max_radius = max_radius
        self.radius_embed_dim = radius_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        
        # The following MLP is used to embed the node features in higher dimensional space with parity invariant features of order l=0
        self.node_embedding = nn.Sequential(
            nn.Linear(in_node_features + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(in_edge_features + radius_embed_dim, nv),
            nn.ReLU(),
            nn.Linear(nv, nv)
        )
        self.distance_expansion = GaussianSmearing(0.0, max_radius, radius_embed_dim)
        conv_layers  = []
        
        if second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e+ {nv}x1o +  {nv}x2e', 
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]
        
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i+1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                residual=residual,
                batch_norm=batch_norm
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)
    def forward(self, data):
        
        node_attr, edge_index, edge_attr, edge_sh = self.build_conv_graph(data)
        src, dst = edge_index
        
        node_attr = self.node_embedding(node_attr)
        edge_attr = self.edge_embedding(edge_attr)
        
        for layer in self.conv_layers:
            edge_attr_ = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], dim=-1)
            node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, reduce="mean")
            
        return node_attr, edge_index
            
        
    def build_conv_graph(self, data):
        
        #radius_edges = radius_graph(data.pos, r=self.max_radius, batch=data.batch) #computes the radius graph of the input graph outputting the edge index of the radius graph
        radius_edges = pbc_radius_graph(data.pos, r=self.max_radius, box_size=data.boxsize, batch=data.batch)
        edge_index = torch.cat([data.edge_index, radius_edges], dim=1).long() #concatenates the edges from the original graph of connected components and the radius graph
        edge_attr = torch.cat([
            data.edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_edge_features, device=data.x.device)
        ], dim=0)
        
        node_sigma = torch.log(data.node_sigma / self.sigma_min) / torch.log(self.sigma_max / self.sigma_min) * 10000 #normalizes the node sigma to the range [0, 1]
        node_sigma_emb = get_timestep_embedding(node_sigma, self.sigma_embed_dim) #embeds the node sigma in a higher dimensional space
        
        edge_sigma_emb = node_sigma_emb[edge_index[0].long()] #embeds the edge sigma in a higher dimensional space
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], dim=1)
        node_attr = torch.cat([data.x, node_sigma_emb], dim=1)
        
        src, dst = edge_index # source and destination nodes of the radius graph + the original graph
        edge_vec = data.pos[dst.long()] - data.pos[src.long()] # relative distance between the source and destination nodes of the radius graph + the original graph
        edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1)) # edge length embedding using a gaussian smearing function
        
        edge_attr = torch.cat([edge_attr, edge_length_emb], dim=1) # concatenates the edge attributes with the edge length embedding
        
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh



class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
         
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

if __name__ == "__main__":
    
    from model.pbc_radius_graph import gen_graph
    import torch
    device = "cuda:0"
    n_atoms = 10000 # Number of atoms
    m = 100 # Number of linear chains
    n_batch = 10 # Number of batches
    batched_conf = gen_graph(n_atoms, m, n_batch=n_batch, device=device)
    model = TensorProductScoreModel()
        
        
        
        
        
        
                  
        
        

