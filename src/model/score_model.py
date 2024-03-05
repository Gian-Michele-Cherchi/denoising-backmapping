import math 

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
#from torch_cluster import radius_graph
from torch_scatter import scatter
import numpy as np 
from e3nn.nn import BatchNorm
from torch import Tensor
from utils.misc import pbc_radius_graph, get_timestep_embedding

N_ATOM_TYPES = 2
N_BOND_TYPES = 3
N_DEGREE_TYPES = 4
N_HYBRIDIZATION_TYPES = 3

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
                 sigma_max: float=np.pi, sh_lmax: int=2, ns: int=32, nv: int=32, num_conv_layers: int=4, max_radius=5, radius_embed_dim: int=50,
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
        
        self.atom_type_embedding = nn.Embedding(N_ATOM_TYPES,2)
        self.bond_type_embedding = nn.Embedding(N_BOND_TYPES,2)
        self.degree_type_embedding = nn.Embedding(N_DEGREE_TYPES,2)
        self.hybridization_type_embedding = nn.Embedding(N_HYBRIDIZATION_TYPES,2)
        
        # The following MLP is used to embed the atom chemical features in higher dimensional space with parity invariant features of order l=0
        self.node_embedding = nn.Sequential(
            nn.Linear(in_node_features + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(in_edge_features + radius_embed_dim + sigma_embed_dim, nv),
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
            irrep_last = f'{ns}x0e + {1}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
        
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]
            irrep_last =  f'{ns}x0e + {1}x1o + {nv}x1e + {ns}x0o'
        
        for i in range(num_conv_layers):
            if i == num_conv_layers - 1:
                in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
                out_irreps = irrep_last
                layer = TensorProductConvLayer(
                    in_irreps=in_irreps,
                    sh_irreps=self.sh_irreps,
                    out_irreps=out_irreps,
                    n_edge_features=3 * ns,
                    residual=residual,
                    batch_norm=batch_norm
                )
            else:
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
            edge_attr_ = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
            node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, reduce="mean")
        
        
        return node_attr, edge_index
            
        
    def build_conv_graph(self, data):
        
        #radius_edges = radius_graph(data.pos, r=self.max_radius, batch=data.batch) #computes the radius graph of the input graph outputting the edge index of the radius graph
        radius_edges = pbc_radius_graph(data.pos, r=self.max_radius, box_size=data.box_size, batch=data.batch)
        
        # Convert to sets of tuples: check for unique edges to add non-bonded features
        edges_cc     = set(map(tuple, data.edge_index.T.cpu().numpy()))
        edges_radius = set(map(tuple, radius_edges.T.cpu().numpy()))
        unique_edges = edges_radius - edges_cc
        non_bonded_edges_index = torch.tensor(list(unique_edges)).T.to(device) # non-bonded set of indexes
        
        non_bonded_attr       = torch.zeros(non_bonded_edges_index.shape[-1], data.edge_attr.shape[-1], device=data.x.device)
        non_bonded_attr[:, 2] = 1 # non-bonded edges are assigned a bond type of 2 
        
        edge_index = torch.cat([data.edge_index, non_bonded_edges_index], dim=1).long() #concatenates the edges from the original graph of connected components and the radius graph
        edge_attr  = torch.cat([data.edge_attr, non_bonded_attr], dim=0)
        assert edge_attr.shape[0] == edge_index.shape[1], "Edge attributes and edge index must have the same number of edges"
        
        #embeddings
        node_attr, edge_attr = self.features_embedding(node_features=data.x, edge_features=edge_attr, edge_index=edge_index)
    
        
        log_sigma_max_min = torch.log(torch.tensor(self.sigma_max / self.sigma_min))
        node_sigma = torch.log(data.node_sigma / self.sigma_min) / log_sigma_max_min * 10000 #normalizes the node sigma to the range [0, 1]
        node_sigma_emb = get_timestep_embedding(node_sigma, self.sigma_embed_dim) #embeds the node sigma in a higher dimensional space
        
        
        edge_sigma_emb = node_sigma_emb[edge_index[0].long()] #embeds the edge sigma in a higher dimensional space
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], dim=1)
        node_attr = torch.cat([node_attr, node_sigma_emb], dim=1)
        
        src, dst = edge_index # source and destination nodes of the radius graph + the original graph
        edge_vec = data.pos[dst.long()] - data.pos[src.long()] # relative distance between the source and destination nodes of the radius graph + the original graph
        edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1)) # edge length embedding using a gaussian smearing function
        
        edge_attr = torch.cat([edge_attr, edge_length_emb], dim=1) # concatenates the edge attributes with the edge length embedding
        
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh

    def features_embedding(self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor):
        
        atom_type_feat = node_features[:,:N_ATOM_TYPES].long()
        degree_type_feat = node_features[:,N_ATOM_TYPES:N_ATOM_TYPES + N_DEGREE_TYPES].long()
        hybr_type_feat = node_features[:,N_ATOM_TYPES + N_DEGREE_TYPES:N_ATOM_TYPES + N_DEGREE_TYPES + N_HYBRIDIZATION_TYPES].long()
        
        # node embeddings 
        atom_type_emb = self.atom_type_embedding(atom_type_feat.argmax(dim=1))
        degree_type_emb = self.degree_type_embedding(degree_type_feat.argmax(dim=1))
        hybr_type_emb = self.hybridization_type_embedding(hybr_type_feat.argmax(dim=1))
        
        bond_type_emb = self.bond_type_embedding(edge_features.argmax(dim=1))
        
        node_attr = torch.cat([atom_type_emb, degree_type_emb, hybr_type_emb], dim=1).to(node_features)
        
        lifted_node_attr = node_attr[edge_index].permute(1,2,0)
        lifted_node_attr = lifted_node_attr[:,:, 0] + lifted_node_attr[:,:, 1] 
        
        edge_attr = torch.cat([lifted_node_attr, bond_type_emb], dim=1).to(node_features)
        
        return node_attr, edge_attr
    

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
           
def create_model(
    in_node_features,
    in_edge_features,
    sigma_embed_dim,
    sh_lmax,
    ns,
    nv,
    num_conv_layers,
    max_radius,
    radius_embed_dim,
    scale_by_sigma,
    second_order_repr,
    batch_norm, 
    residual,
    model_path="",
    ):

    model= TensorProductScoreModel(in_node_features=in_node_features, in_edge_features=in_edge_features, sigma_embed_dim=sigma_embed_dim, 
                                    sh_lmax=sh_lmax, ns=ns, nv=nv, num_conv_layers=num_conv_layers, max_radius=max_radius, 
                                    radius_embed_dim=radius_embed_dim,
                                    second_order_repr=second_order_repr, batch_norm=batch_norm, residual=residual, scale_by_sigma=scale_by_sigma)

    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        print(f"Got exception: {e} / Randomly initialize")
    return model







if __name__ == "__main__":
    
    from tests.pbc_radius_graph import gen_graph
    import torch
    import time 
    device = "cuda:0"
    n_atoms = 10000 # Number of atoms
    m = 100 # Number of linear chains
    n_batch = 10 # Number of batches
    node_features_dim = 6
    edge_features_dim = 8
    max_radius = 0.02
    batched_conf = gen_graph(n_atoms, m, n_batch=n_batch, node_features_dim=9, edge_features_dim=3,  device=device)
    score_model = TensorProductScoreModel(in_node_features=node_features_dim, in_edge_features=edge_features_dim, sigma_embed_dim=32, 
                                    sh_lmax=2, ns=32, nv=32, num_conv_layers=4, max_radius=max_radius, radius_embed_dim=50).to(device)
    
    start = time.time()
    node_attr, edge_index = score_model(batched_conf)
    end = time.time()
    print(f"Wall time: {end - start} seconds")    
        
        
        
        
        
                  
        
        

