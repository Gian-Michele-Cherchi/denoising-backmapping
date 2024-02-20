import torch
import numpy as np
from score import pbc_radius_graph
from torch_geometric.data import Data
from torch_geometric.data import Batch
import time 
def gen_graph(n_nodes: int, n_mols: int, n_batch: int = 1, device: str="cpu"):
    node_features_dim = 50
    edge_features_dim = 4

    # Generate random node positions and features
    node_positions = torch.rand(n_nodes, 3) # 2 for 3D positions
    node_features = torch.randn(n_nodes, node_features_dim)
    data_list = []
    box_size = 1 
    for _ in range(n_batch):
    # Generate edge index for m linear chains
        edge_index = []
        atoms_per_chain = n_nodes // n_mols
        for i in range(n_mols):
            start = i * atoms_per_chain
            end = start + atoms_per_chain
            chain_edge_index = torch.tensor([[i, i+1] for i in range(start, end-1)], dtype=torch.long).t().contiguous()
            edge_index.append(chain_edge_index)
        edge_index = torch.cat(edge_index, dim=1)

        # Generate random edge features
        edge_features = torch.randn(edge_index.size(1), edge_features_dim).to(device)
        graph = Data(pos=node_positions, x=node_features, edge_index=edge_index, edge_attr=edge_features).to(device)
        graph.box_size = box_size
        # Create Data object
        data_list.append(graph)
        
    return Batch.from_data_list(data_list)
    
    

# Test case 1: Single batch, single point
n_atoms = 10000
n_batch = 16
n_mols = 100
r = 0.1
batched_confs = gen_graph(n_atoms, n_mols, n_batch, device="cuda")
start = time.time()
edges_index = pbc_radius_graph(batched_confs.pos, r, batched_confs.box_size, batched_confs.batch)
end = time.time()
wall_time = end - start
dx = batched_confs.pos[edges_index[0]] - batched_confs.pos[edges_index[1]]
dx -= torch.round(dx / batched_confs.box_size[0]) * batched_confs.box_size[0]
dr = torch.linalg.norm(dx, dim=-1)
assert torch.all(dr <= r), "Test case 1 failed"

print(f"Wall time: {wall_time} seconds")
print("All test cases passed!")
