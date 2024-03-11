import torch
import numpy as np
from src.model.score_model import pbc_radius_graph
from torch_geometric.data import Data
from torch_geometric.data import Batch
import time 

def gen_graph(n_nodes: int, n_mols: int, node_features_dim: int, edge_features_dim: int, n_batch: int = 1, device: str="cpu"):

    # Generate random node positions and features
    node_positions = torch.rand(n_nodes, 3) # 2 for 3D positions
    node_features = torch.zeros(node_features_dim).unsqueeze(0).repeat(n_nodes, 1).to(device)
    mol_id = torch.tensor(n_mols).to(device)
    tot_mass = torch.tensor(1, dtype=torch.float64) # total monomer mass 
    atom_monomer_id = torch.tensor(n_nodes).to(device) # [n_atoms x 1], values from 0 to n_monomers -1 
    
    #n_atoms_monomer = self.n_monomers*( atom_monomer_id == 0 ).sum().item() # number of atoms per polymer
    node_features[:, 0] = 1
    node_features[:, 2] = 1
    node_features[:, 6] = 1
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
        edge_features = torch.Tensor(edge_index.size(1), edge_features_dim).uniform_(0,1.5).to(device)
        graph = Data(pos=node_positions, x=node_features, edge_index=edge_index, edge_attr=edge_features).to(device)
        graph.box_size = box_size
        timesteps = torch.rand(n_nodes).to(device)
        graph.node_sigma = timesteps
        # Create Data object
        data_list.append(graph)
        
    return Batch.from_data_list(data_list)
    
    
if __name__ == "__main__":
    n_atoms = 10000
    n_batch = 16
    n_mols = 100
    r = 0.1
    device = "cpu"
    batched_confs = gen_graph(n_atoms, n_mols, n_batch, device=device)
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
