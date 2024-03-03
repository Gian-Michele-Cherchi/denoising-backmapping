from .pbc_radius_graph import gen_graph
import torch 
from sampling.measurements import CoarseGrainOperator


def test_coarse_grain_operator():
    n_atoms = 80000
    n_batch = 16
    n_mols = 100
    r = 0.1
    device = "cpu"
    batched_confs = gen_graph(n_atoms, n_mols, n_batch, device=device)
    
    
    cg_op = CoarseGrainOperator(n_atoms=n_atoms, n_monomers=, n_polymers, device)
    