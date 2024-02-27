# In this script we are going to define the featurization process for the molecules dataset.
# We are going to use the RDKit library to generate the molecular descriptors.
import numpy as np 
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType as CT

import torch
import torch.nn.functional as F
from torch_scatter import scatter 
from torch_geometric.data import Data, Dataset, DataLoader


bonds = {BT.SINGLE: 0, BT.DOUBLE: 1,"NONBONDED": 2}
atom_types = {'H': 0, 'C': 1}

# The following function performes a one-hot encoding of the atom type
def one_k_encoding(value, choices):
    # choices are the possible values that the atom type can take
    encoding = [0] * (len(choices) +1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    
    return encoding
    
# The following function featurize the polymer 
def featurize_polymer(polymer, types=atom_types):
    """
    Part of the featurisation code taken from GeoMol https://github.com/PattanaikL/GeoMol
    Returns:
        x:  node features
        z: atomic numbers of the nodes (the symbol one hot is included in x)
        edge_index: [2, E] tensor of node indices forming edges
        edge_attr: edge features
    """
    N = polymer.GetNumAtoms()
    #dehydr_polymer = Chem.RemoveHs(polymer)
    #N_carbons = dehydr_polymer.GetNumAtoms()
    #monomer_type = {'mon'+str(i): i for i in range(1, 101)}
    atom_type_idx = []
    atomic_number = []
    atom_features = []
    for i, atom in enumerate(polymer.GetAtoms()):
        atom_type = atom.GetSymbol()
        atom_type_idx.append(types[atom_type])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3]))
        
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,]))
        #atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
    
    z = torch.tensor(atomic_number, dtype=torch.long)
    
    row, col, edge_type = [], [], []
    for bond in polymer.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        #bond_type += 2 * [bond_nonbond["BONDED"]]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
    
    
    x1 = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)
    #x = x1

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z=z)


def featurize_pol_from_smiles(smiles_rep: str, hydrogens: bool = False):
    polymer = Chem.MolFromSmiles(smiles_rep)
    if hydrogens:
        polymer = Chem.AddHs(polymer)
    
    return featurize_polymer(polymer)



if __name__ == "__main__":
    # Test the featurization process
    from rdkit import Chem
    polymer_smiles = r"C\C=C/C" * 10
    pol_data = featurize_pol_from_smiles(polymer_smiles, hydrogens=True)
    