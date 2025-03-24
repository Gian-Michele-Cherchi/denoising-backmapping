# In this script we are going to define the featurization process for the molecules dataset.
# We are going to use the RDKit library to generate the molecular descriptors.
import numpy as np 
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType as CT
from collections import Counter
import torch
import torch.nn.functional as F
from torch_scatter import scatter 
from torch_geometric.data import Data, Dataset, DataLoader


bonds = {BT.SINGLE: 0, BT.DOUBLE: 1,"NONBONDED": 2}
TYPES = {
    'H': 0,
    'O': 1,
    'Cl': 2,
    'F': 3,
    'N': 4,
    'C': 5
}



# The following function performes a one-hot encoding of the atom type
def one_k_encoding(value, choices):
    # choices are the possible values that the atom type can take
    encoding = [0] * (len(choices) +1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    
    return encoding
    
# The following function featurize the polymer 
def featurize_polymer(polymer):
    """
    Part of the featurisation code taken from GeoMol https://github.com/PattanaikL/GeoMol
    Returns:
        x:  node features
        z: atomic numbers of the nodes (the symbol one hot is included in x)
        edge_index: [2, E] tensor of node indices forming edges
        edge_attr: edge features
    """
    #types = TYPES[key] for key in types
    #types = {k: TYPES[k] for k in types if k in TYPES}

    N = polymer.GetNumAtoms()
    #dehydr_polymer = Chem.RemoveHs(polymer)
    #N_carbons = dehydr_polymer.GetNumAtoms()
    #monomer_type = {'mon'+str(i): i for i in range(1, 101)}
    atom_type_idx = []
    atomic_number = []
    atom_features = []
    atom_symbols = [atom.GetSymbol() for atom in polymer.GetAtoms()]

    for i, atom in enumerate(polymer.GetAtoms()):
        atom_type = atom.GetSymbol()
        atom_type_idx.append(TYPES[atom_type])
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
    
    
    x1 = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(TYPES))
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)
    #x = x1

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z=z), len(atom_type_idx), atom_symbols


def featurize_pol_from_smiles(smiles_rep: str, hydrogens: bool = False):
    polymer = Chem.MolFromSmiles(smiles_rep)
    if hydrogens:
        polymer = Chem.AddHs(polymer)
    
    return featurize_polymer(polymer)



def load_lammps_edge_index(dump_data, file_path, hydrogens=False):
    
    time_index = dump_data["time_index"]
    edge_index = []
    in_angles_section = False

    with open(file_path, 'r') as file:
        
        count = 0
        for line in file:
            
            if "Bonds\n" in line:
                in_angles_section = True
                continue
            
            if in_angles_section:
                
                if line.strip() == "":
                    count += 1
                    continue
                    
                if count == 2:
                    break
                
                elif count==1 and line.strip() != "":
                    
                    tmp = [int(x) for x in line.strip().split()][2:]
                    if hydrogens:
                         edge_index.append(tmp)
                    else:
                        
                        atom1_id = dump_data['element'][time_index][dump_data['id'][time_index]==tmp[0]]
                        atom2_id = dump_data['element'][time_index][dump_data['id'][time_index]==tmp[1]]
                        
                        if atom1_id != 'H' and atom2_id != 'H':
                            edge_index.append(tmp)
            
            
    return torch.tensor(edge_index, dtype=torch.int32).swapaxes(1,0)




def lammps_rdkit_matching(dump_data, chemfeatures, lmp_filepath, n_monomers, hydrogens):
    
    # The simulation is non-reactive, therefore the CHARMM-GUI generated atom order is the same throughtout the simulation
    # We can use the atom order to match the atoms in the LAMMPS dump with the atoms in the rdkit library
    
    time_index = dump_data["time_index"]
    edge_index_key = 'edge_index_nohydr' if not hydrogens else 'edge_index'
    
    mol_indexes, counts =  torch.unique(dump_data["mol"][0], return_counts=True)
    monomer_atoms_list1 = chemfeatures[0][2][:chemfeatures[0][1]//n_monomers]
    monomer_atoms_list2 = chemfeatures[1][2][:chemfeatures[1][1]//n_monomers]
    
    ref_atomcounter1 = Counter(monomer_atoms_list1)
    ref_atomcounter2 = Counter(monomer_atoms_list2)
    
    dump_data[edge_index_key] = load_lammps_edge_index(dump_data, 
                                                        lmp_filepath, 
                                                        hydrogens=hydrogens)
    
    for mol in mol_indexes:
        
        atoms_in_mol = dump_data["element"][time_index][dump_data["mol"][time_index] == mol] #the order of atoms is the same for all the configurations in LAMMPS
        atoms_in_mol = atoms_in_mol[atoms_in_mol != 'H'] if not hydrogens else atoms_in_mol
        lammps_mol_counter = Counter(atoms_in_mol)
        
        flags = [lammps_mol_counter == ref_atomcounter1, 
                 lammps_mol_counter == ref_atomcounter2]
        
        assert flags[0] or flags[1], "The atoms in the LAMMPS dump do not match the atoms in the polymer"
        
        sel_comp_index = np.argmax(flags)
        
        edge_index = chemfeatures[sel_comp_index][0].edge_index
        
        ids_mol =  dump_data["id"][time_index][dump_data["mol"][time_index] == mol]
        
        mol_index_sel = [(dump_data[edge_index_key] == id.item())[None, ...] for id in ids_mol]
        
        mol_index_sel = torch.cat(mol_index_sel, dim=0).sum(dim=(0,1)).bool()
        
        
        index_map = {id.item(): n.item() for id,n in zip(ids_mol, torch.arange(len(ids_mol)))}
        
        
        
        
        

        


if __name__ == "__main__":
    # Test the featurization process
    from rdkit import Chem
    polymer_smiles = r"C\C=C/C" * 10
    pol_data = featurize_pol_from_smiles(polymer_smiles, hydrogens=True)
    