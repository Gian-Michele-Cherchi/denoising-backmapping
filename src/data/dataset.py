import os.path as osp
import os 
from multiprocessing import Pool
from collections import Counter
from rdkit import Chem
import numpy as np 
import torch
#from tqdm import tqdm 
import copy
import pandas as pd
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from data.featurization import featurize_pol_from_smiles
TYPES = {'H': 0, 'C': 1}
ATOMIC_NUMBERS = {'H': 1, 'C': 6}
__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper

def get_dataset(name: str, path: str,  **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](path=path, **kwargs)

@register_dataset('polymer_melt')
class PolymerMeltDataset(Dataset):
    def __init__(self,  path , mode, coarse_grain: bool=True ,  save_ckpt:bool=True, device: str='cpu'):
        super(PolymerMeltDataset, self).__init__()
        self.device = device
        self.cg = coarse_grain
        self.path = path
        try: 
            self.datapoints = torch.load(path+'/datapoints.pt', map_location=device)
        except:
            self.datapoints = self.preprocessing(self.path)
            if save_ckpt:
                torch.save(self.datapoints, path+'/datapoints.pt')
        self.n_conf = self.len()
        self.datapoints = list(self.datapoints.values())
        if mode == 'train':
            self.datapoints = self.datapoints[:int(0.1*self.n_conf)]
        elif mode == 'val':
            self.datapoints = self.datapoints[int(0.6*self.n_conf):int(0.8*self.n_conf)]
        elif mode == 'test':
            self.datapoints = self.datapoints[int(0.8*self.n_conf):]
            
    def preprocessing(self, path):
        """
        Preprocess the data
        """
        listdir = os.listdir(path)
        types = []
        smiles = []
        dataset = []
        residue = "C\C=C/C"
        for ds in listdir:
            data = self.read_lammpstraj(osp.join(path, ds, 'dump.lammpstrj'))
            dataset.append(data)
            types.append(data['types'])
            smiles.append(r"C\C=C/C" * self.n_monomers)
        
        self.smiles = set(smiles)
        self.types = set(types[0])
        dataset = self.coarse_grain(dataset[0])
        confs = self.features_from_smiles(dataset)
        return confs
    
    
    def features_from_smiles(self, dataset):
        
        self.mol_features= [featurize_pol_from_smiles(smile, types=self.types, hydrogens=True) for smile in self.smiles][0]
        n_edges = self.mol_features[0].edge_index.size(1)
        match_order = dataset["atom_types"][:self.atom_in_mol]
        indexes_res = [i for i, x in enumerate(match_order) if x != 'H']
        indexes_hyd = [i for i, x in enumerate(match_order) if x == 'H']
        self.reindex = indexes_res + indexes_hyd
        ext_edge_index = self.mol_features[0].edge_index.repeat(1,self.n_molecules)
        const = 0 
        assert len(self.reindex) == self.atom_in_mol
        for i in range(self.n_molecules):
            dataset["pos"][:,i*self.atom_in_mol:(i+1)*self.atom_in_mol] = dataset["pos"][:,i*self.atom_in_mol:(i+1)*self.atom_in_mol][:,self.reindex]
            dataset["mol"][i*self.atom_in_mol:(i+1)*self.atom_in_mol] = dataset["mol"][i*self.atom_in_mol:(i+1)*self.atom_in_mol][self.reindex]
            dataset["atom_types"][i*self.atom_in_mol:(i+1)*self.atom_in_mol] = dataset["atom_types"][i*self.atom_in_mol:(i+1)*self.atom_in_mol][self.reindex]
            dataset["cg_dist"][:,i*self.atom_in_mol:(i+1)*self.atom_in_mol] = dataset["cg_dist"][:,i*self.atom_in_mol:(i+1)*self.atom_in_mol][:,self.reindex]
            #self.com_matrix[i*self.n_monomers:(i+1)*self.n_monomers] = self.com_matrix[i*self.n_monomers:(i+1)*self.n_monomers][:,self.reindex]
            ext_edge_index[:,i*n_edges:(i+1)*n_edges] = const + ext_edge_index[:,i*n_edges:(i+1)*n_edges]
            const += self.atom_in_mol
        graph_inst = {}
        for index,conf in enumerate(dataset["pos"]):
            graph_conf = Data(x=self.mol_features[0].x.repeat(self.n_molecules, 1),
                              edge_index=ext_edge_index, 
                              edge_attr=self.mol_features[0].edge_attr.repeat(self.n_molecules, 1), 
                              z=self.mol_features[0].z.repeat(self.n_molecules))
            graph_conf.conf = conf
            graph_conf.boxsize = dataset["boxsize"][index]
            graph_conf.mol = dataset["mol"]
            graph_conf.cg_dist = dataset["cg_dist"][index]
            graph_conf.cg_pos = dataset["cg_pos"][index]
            graph_inst["conf"+str(index)] = graph_conf
            
        return graph_inst
        
    def read_lammpstraj(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        timestep_indices = [i for i, line in enumerate(lines) if line.startswith('ITEM: TIMESTEP')]
        n_atoms_indices = [i for i, line in enumerate(lines) if line.startswith('ITEM: NUMBER OF ATOMS')]
        boxsize_indexes = [i for i, line in enumerate(lines) if line.startswith('ITEM: BOX BOUNDS pp pp pp')]
        boxsize_x = torch.tensor([ [float(lines[boxsize_indexes[i]+1].strip().split(' ')[0]),
                       float(lines[boxsize_indexes[i]+1].strip().split(' ')[1])] for i in range(len(boxsize_indexes))])
        boxsize_y = torch.tensor([[float(lines[boxsize_indexes[i]+2].strip().split(' ')[0]),
                      float(lines[boxsize_indexes[i]+1].strip().split(' ')[1])] for i in range(len(boxsize_indexes))])
        boxsize_z = torch.tensor([[float(lines[boxsize_indexes[i]+3].strip().split(' ')[0]),
                      float(lines[boxsize_indexes[i]+1].strip().split(' ')[1])] for i in range(len(boxsize_indexes))])
        boxsize = torch.cat([boxsize_x[[...,None]], boxsize_y[...,None], boxsize_z[...,None]], dim=-1)
        self.n_atoms = int(lines[n_atoms_indices[0]+1].strip().split('\n')[0])
        header = lines[timestep_indices[0]+8].strip().split(' ')[2:]
        skip_index = np.array([list(range(i,i+9)) for i in timestep_indices]).reshape(-1)
        df = pd.read_csv(filename, skiprows=skip_index, sep=' ', names=header)
        self.n_monomers = df.mol.unique().max()
        types  = list(df.element.unique())
        n_conf = len(df) // self.n_atoms
        pos = torch.Tensor(df[['xu', 'yu', 'zu']].values)[None,...].view(n_conf,self.n_atoms, 3)
        mol = torch.Tensor(df["mol"].values)[None,...].view(n_conf,self.n_atoms)[0]
        atom_type = df['element'].values[:skip_index[9]-9]
        self.atom_in_mol = 102
        self.n_molecules = pos.size(1) // self.atom_in_mol
        assert len(atom_type) == mol.size(0)
        del df
            
        return {"pos": pos,
            "mol":mol,
            "types":types,
            "atom_types":atom_type,
            "boxsize":boxsize,
            }
        
    def coarse_grain(self, dataset):
        atom_monomer_id = dataset["mol"]# [n_atoms x 1], values from 1 to n_monomers 
        self.com_matrix = torch.zeros(self.n_molecules * self.n_monomers, self.n_atoms, device=self.device)
        partial_relative_mass = torch.tensor([ATOMIC_NUMBERS[dataset["atom_types"][i]] for i in range(len(dataset["atom_types"]))])
        for i in range(self.n_monomers*self.n_molecules):
            mon_id = i % self.n_monomers
            pol_id = i // self.n_monomers
            index = atom_monomer_id == mon_id+1
            select_index = torch.arange(self.n_molecules) != pol_id
            index = index.reshape(self.n_molecules, self.atom_in_mol)
            index[select_index] = False
            index = index.reshape(self.n_molecules* self.atom_in_mol)
            total_monomer_mass = torch.tensor(list(Counter(dataset["atom_types"][index]).values())) @ torch.tensor([6,1])
            self.com_matrix[i,:] = (partial_relative_mass / total_monomer_mass) * index 

        com_confs = torch.bmm(self.com_matrix[None,...].expand(dataset["pos"].size(0),-1,-1), dataset["pos"])
        dataset["cg_pos"] = com_confs
        dist = torch.zeros(dataset["pos"].size(0), self.n_molecules*self.atom_in_mol,3)
        ################################
        for i in range(self.n_monomers*self.n_molecules):
            mon_id = i % self.n_monomers
            pol_id = i // self.n_monomers
            index = atom_monomer_id == mon_id+1
            select_index = torch.arange(self.n_molecules) != pol_id
            index = index.reshape(self.n_molecules, self.atom_in_mol)
            index[select_index] = False
            index = index.reshape(self.n_molecules* self.atom_in_mol)
            dist[:,index] = dataset["pos"][:,index] -dataset["cg_pos"][:,i:i+1].repeat(1,index.sum().item(),1)
        dataset["cg_dist"] = dist
        return dataset
    
    def reverse_coarse_grain(self,batch, batch_size: int):
        """
        Reverse the coarse graining
        """
        n_atom_in_mol = 102
        n_atoms = batch.conf.size(0) // batch_size
        n_molecules = n_atoms  // n_atom_in_mol
        n_monomers = int(batch.mol.max().item())
        pos = batch.conf.view(batch_size, n_atoms, -1) 
        cg_perb_dist = batch.cg_perb_dist.view(batch_size, n_atoms, -1)
        cg_confs =  batch.cg_pos.view(batch_size, int(n_molecules*n_monomers), -1)
        atom_monomer_id = batch.mol[batch.batch == 0]
        # batch [batch_size, n_atoms, 3]
        batch.perb_pos = pos.clone()
        for i in range(int(n_monomers*n_molecules)):
            mon_id = i % n_monomers
            pol_id = i // n_monomers
            index = atom_monomer_id == mon_id+1
            select_index = torch.arange(n_molecules) != pol_id
            index = index.reshape(n_molecules, n_atom_in_mol)
            index[select_index] = False
            index = index.reshape(int(n_molecules* n_atom_in_mol))
            batch.perb_pos[:,index] = cg_confs[:,i:i+1].repeat(1,index.sum().item(),1) + cg_perb_dist[:,index]
        batch.perb_pos = batch.perb_pos.reshape(-1,3)
        return batch
    
    
    def get_com_matrix(self):
        return self.com_matrix
    
    def get(self, idx: int):
        data = self.datapoints[idx]
        return copy.deepcopy(data)
    
    def len(self):
        return len(self.datapoints)
            
    
def get_dataloader(args, batch_size, modes=('train', 'val')):
    """
    Get the dataloader for the dataset
    """
    
    if isinstance(modes, str):
        modes = [modes]
        
    loaders = {}
    for mode in modes:
        dataset = PolymerMeltDataset(**args)
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False if mode == 'test' else True)
        loaders[mode] = loader
    return loaders, dataset

    
if __name__ == "__main__":
    
    import argparse 
    args = {"dataset_path":"dataset", "num_workers":4, "batch_size":32, "device": 'cuda:0'}
    args = argparse.Namespace(**args)
    loaders, dataset = get_dataloader(args, batch_size=args.batch_size)
    for batch in loaders["train"]:
        print(batch.to(args.device))
        batch_cg = dataset.coarse_grain(batch.to(args.device))
        break
    
    