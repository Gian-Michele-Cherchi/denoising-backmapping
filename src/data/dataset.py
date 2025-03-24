import os.path as osp
import os
import random
from collections import Counter
import numpy as np 
import torch
import copy
import pandas as pd
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from featurization import featurize_pol_from_smiles

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
    def __init__(self,  path , mode, coarse_grain: bool=True , hydrogens: bool=False,  save_ckpt:bool=True, device: str='cpu', override: bool=False):
        super(PolymerMeltDataset, self).__init__()
        self.device = device
        self.hydrogens = hydrogens
        self.cg = coarse_grain
        self.path = path
        if override:
            self.datapoints = self.preprocessing(self.path)
        else:
            try: 
                self.datapoints = torch.load(path+'/datapoints_cg_graph.pt', map_location=device)
            except:
                self.datapoints = self.preprocessing(self.path)
                if save_ckpt:
                    torch.save(self.datapoints, path+'/datapoints_cg_graph.pt')
        self.n_conf = self.len()
        self.datapoints = list(self.datapoints.values())
        random.shuffle(self.datapoints)
        if mode == 'train':
            self.datapoints = self.datapoints[:int(0.05*self.n_conf)]
        elif mode == 'val':
            self.com_matrix, self.atom_monomer_id = self.get_com_matrix(self.datapoints[0])
            self.datapoints = self.datapoints[int(0.1*self.n_conf):int(0.125*self.n_conf)]
        elif mode == 'test':
            self.com_matrix, self.atom_monomer_id = self.get_com_matrix(self.datapoints[0])
            self.datapoints = self.datapoints[int(0.3*self.n_conf):int(0.4*self.n_conf)]
            
    def preprocessing(self, path):
        """
        Preprocess the data
        """
        #listdir = os.listdir(path)
        types = []
        smiles = []
        dataset = []
        residue = "C\C=C/C"
        
        data = self.read_lammpstraj(osp.join(path, 'dump.lammpstrj'))
        dataset.append(data)
        types.append(data['types'])
        smiles.append(r"C\C=C/C" * self.n_monomers)
        
        self.smiles = set(smiles)
        self.types = set(types[0])
        confs = self.features_from_smiles(dataset[0])
        
        if self.cg:
            for conf in confs.values():
                self.com_matrix, _ = self.get_com_matrix(conf)
                conf.cg_pos = self.com_matrix @ conf.conf
                conf.cg_dist = self.get_cg_dist(conf)
                conf.cg_dist = torch.cat([conf.cg_dist, torch.zeros_like(conf.cg_pos)], dim=0)
                conf.cg_std_dist = torch.sqrt(torch.trace(conf.cg_dist.T @ conf.cg_dist) / (3*conf.cg_dist.size(0)))
                conf.full_conf = torch.cat([conf.conf, conf.cg_pos], dim=0)
                conf.mask = torch.cat([torch.ones_like(conf.conf[:,0]), torch.zeros_like(conf.cg_pos[:,0])], dim=0) 
        return confs
    
    def features_from_smiles(self, dataset):
        
        self.mol_features= [featurize_pol_from_smiles(smile, types=self.types, hydrogens=True) for smile in self.smiles][0]
        n_edges = self.mol_features[0].edge_index.size(1)
        atoms_in_mol  = self.mol_features[0].x.size(0)
        match_order = dataset["atom_types"][:atoms_in_mol]
        indexes_res = [i for i, x in enumerate(match_order) if x != 'H']
        indexes_hyd = [i for i, x in enumerate(match_order) if x == 'H']
        self.reindex = indexes_res + indexes_hyd
        ext_edge_index = self.mol_features[0].edge_index.repeat(1,self.n_molecules)
        const = 0 
        const_cg = 0
        ################################################################################################# REINDEXING
        cg_atom_index = torch.cat([torch.arange(0,40).repeat(self.n_molecules)[None,...], 
                                   torch.arange( self.n_molecules * 40, self.n_molecules * 40 + 10).repeat_interleave(4).repeat(self.n_molecules)[None,...]], dim=0)
        carbon_index = []
        edge_carbon_index = []
        for i in range(self.n_molecules):
            dataset["pos"][:,i*atoms_in_mol:(i+1)*atoms_in_mol] = dataset["pos"][:,i*atoms_in_mol:(i+1)*atoms_in_mol][:,self.reindex]
            dataset["mol"][i*atoms_in_mol:(i+1)*atoms_in_mol] = dataset["mol"][i*atoms_in_mol:(i+1)*atoms_in_mol][self.reindex]
            dataset["atom_types"][i*atoms_in_mol:(i+1)*atoms_in_mol] = dataset["atom_types"][i*atoms_in_mol:(i+1)*atoms_in_mol][self.reindex]
            
            ext_edge_index[:,i*n_edges:(i+1)*n_edges] = const + ext_edge_index[:,i*n_edges:(i+1)*n_edges]
            carbon_index += [i for i in range(i*atoms_in_mol, (i+1)*atoms_in_mol - 62 ) ]
            edge_carbon_index += [i for i in range(i*n_edges, (i+1)*n_edges- 124) ] 
            
            cg_atom_index[0,i * 40: i*40+40] = const + cg_atom_index[0,i*40:i*40+40]
            cg_atom_index[1,i * 40: i*40+40] = const_cg + cg_atom_index[1,i * 40: i*40+40]
            #const += self.atom_in_mol
            const += 40 # just taking carbons
            const_cg += 10
        self.atom_in_mol = 40
        ################################################################################## CG-CG INDEX, CG-ATOM INDEX 
        cg_atom_index = torch.cat([cg_atom_index, cg_atom_index.flip(dims=(0,))], dim=1)
        cg_pol_index = torch.arange(0,2).repeat(self.n_monomers-1)
        cg_pol_index = (cg_pol_index + torch.arange(0,self.n_monomers -1).repeat_interleave(2))
        cg_pol_index = cg_pol_index.view(self.n_monomers - 1,-1).permute(1,0)
        cg_pol_index = const + torch.cat([cg_pol_index, cg_pol_index.flip(dims=(0,))], dim=1)
        ####################################################################################### NODE AND ENDGE FEATURES
        x = self.mol_features[0].x[:self.atom_in_mol].repeat(self.n_molecules, 1)
        edge_attr = self.mol_features[0].edge_attr[:78].repeat(self.n_molecules, 1)
        edge_attr = torch.cat([edge_attr, torch.zeros(edge_attr.size(0),1)], dim=1)
        z = self.mol_features[0].z[:self.atom_in_mol].repeat(self.n_molecules)
        # cg node and edge feature tensors
        cg_x = torch.zeros(self.n_monomers*self.n_molecules, x.size(1))
        cg_x[:,2], cg_x[:,7] = 1, 1
        cg_x[:,0] = 1
        cg_edge_attr = torch.zeros(len(cg_atom_index[0]), edge_attr.size(1))
        cg_edge_attr[:,2] = 1
        cgcg_edge_attr = torch.zeros(len(cg_pol_index[0]), edge_attr.size(1))
        cgcg_edge_attr[:,3] = 1
        graph_inst = {}
        for index,conf in enumerate(dataset["pos"]):
            graph_conf = Data(x=torch.cat([x,cg_x], dim=0),
                              edge_index=ext_edge_index[:,edge_carbon_index],
                              edge_attr=torch.cat([edge_attr, cg_edge_attr, cgcg_edge_attr], dim=0), 
                              z=z)
            graph_conf.conf = conf[carbon_index]
            graph_conf.full_edge_index = torch.cat([ext_edge_index[:,edge_carbon_index], cg_atom_index, cg_pol_index], dim=1)
            graph_conf.boxsize = dataset["boxsize"][index,1] -dataset["boxsize"][index,0]
            graph_conf.mol = dataset["mol"][carbon_index]
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
        #read the data
        df = pd.read_csv(filename, skiprows=skip_index, sep=' ', names=header)
        n_conf = len(df) // self.n_atoms
        pos = torch.tensor(df[['xu', 'yu', 'zu']].values, dtype=torch.double)[None,...].view(n_conf,self.n_atoms, 3)
        mol = torch.tensor(df["mol"].values)[None,...].view(n_conf,self.n_atoms)
        id = torch.tensor(df["id"].values)[None,...].view(n_conf,self.n_atoms)
        element = df['element'].values[None,...].reshape(n_conf,self.n_atoms)
        self.n_molecules = 10
        self.atom_in_mol = pos.size(1) // self.n_molecules
        #assert len(atom_type) == mol.size(0)
        del df
            
        return { 'id': id,
            "pos": pos,
            "mol":mol,
            "element":element,
            "boxsize":boxsize,
            }
        
    def get_cg_dist(self, data):
        
        atom_monomer_id = data.mol # [n_atoms x 1], values from 1 to n_monomers 
        dist = torch.zeros(self.n_molecules*self.atom_in_mol,3, dtype=torch.float64, device=self.device)
        ################################
        for i in range(self.n_monomers*self.n_molecules):
            mon_id = i % self.n_monomers
            pol_id = i // self.n_monomers
            index = atom_monomer_id == mon_id+1
            select_index = torch.arange(self.n_molecules) != pol_id
            index = index.reshape(self.n_molecules, self.atom_in_mol)
            index[select_index] = False
            index = index.reshape(self.n_molecules* self.atom_in_mol)
            dist[index] = data.conf[index] - data.cg_pos[i:i+1].repeat(index.sum().item(),1)
       
        return dist
    
    def reverse_coarse_grain(self,batch, batch_size: int):
        """
        Reverse the coarse graining
        
        """
        n_atom_in_mol = 40
        n_atoms = batch.conf.size(0) // batch_size
        n_molecules = n_atoms  // n_atom_in_mol
        n_monomers = int(batch.mol.max().item())
        pos = batch.conf.view(batch_size, n_atoms, -1) 
        cg_perb_dist = batch.cg_perb_dist[batch.mask.bool()].view(batch_size, n_atoms, -1)
        cg_confs =  batch.cg_pos.view(batch_size, int(n_molecules*n_monomers), -1)
        #sel = batch.batch == 0
        #atom_monomer_id = batch.mol[sel[:batch.conf.size(0) // batch_size]] 
        perb_pos = pos.clone()
        batch.perb_pos = torch.zeros_like(batch.cg_perb_dist)
        for i in range(int(n_monomers*n_molecules)):
            mon_id = i % n_monomers
            pol_id = i // n_monomers
            index = self.atom_monomer_id == mon_id+1
            select_index = torch.arange(n_molecules) != pol_id
            index = index.reshape(n_molecules, n_atom_in_mol)
            index[select_index] = False
            index = index.reshape(int(n_molecules* n_atom_in_mol))
            perb_pos[:,index] = cg_confs[:,i:i+1].repeat(1,index.sum().item(),1) +  cg_perb_dist[:,index]
        # batch.cg_std_dist[...,None,None] * 
        perb_pos = perb_pos.reshape(-1,3) 
        batch.perb_pos[batch.mask.bool()] = perb_pos
        batch.perb_pos[~batch.mask.bool()] = batch.cg_pos
        return batch
    
    
    def get_com_matrix(self,data):
    
        self.n_atoms = data.conf.size(0)
        self.n_monomers = data.mol.max().item() 
        self.n_polymers = 50
        self.n_atoms_polymer =self.n_atoms // self.n_polymers
        self.com_matrix = torch.zeros(self.n_monomers*self.n_polymers, self.n_atoms, dtype=torch.float64, device=self.device)
        self.atom_monomer_id = data.mol # [n_atoms x 1], values from 0 to n_monomers -1 
       
        for i in range(self.n_monomers*self.n_polymers):
            mon_id = i % self.n_monomers #if  i != self.n_monomers else self.n_monomers-1
            pol_id = i // self.n_monomers
            monomers_atom_index = torch.where(self.atom_monomer_id - 1 == mon_id)[0]
            atom_select_index = (monomers_atom_index>=pol_id*self.n_atoms_polymer) & (monomers_atom_index<(pol_id+1)*self.n_atoms_polymer)
            
            monomers_atom_index = monomers_atom_index[atom_select_index]
            atom_numbers = data.z[monomers_atom_index]
            monomer_mass = atom_numbers.sum().item()
            mass_weight = atom_numbers / monomer_mass
            #total_monomer_mass = torch.tensor(list(Counter(dataset["atom_types"][index]).values())) @ torch.tensor([6,1])
            self.com_matrix[i, monomers_atom_index] = mass_weight * torch.ones_like(
                monomers_atom_index, dtype=torch.float64, device=self.device)
            
        return (self.com_matrix, self.atom_monomer_id)
    
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
        dataset = PolymerMeltDataset(**args, mode=mode)
        torch.manual_seed(12345)
        #chunks = len(dataset) // world_size
        #dataset = dataset[int(rank*chunks):int((rank+1)*chunks)]
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False if mode == 'test' else True,
        )
        loaders[mode] = loader
    return loaders, dataset
    

    
if __name__ == "__main__":
    
    import argparse 
    args = {"path":"dataset", "device": 'cpu'}
    args = argparse.Namespace(**args)
    loaders, dataset = get_dataloader(args, batch_size=1)
    for batch in loaders["train"]:
        print(batch.to(args.device))
        batch_cg = dataset.coarse_grain(batch.to(args.device))
        break
    
    
