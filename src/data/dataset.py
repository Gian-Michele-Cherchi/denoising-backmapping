import os.path as osp
import os 
from multiprocessing import Pool

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
    def __init__(self,  path , mode,  save_ckpt:bool=True, device: str='cpu'):
        super(PolymerMeltDataset, self).__init__()
        self.device = device
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
            self.datapoints = self.datapoints[:int(0.6*self.n_conf)]
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
            smiles.append(r"C\C=C/C" * data['n_monomers'].max())
            
        self.smiles = set(smiles)
        self.types = set(types[0])
        confs = self.features_from_smiles(dataset[0])
        return confs
    
    
    def features_from_smiles(self, dataset):
        
        self.mol_features= [featurize_pol_from_smiles(smile, types=self.types, hydrogens=True) for smile in self.smiles][0]
        atom_in_mon = 102
        n_molecules = dataset["pos"].size(1) // atom_in_mon
        n_edges = self.mol_features[0].edge_index.size(1)
        match_order = dataset["atom_types"][:atom_in_mon]
        indexes_res = [i for i, x in enumerate(match_order) if x != 'H']
        indexes_hyd = [i for i, x in enumerate(match_order) if x == 'H']
        reindex = indexes_res + indexes_hyd
        ext_edge_index = self.mol_features[0].edge_index.repeat(1,n_molecules)
        const = 0 
        assert len(reindex) == atom_in_mon
        for i in range(n_molecules):
            dataset["pos"][:,i*atom_in_mon:(i+1)*atom_in_mon] = dataset["pos"][:,i*atom_in_mon:(i+1)*atom_in_mon][:,reindex]
            ext_edge_index[:,i*n_edges:(i+1)*n_edges] = const + ext_edge_index[:,i*n_edges:(i+1)*n_edges]
            const += atom_in_mon
        graph_inst = {}
        for index,conf in enumerate(dataset["pos"]):
            graph_conf = Data(x=self.mol_features[0].x.repeat(n_molecules, 1),
                              edge_index=ext_edge_index, 
                              edge_attr=self.mol_features[0].edge_attr.repeat(n_molecules, 1), 
                              z=self.mol_features[0].z.repeat(n_molecules))
            graph_conf.pos = conf
            graph_inst["conf"+str(index)] = graph_conf
        return graph_inst
        
    def read_lammpstraj(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        timestep_indices = [i for i, line in enumerate(lines) if line.startswith('ITEM: TIMESTEP')]
        n_atoms_indices = [i for i, line in enumerate(lines) if line.startswith('ITEM: NUMBER OF ATOMS')]
        n_atoms = int(lines[n_atoms_indices[0]+1].strip().split('\n')[0])
        header = lines[timestep_indices[0]+8].strip().split(' ')[2:]
        skip_index = np.array([list(range(i,i+9)) for i in timestep_indices]).reshape(-1)
        df = pd.read_csv(filename, skiprows=skip_index, sep=' ', names=header)
        n_monomers = df.mol.unique()
        types  = list(df.element.unique())
        n_conf = len(df) // n_atoms
        pos = torch.Tensor(df[['xu', 'yu', 'zu']].values)[None,...].view(n_conf,n_atoms, 3)
        mol = torch.Tensor(df["mol"].values)[None,...].view(n_conf,n_atoms)[0]
        atom_type = df['element'].values[:skip_index[9]-9]
        assert len(atom_type) == mol.size(0)
        del df
            
        return {"pos": pos,
            "mol":mol,
            "types":types,
            "atom_types":atom_type,
            "n_monomers":n_monomers}
        
    def coarse_grain(self, ):
        
        for conf in batch:
            conf = self.smiles_to_graph(conf)
    
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
                            shuffle=False if mode == 'test' else True, num_workers=4)
        loaders[mode] = loader
    return loaders, dataset

    
if __name__ == "__main__":
    
    import argparse 
    args = {"dataset_path":"dataset", "num_workers":4, "batch_size":32, "device": 'cuda:0'}
    args = argparse.Namespace(**args)
    loaders, dataset = get_dataloader(args)
    for batch in loaders["train"]:
        print(batch.to(args.device))
        batch_cg = dataset.coarse_grain(batch.to(args.device))
        break
    
    