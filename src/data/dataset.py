import os.path as osp
import os 
from multiprocessing import Pool

from rdkit import Chem
import numpy as np 
import torch
from tqdm import tqdm 
import copy
import pandas as pd
from torch_geometric.data import Dataset, DataLoader, Data

from featurization import featurize_polymer, featurize_pol_from_smiles
TYPES = {'H': 0, 'C': 1}

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper

def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)

@register_dataset('polymer_melt')
class PolymerMeltDataset(Dataset):
    def __init__(self,  dataset_path , mode, num_workers=1, device: str='cpu'):
        super(PolymerMeltDataset, self).__init__()
        self.device = device
        self.path = dataset_path
        self.num_workers = num_workers
        
        self.data = self.preprocessing(self.path)
        
    def preprocessing(self, path):
        """
        Preprocess the data
        """
        listdir = os.listdir(path)
        types = []
        smiles = []
        dataset = []
        residue = "C\C=C/C"
        mol_features = []
        for ds in listdir:
            data = self.read_lammpstraj(osp.join(path, ds, 'dump.lammpstrj'))
            dataset.append(data)
            types.append(data['types'])
            smiles.append(r"C\C=C/C" * data['n_monomers'].max())
            
        self.smiles = set(smiles)
        self.types = set(types[0])
        self.datapoints = self.smiles_to_graph(dataset[0])
        return self.confs
    
    
    def smiles_to_graph(self, dataset):
        
        self.mol_features= [featurize_pol_from_smiles(smile, types=self.types, hydrogens=True) for smile in self.smiles][0]
        atom_in_mon = 102
        perm_index = []
        match_order = dataset["atom_types"][:atom_in_mon]
        for i in range(dataset["n_monomers"].max()):
            
            perm_index += 1 
            
        graph_inst = []
        for data in dataset["pos"]:
            assert int(self.mol_features.x.size(0) * 50) == data.size(0)
            
            graph_inst.append(Data(pos=data))
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
        pos = torch.Tensor(df[['xu', 'yu', 'zu']].values).to(self.device)[None,...].view(n_conf,n_atoms, 3)
        mol = torch.Tensor(df["mol"].values).to(self.device)[None,...].view(n_conf,n_atoms)[0]
        atom_type = df['element'].values[:skip_index[9]-9]
        assert len(atom_type) == mol.size(0)
        del df
            
        
        #n_molecules = 
        return {"pos": pos,
            "mol":mol,
            "types":types,
            "atom_types":atom_type,
            "n_monomers":n_monomers}
        
    
    def get(self, idx):
        data = self.datapoints[idx]
        return copy.deepcopy(data)
    
    def len(self):
        return len(self.datapoints)
            
    
def get_dataloader(args, modes=('train', 'val')):
    """
    Get the dataloader for the dataset
    """
    if isinstance(modes, str):
        modes = [modes]
        
    loaders = {}
    for mode in modes:
        dataset = PolymerMeltDataset(args.dataset_path, mode, args.num_workers)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False if mode == 'test' else True, num_workers=args.num_workers)
        loaders.append(loader)
    return loaders

    
if __name__ == "__main__":
    
    
    datapath = "dataset"
    dataset = PolymerMeltDataset(datapath, 'train')
    #print(data)
    
    