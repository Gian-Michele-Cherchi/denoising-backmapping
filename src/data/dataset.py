import os.path as osp
import os 
from multiprocessing import Pool

from rdkit import Chem
import MDAnalysis as mda
import numpy as np 
import pickle 
import torch, tqdm 

from torch_geometric.data import Dataset, DataLoader

from featurization import featurize_polymer, featurize_pol_from_smiles

def read_lammpstrj(filename):
    u = mda.Universe(filename, topology_format="LAMMPSDUMP", format="LAMMPSDUMP")
    return u


class PolymerMeltDataset(Dataset):
    def __init__(self,  dataset_path , mode, num_workers=1):
        super(PolymerMeltDataset, self).__init__()

        self.path = dataset_path
        self.num_workers = num_workers
        
        self.datapoints, self.types = self.preprocessing(self.path)
        
    def preprocessing(self, path):
        """
        Preprocess the data
        """
        listdir = os.listdir(path)
        for ds in listdir:
            filename = osp.join(path, ds+"/dump.lammpstrj")
            universe = mda.Universe(filename, topology_format="LAMMPSDUMP", format="LAMMPSDUMP")
            types = self.collect_types(universe)
            all_positions = [universe.atoms.positions.copy() for ts in universe.trajectory]
            
    
    def collect_types(self, universe):
        """
        Collect the types of atoms in the universe
        """
        return np.array(universe.atoms.types)
    
    
    
def get_dataloader(args, modes=('train', 'val')):
    """
    Get the dataloader for the dataset
    """
    if isinstance(modes, str):
        modes = [modes]
        
    loaders = {}
    for mode in modes:
        dataset = PolymerMeltDataset(args.dataset_path, mode, args.num_workers)
        loaders[mode] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loaders
    
if __name__ == "__main__":
    
    
    datapath = "dataset/cPB_50C_10M/dump.lammpstrj"
    data = read_lammpstrj(datapath)
    #print(data)
    
    