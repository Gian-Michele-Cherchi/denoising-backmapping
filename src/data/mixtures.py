from dataset import PolymerMeltDataset
from dataset import register_dataset, get_dataset
import pandas as pd
import os.path as osp
import os
import random
from collections import Counter
import numpy as np 
import torch
import copy
from pyexcel_ods import get_data
import re
import pandas as pd
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from featurization import featurize_pol_from_smiles, lammps_rdkit_matching

# this class inherits all the methods of polymer melt dataset class but can deal with mixtures of polymers and different smiles representations

@register_dataset('polymer_mixture')
class PolymerMixtureDataset(PolymerMeltDataset):
    def __init__(self,  path: str, mode: str, n_conf: int,coarse_grain: bool=True , hydrogens: bool=False,  save_ckpt:bool=True, device: str='cpu'):
        #super(PolymerMixtureDataset, self).__init__(path, mode, coarse_grain, hydrogens, save_ckpt, device) # omitted 
        
        self.path = path
        self.n_monomers = 10
        self.save = save_ckpt
        self.nconf = n_conf
        self.hydrogens = hydrogens
        smiles_path = os.path.join(path, 'mixture_smiles.csv')
        
        mix_smiles = pd.read_csv(smiles_path)
        
        dirs = [s for s in os.listdir(path) if not s.endswith(('.ods', '.csv'))]
        
        for directory in dirs:
            
            self.sim_directory = directory
            self.mixture_path = os.path.join(path, directory)
            smiles1 = mix_smiles[mix_smiles['ID'] == directory]['Comp1Smiles'].values[0]
            smiles2 = mix_smiles[mix_smiles['ID'] == directory]['Comp2Smiles'].values[0]
            self.monomers_smiles = (smiles1, smiles2) # gets updated for every simulation folder 
            
            confs = self.preprocessing() #checkpoint save the data
            
            
            
    
    def load_dataset(self):
        pass
    
    def preprocessing(self):
        """
        Preprocess the data
        """
        
        dump_data = self.read_lammpstraj(osp.join(self.mixture_path, 'dump.lammpstrj')) # read the mixture lammpstraj file and get the mixture data dict
        dump_data['time_index'] = 0 
        polymers_smiles = [self.monomers_smiles[0] * self.n_monomers, 
                           self.monomers_smiles[1] * self.n_monomers] # mixture polymer smiles

        mol_features= [featurize_pol_from_smiles(smile, hydrogens=self.hydrogens) for smile in polymers_smiles]
        
        dump_data = lammps_rdkit_matching(dump_data, 
                                          mol_features,
                                          lmp_filepath=os.path.join(self.path, self.sim_directory, "data.equil.lmp"),
                                          n_monomers=self.n_monomers,
                                          hydrogens=self.hydrogens
                                          )
        
        confs = self.GetFeaturizedDataConfs(dump_data, mol_features)
        
        if self.cg:
            for conf in confs.values():
                self.com_matrix, _ = self.get_com_matrix(conf)
                conf.cg_pos = self.com_matrix @ conf.conf
                conf.cg_dist = self.get_cg_dist(conf)
                conf.cg_dist = torch.cat([conf.cg_dist, torch.zeros_like(conf.cg_pos)], dim=0)
                conf.cg_std_dist = torch.sqrt(torch.trace(conf.cg_dist.T @ conf.cg_dist) / (3*conf.cg_dist.size(0)))
                conf.full_conf = torch.cat([conf.conf, conf.cg_pos], dim=0)
                conf.mask = torch.cat([torch.ones_like(conf.conf[:,0]), torch.zeros_like(conf.cg_pos[:,0])], dim=0) 
        if self.save:
            self.save_ckpt(confs)
            
        return confs
    
    def GetFeaturizedGraphData(self, dump_data, features):
        """
        Get the featurized data for each configuration
    
        """
        n_edges = self.mol_features[0].edge_index.size(1)
        atoms_in_mol  = self.mol_features[0].x.size(0)
        match_order = dataset["atom_types"][:atoms_in_mol]
        indexes_res = [i for i, x in enumerate(match_order) if x != 'H']
        indexes_hyd = [i for i, x in enumerate(match_order) if x == 'H']
        self.reindex = indexes_res + indexes_hyd
        ext_edge_index = self.mol_features[0].edge_index.repeat(1,self.n_molecules)
        const = 0 
        const_cg = 0
        
        graph_inst = {}
        for index in range(self.nconf):
             graph_inst["conf"+str(index)] = self.GetGraphConf(dump_data, features, index)
        return graph_inst
    
    

    def GetGraphConf(self, dump_data, features, index):
        
        return 0
        
     
    
# Function to extract unique elements from a SMILES string
def extract_elements(smiles):
    # Regex to match elements from the periodic table (one or two capital letters, possibly followed by lowercase)
    return re.findall(r'[A-Z][a-z]?', smiles)

    
        
if __name__ == "__main__":
    
    import argparse 
    #args = {"path":"dataset/run_mixtures", "device": 'cpu', "coarse_grain": True, "hydrogens": False, "save_ckpt": True}
    #args = argparse.Namespace(**args)
    dataset = PolymerMixtureDataset(
        path="dataset/run_mixtures", 
        mode='train', n_conf=10, 
        device='cpu', 
        coarse_grain=True, 
        hydrogens=False, 
        save_ckpt=True
        )
    
    
    
    
    