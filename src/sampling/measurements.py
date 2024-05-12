'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from collections import Counter
import torch

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)

# =============
# Operators Abstract Classes
# =============

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

# =============
# Operators
# =============
@register_operator(name="coarse_grain")
class CoarseGrainOperator(LinearOperator):
    def __init__(self, device, operator, n_atoms, n_monomers, n_polymers):
        self.device = device
        self.com_matrix, self.atom_monomer_id = operator
        self.n_atoms = n_atoms
        self.n_monomers = n_monomers
        self.n_polymers = n_polymers
        self.n_atom_in_mol = n_atoms // n_polymers
        
    def forward(self, cg_distances, noisy_measurement=None, **kwargs):
        if len(cg_distances.shape) == 3:
            batch_size = cg_distances.shape[0]
            cg_distances = cg_distances.view(batch_size, self.n_atoms, -1)
            noisy_measurement =  noisy_measurement.view(batch_size, int(self.n_polymers*self.n_monomers), -1)
        elif len(cg_distances.shape) == 2:
            cg_distances = cg_distances.unsqueeze(0)
            noisy_measurement = noisy_measurement.unsqueeze(0)
            if len(self.com_matrix.shape) == 2:
                self.com_matrix = self.com_matrix.unsqueeze(0)
            batch_size = 1
        else:
            raise ValueError("cg_distances should have 2 or 3 dimensions.")
        
        lifted_measurement = cg_distances.clone()
        for i in range(int(self.n_monomers*self.n_polymers)):
            mon_id = i % self.n_monomers
            pol_id = i // self.n_monomers
            index = self.atom_monomer_id == mon_id+1
            select_index = torch.arange(self.n_polymers) != pol_id
            index = index.reshape(self.n_polymers, self.n_atom_in_mol)
            index[select_index] = False
            index = index.reshape(int(self.n_polymers* self.n_atom_in_mol))
            lifted_measurement[:,index] = noisy_measurement[:,i:i+1].repeat(1,index.sum().item(),1) 
        lifted_measurement = lifted_measurement.view(batch_size, self.n_atoms, -1)
        
        perb_pos = lifted_measurement +  self.cg_std_dist[...,None,None] * cg_distances
        reconstructed_cg_pos = self.com_matrix.to(self.device).bmm(perb_pos)
        if batch_size == 1:
            return reconstructed_cg_pos.squeeze(0)
        else:
            return reconstructed_cg_pos
    
        #batch_size = data.batch.unique().shape[0]
        #batch_com_matrix = torch.block_diag(*[self.com_matrix for _ in range(batch_size)])
        
        #com_confs =  self.com_matrix @ data.conf
        
    def inject_std(self, sigma):
        self.cg_std_dist = sigma
    
    def get_cg_matrix(self):
        return self.com_matrix 
    
    def transpose(self, data, **kwargs):
        return self.com_matrix.T @ data
    
    

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma