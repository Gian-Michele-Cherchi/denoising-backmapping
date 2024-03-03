'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
from torch.nn import functional as F
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
    def __init__(self, n_atoms, n_monomers, n_polymers, device):
        self.device = device
        self.n_atoms = n_atoms
        self.n_monomers = n_monomers
        self.n_polymers = n_polymers
        self.com_matrix = torch.zeros(n_monomers, n_atoms, device=device)

    def forward(self, data, **kwargs):
        tot_mass = data.mon_masses.sum() # total monomer mass 
        atom_monomer_id = data.atom_monomer_id # [n_atoms x 1], values from 0 to n_monomers -1 
        n_atoms_monomer = self.n_monomers*( atom_monomer_id[:50] == 0 ).sum().item() # number of atoms per polymer
        
        for i in range(self.n_monomers*self.n_polymers):
            mon_id = i % (self.n_monomers-1) if  i != self.n_monomers else self.n_monomers-1
            pol_id = i // self.n_monomers
            self.com_matrix[i, int(pol_id * n_atoms_monomer * self.n_monomers) : int(pol_id * n_atoms_monomer * self.n_monomers) + 1] = torch.tensor(
                atom_monomer_id == mon_id, dtype=torch.float64, device=self.device)
            
        batch_size = data.batch.unique().shape[0]
        batch_com_matrix = torch.block_diag(*[self.com_matrix for _ in range(batch_size)])
        com_confs = (1. / tot_mass ) * (batch_com_matrix @ data.atom_positions)
        
        return com_confs
            
    def get_cg_kernel(self):
        return self.com_matrix 
    
    

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