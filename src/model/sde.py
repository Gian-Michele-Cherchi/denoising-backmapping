import numpy as np 
import os
import abc
import torch
import math

from utils.misc import get_beta_scheduler

class SDE(abc.SDE):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""
    
    def __init__(self, N):
       """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
       super().__init__()
       self.N = N
    
    
    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass
    
    @abc.abstractmethod
    def sde(self, x, t):
        pass
    
    @abc.abstractmethod
    def prior_sampling(self, shape):
       """Generate one sample from the prior distribution, $p_T(x)$."""
       pass
   
    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
        x: a torch tensor
        t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
        f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x ,t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G
    
    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

            Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
            """
        
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize
    
        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self,schedule: str="sigmoid", beta_0: float=1e-7, beta_T: float=1e-3 ,N: int=1000):
        """Construct a Variance Preserving SDE.

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_0
        self.beta_T = beta_T
        self.N = N
        
        self.discrete_betas = get_beta_scheduler(beta_schedule=schedule,
                                                 beta_start=beta_0, 
                                                 beta_end=beta_T, 
                                                 diffusion_timesteps=N
                                                 )
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    @property
    def T(self):
        return 1
    
    def sde(self, x, t):
        beta_t = self.discrete_betas[t]
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion 
    
    def marginal_prob(self, x, t):
        log_mean_coeff = - 0.5 * ( self.beta_0 * t + (self.beta_T - self.beta_0) * torch.log(torch.exp(t) + 1) - math.log(2) )  
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean, std
    
    def prior_sampling(self, shape):
        return torch.randn(*shape)
    
    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps 
    
    def discretize(self, x, t):
        """ DDPM discretization of the SDE. """
        timestep = (t * (self.N -1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None] * x - x
        G = sqrt_beta
        return f, G
    

        
        
        
    

    
    