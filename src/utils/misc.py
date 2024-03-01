from scipy.spatial import KDTree
import torch
from torch import Tensor
import numpy as np
import math
from torch.nn import functional as F
import model.sde as sde_lib

# The following function computes the radius graph of the input nodes positions using KDTrees and the box size
def pbc_radius_graph(pos: Tensor, r: float, box_size: Tensor, batch=None):
    # KDTree generation for batched positions input
    edge_index = []
    
    for batch_idx in range(box_size.shape[0]):
        kdtree = KDTree(pos[batch == batch_idx].cpu().numpy(), boxsize=box_size[batch_idx].cpu().numpy())
        pairs = kdtree.query_pairs(r, output_type='ndarray').swapaxes(0,1)
        edge_index.append(pairs)
    edge_index = np.array(edge_index).swapaxes(0,1).reshape(2, -1)
    return torch.tensor(edge_index, dtype=torch.long).contiguous()


# Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py 
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb   

def get_beta_scheduler(beta_schedule: str, *, beta_start:float, beta_end:float, diffusion_timesteps: int):
    
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    
    if beta_schedule == "sigmoid":
        betas = torch.linspace(-10, 10, steps=diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "linear":
        betas = torch.linspace(
            beta_start, beta_end, steps=diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"Unknown or Not Implemented beta schedule: {beta_schedule}")
    assert betas.shape == (diffusion_timesteps,)
    return betas

_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]

def create_model(config):
  """Create the score model."""
  model_name = config.model.name
  score_model = get_model(model_name)(config)
  score_model = score_model.to(config.device)
  score_model = torch.nn.DataParallel(score_model)
  return score_model


def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

        Args:
            model: The score model.
            train: `True` for training and `False` for evaluation.

        Returns:
            A model function.
        """

  def model_fn(x, labels):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
     """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model.
        train: `True` for training and `False` for evaluation.
        continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
        A score function.
    """
     model_fn = get_model_fn(model, train=train)
     
     if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                diff_timestep_index = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[diff_timestep_index.long()]

            score = -score / std[:, None]
            return score
        
     elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model_fn(x, labels)
            return score

     else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    
     return score_fn





