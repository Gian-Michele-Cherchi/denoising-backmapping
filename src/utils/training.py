import numpy as np
from tqdm import tqdm
import torch
#import diffusion.torus as torus


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_tot = 0
    base_tot = 0

    for data in tqdm(loader, total=len(loader)):
        data = data.to(device)
        optimizer.zero_grad()

        data = model(data)
        pred = data.edge_pred

        score = torus.score(
            data.edge_rotate.cpu().numpy(),
            data.edge_sigma.cpu().numpy())
        score = torch.tensor(score, device=pred.device)
        #score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm, device=pred.device)
        loss = ((score - pred) ** 2 / score_norm).mean()

        loss.backward()
        optimizer.step()
        loss_tot += loss.item()
        base_tot += (score ** 2 / score_norm).mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg


@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    loss_tot = 0
    base_tot = 0

    for data in tqdm(loader, total=len(loader)):

        data = data.to(device)
        data = model(data)
        pred = data.edge_pred.cpu()

        #score = torus.score(
         #   data.edge_rotate.cpu().numpy(),
         #   data.edge_sigma.cpu().numpy())
        score = torch.tensor(score)
        #score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm)
        loss = ((score - pred) ** 2 / score_norm).mean()

        loss_tot += loss.item()
        base_tot += (score ** 2 / score_norm).mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg



def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn




def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    
    if isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")
  
  
  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn