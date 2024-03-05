import numpy as np
from tqdm import tqdm
import torch
#import diffusion.torus as torus


def train_epoch(loss_fn, model, loader, optimizer, device):
    loss_tot = 0
    base_tot = 0
    model.train()
    for data in tqdm(loader, total=len(loader)):
        data = data.to(device)
        optimizer.zero_grad()

        loss = loss_fn(model,data)

        loss.backward()
        optimizer.step()
        loss_tot += loss.item()
        #base_tot += (score ** 2 / score_norm).mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg


@torch.no_grad()
def test_epoch(loss_fn, model, loader, device):
    loss_tot = 0
    base_tot = 0
    model.eval()
    for data in tqdm(loader, total=len(loader)):

        loss = loss_fn(model,data)

        loss_tot += loss.item()
        #base_tot += (score ** 2 / score_norm).mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg



def get_ddpm_loss_fn(sampler, model, batch, train, reduce_mean=True):
  
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  
  labels = torch.randint(0, sampler.N, (batch.shape[0],), device=batch.device)
  sqrt_alphas_cumprod = sampler.sqrt_alphas_cumprod.to(batch.device)
  sqrt_1m_alphas_cumprod = sampler.sqrt_1m_alphas_cumprod.to(batch.device)
  noise = torch.randn_like(batch)
  perturbed_data = sqrt_alphas_cumprod[labels, None] * batch + \
                    sqrt_1m_alphas_cumprod[labels, None] * noise
  score = model(perturbed_data, labels)
  losses = torch.square(score - noise)
  losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
  loss = torch.mean(losses)
  return loss