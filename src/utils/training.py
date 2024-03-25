import numpy as np
from tqdm import tqdm
import torch
from pickle import dump


def train_epoch(loss_fn, model, loader, optimizer, device):
    loss_tot = 0
    model.train()
    for data in tqdm(loader, total=len(loader)):
        optimizer.zero_grad()
        loss = loss_fn(model=model,
                       batch=data.to(device))
        loss.backward()
        optimizer.step()
        loss_tot += loss.detach().item()
        del loss
    loss_avg = loss_tot / len(loader)
    return loss_avg


@torch.no_grad()
def test_epoch(loss_fn, model, loader, device):
    loss_tot = 0
    model.eval()
    for data in tqdm(loader, total=len(loader)):
        loss = loss_fn(model=model,
                       batch=data.to(device))
        loss_tot += loss.item()
        del loss
    loss_avg = loss_tot / len(loader)
    return loss_avg



def get_ddpm_loss_fn(sampler, dataset, model, batch, train, reduce_mean=True):
  
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  batch_size = batch.batch.max().item() + 1 
  t = torch.randint(0, sampler.num_timesteps, (batch_size,), device=batch.conf.device)
  t= t.repeat_interleave(batch.conf.shape[0] // batch_size)
  assert t.shape[0] == batch.conf.shape[0]
  batch.node_sigma = t
  perb_dist, noise = sampler.q_sample(batch.cg_dist, t)
  batch.cg_perb_dist = perb_dist
  batch = dataset.reverse_coarse_grain(batch, batch_size=batch_size)
  score, _ = model(batch)
  losses = torch.square(score - noise)
  losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
  loss = torch.mean(losses)
  return loss