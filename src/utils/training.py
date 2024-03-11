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

        loss = loss_fn(model=model,
                       batch=data)

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

        loss = loss_fn(model=model,
                       batch=data)

        loss_tot += loss.item()
        #base_tot += (score ** 2 / score_norm).mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg



def get_ddpm_loss_fn(sampler, dataset, model, batch, train, reduce_mean=True):
  
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  batch_size = batch.batch.max().item() + 1 
  t = torch.randint(0, sampler.num_timesteps, (batch_size,), device=batch.pos.device)
  t= t.repeat_interleave(batch.pos.shape[0] // batch_size)
  assert t.shape[0] == batch.pos.shape[0]
  batch.node_sigma = t
  perb_dist, noise = sampler.q_sample(batch.cg_dist, t)
  batch.cg_perb_dist = perb_dist
  batch = dataset.reverse_coarse_grain(batch, batch_size=batch_size)
  score = model(batch)
  losses = torch.square(score - noise)
  losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
  loss = torch.mean(losses)
  return loss