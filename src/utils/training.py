from os import makedirs
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from functools import partial
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader
from model.score_model import create_model
from sampling.cond_sampler import create_sampler
from data.dataset import get_dataloader
from utils.logger import get_logger

import os
def train_epoch(loss_fn, model, loader, optimizer, rank: int):
    loss_tot = 0
    #rank = 'cpu'
    model.train()
    for data in tqdm(loader, total=len(loader)):
        optimizer.zero_grad()
        loss = loss_fn(model=model,
                       batch=data.to(rank))
        loss.backward()
        optimizer.step()
        loss_tot += loss.detach().item()
    loss_avg = loss_tot / len(loader)
    return loss_avg


def test_epoch(loss_fn, model, loader, rank: int):
    loss_tot = 0
    #rank = 'cpu'
    model.eval()
    for data in tqdm(loader, total=len(loader)):
        with torch.no_grad():
            loss = loss_fn(model=model,
                        batch=data.to(rank))
        loss_tot += loss.item()
        
    loss_avg = loss_tot / len(loader)
    return loss_avg



def get_ddpm_loss_fn(sampler, dataset, model, batch, reduce_mean=True):
  
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  batch_size = batch.batch.max().item() + 1 
  t = torch.randint(0, sampler.num_timesteps, (batch_size,), device=batch.conf.device)
  t= t.repeat_interleave(batch.conf.shape[0] // batch_size)
  assert t.shape[0] == batch.conf.shape[0]
  batch.node_sigma = t
  perb_dist, noise = sampler.q_sample(batch.cg_dist, t)
  batch.cg_perb_dist = perb_dist  /  batch.cg_std_dist[0]
  batch = dataset.reverse_coarse_grain(batch, batch_size=batch_size)
  score = model(batch)
  losses = torch.square(score - noise)
  losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
  loss = torch.mean(losses)
  return loss



def run_process(rank: int,ddp:int, world_size: int, device, train_config, diffusion_config, data_config ,model_config, run):
    
    logger = get_logger()
    #assert train_config.batch_size % world_size == 0, "Batch size must be divisible by the number of devices."
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12345'
    #dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    
    # Dataset 
    loader, dataset = get_dataloader(data_config, 
                                     batch_size=train_config.batch_size,
                                     rank=rank, 
                                     world_size=world_size)
    
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    # Loss function 
    loss_fn = partial(get_ddpm_loss_fn, sampler=sampler, dataset=dataset)

    
    torch.manual_seed(12345)
    model = create_model(**model_config)
    model = model.to(rank if device == 'gpu' else 'cpu')
    if ddp:
        ddp_model = DistributedDataParallel(model, 
                                            device_ids=[rank] if device == 'gpu' else None, 
                                            find_unused_parameters=True)
    else:
        ddp_model = model
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=train_config.lr)
    
    best_loss = np.inf
    if train_config.save:
        savepath = os.path.join(train_config.savepath, run.name)
        makedirs(savepath, exist_ok=True)
        
    # Train & Val
    for epoch in range(train_config.epochs):
        
        #train_loader = partition_batch(rank, world_size, loader["train"])
        train_loss = train_epoch(loss_fn, ddp_model, loader["train"], optimizer, rank)
        #dist.barrier()
        test_loss = test_epoch(loss_fn, ddp_model, loader["val"], rank)
        #dist.barrier()
        if rank == 0:
            logger.info(f"Epoch {epoch+1}, train_loss: {train_loss}, test_loss: {test_loss} ")
            run.log({"epoch": epoch+1, "train_loss": train_loss, "test_loss": test_loss})
            
        if rank == 0 and epoch % train_config.checkpt_freq == 0 and train_config.save:
            checkpoint_path = os.path.join(savepath, f"checkpoint_{epoch+1}.h5")
            torch.save(ddp_model.state_dict(), checkpoint_path)
            #run.save(checkpoint_path, base_path=train_config.savepath)
            if test_loss < best_loss and test_loss >= train_loss:
                    best_loss = test_loss
                    run.log({"best_epoch": epoch, 
                              "best_test_loss": best_loss})
                    checkpoint_path = os.path.join(savepath, "best_model.h5")
                    torch.save(model.state_dict(), checkpoint_path)
                    run.save(checkpoint_path, base_path=train_config.savepath)
        #dist.barrier()

    #dist.destroy_process_group()
    
    
    
def partition_batch(rank: int, world_size: int, loader):
    new_batch = []
    for batch in loader:
        unique_batch_indices = batch.batch.unique()
        
        # Split the unique batch indices into chunks
        chunks = torch.chunk(unique_batch_indices, world_size)
        
        # Get the batch indices for the current GPU
        batch_indices = chunks[rank]
        box_size = batch.boxsize.view( batch.boxsize.size(0) // 2, 2, 3)[batch_indices]
        box_size = box_size[:,1] - box_size[:,0]
        edge_indices = torch.isin(batch.batch[batch.edge_index[0]], batch_indices)
        sub_batch_index = torch.isin(batch.batch, batch_indices)
        new_batch.append(Data(conf = batch.conf[sub_batch_index],
                        x= batch.x[sub_batch_index],
                        z= batch.z[sub_batch_index],
                        mol= batch.mol[sub_batch_index],
                        cg_dist= batch.cg_dist[sub_batch_index],
                        boxsize=box_size, 
                        edge_index= batch.edge_index[:,edge_indices] - batch.edge_index[:,edge_indices].min().long(),
                        edge_attr=batch.edge_attr[edge_indices],
                        batch=batch.batch[sub_batch_index] - batch_indices.min().long(),
                        cg_pos=batch.cg_pos[batch_indices.min().long().item()* 500:(batch_indices.max().long().item()+1)*500]                                
        ))
        
    new_loader = DataLoader(dataset=new_batch,
                        batch_size=1,
                        shuffle=True)
    return new_loader