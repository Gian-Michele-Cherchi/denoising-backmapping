import os
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

def train_epoch(loss_fn, model, loader, optimizer, rank: int):
    loss_tot = []
    #rank = 'cpu'
    model.train()
    for data in tqdm(loader, total=len(loader)):
        optimizer.zero_grad()
        loss = loss_fn(model=model,
                       batch=data.to(rank))
        loss.backward()
        optimizer.step()
        loss_tot.append(loss.detach().item())
    loss_avg = np.mean(loss_tot)
    loss_std = np.std(loss_tot)
    return loss_avg, loss_std


def test_epoch(loss_fn, model, loader, rank: int):
    loss_tot = []
    #rank = 'cpu'
    model.eval()
    for data in tqdm(loader, total=len(loader)):
        with torch.no_grad():
            print("before loss_fn")
            loss = loss_fn(model=model,
                        batch=data.to(rank))
        print("before append loss")
        loss_tot.append(loss.item())
    print("before loss_avg")
    loss_avg = np.mean(loss_tot)
    loss_std = np.std(loss_tot)
    return loss_avg, loss_std



def get_ddpm_loss_fn(sampler, dataset, model, batch, reduce_mean=True):
  
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  batch_size = batch.batch.max().item() + 1 
  t = torch.randint(0, sampler.num_timesteps, (batch_size,), device=batch.conf.device)
  t= t.repeat_interleave(batch.conf.shape[0] // batch_size)
  assert t.shape[0] == batch.conf.shape[0]
  batch.node_sigma = t
  sigma = batch.cg_std_dist.repeat_interleave(batch.conf.shape[0] // batch_size)
  perb_dist, noise = sampler.q_sample(batch.cg_dist / sigma[...,None], t)
  batch.cg_perb_dist = perb_dist
  batch = dataset.reverse_coarse_grain(batch, batch_size=batch_size)
  score = model(batch)
  losses = torch.square(score - noise)
  losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
  loss = torch.mean(losses)
  return loss



def run_process(rank: int,ddp:int, world_size: int, device, train_config, diffusion_config, data_config ,model_config, run):
    checkpt = False
    logger = get_logger()
    
    setup(train_config.batch_size, world_size, rank) if ddp else None
    
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
    # checkpoints 
    if checkpt:
        checkpt_path = os.path.join(train_config.savepath, run.name, 
                                    "checkpoint"+str(train_config.checkpt_epoch)+".h5")
        checkpoint = torch.load(checkpt_path)
        model.load_state_dict(checkpoint)
        
    ddp_model = get_ddp_model(model, rank, device, ddp)
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=train_config.lr)
    
    if train_config.save:
        savepath = os.path.join(train_config.savepath, run.name)
        makedirs(savepath, exist_ok=True)
        
    # Train & Val
    best_loss = np.inf
    if checkpt:
        n_start = train_config.checkpt_epoch
    else:
        n_start = 0
    for epoch in range(n_start, train_config.epochs):
        
        #train_loader = partition_batch(rank, world_size, loader["train"])
        #dist.barrier()
        train_loss, train_std = train_epoch(loss_fn, ddp_model, loader["train"], optimizer, rank)
        dist.barrier()
        test_loss, test_std = test_epoch(loss_fn, ddp_model, loader["val"], rank)
        print("after test loss")
        dist.barrier()
        print("outside")
        if rank == 0:
            #print("inside rank 0")
            #test_loss, test_std = test_epoch(loss_fn, ddp_model, loader["val"], rank)
            print("after test rank 0")
            logger.info(f"Epoch {epoch+1}, train_loss: {train_loss}, test_loss: {test_loss} ")
            run.log({"epoch": epoch+1, "train_loss": train_loss, "test_loss": test_loss, 
                     "train_std": train_std, "test_std": test_std})
        print("before 2nd if")   
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
        print("before 2nd barrier")            
        dist.barrier()
        print("after 2nd barrier")
    clean() if ddp else None
    
    
    
def partition_batch(rank: int, world_size: int, loader):
    new_batch = []
    for batch in loader:
        unique_batch_indices = batch.batch.unique()
        
        # Split the unique batch indices into chunks
        chunks = torch.chunk(unique_batch_indices, world_size)
        
        # Get the batch indices for the current GPU
        batch_indices = chunks[rank]
        box_size = batch.boxsize.view( unique_batch_indices.max() +1, 3)[batch_indices]
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



def setup(batch_size, world_size, rank):
    assert batch_size % world_size == 0, "Batch size must be divisible by the number of devices."
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    
def clean():
    dist.destroy_process_group()
    

def get_ddp_model(model, rank, device, ddp):
    if ddp:
        ddp_model = DistributedDataParallel(model, 
                                            device_ids=[rank] if device == 'gpu' else None, 
                                            find_unused_parameters=True)
    else:
        ddp_model = model
    return ddp_model
