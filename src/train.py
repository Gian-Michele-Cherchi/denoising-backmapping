
#########################################CONFIG 
import torch
import torch.multiprocessing as mp
from utils.training import run_process
import wandb
import hydra 
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="base_config")
def train_app(cfg: DictConfig) -> None:
    # log to wandb and  get train config
    run = wandb.init(
        project="denoising-backmapping",
        config=cfg.train.__dict__['_content']
    )
    diffusion_config = cfg.train.diffusion
    model_config = cfg.train.model
    data_config = cfg.train.data
    train_config = cfg.train.train
    
    ddp=False
    device = 'gpu'
    #CPU/GPU Multiprocessing spawn process 
    
    world_size = torch.cuda.device_count() if device == 'gpu' else 2
    if ddp:
        mp.spawn(run_process, args=(
            world_size, 
            device,
            train_config, 
            diffusion_config,
            data_config,
            model_config, 
            run,
            ), 
            nprocs=world_size, 
            join=True
                )
    else:
        gpu_id = train_config.gpu_id
        run_process(gpu_id,ddp, world_size, device, train_config, diffusion_config, data_config, model_config, run)
        

    
if __name__ == "__main__":
    train_app()