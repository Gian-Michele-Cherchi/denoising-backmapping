
#########################################CONFIG 
from functools import partial
import os
import torch 
from utils.logger import get_logger
from model.score_model import create_model
from sampling.cond_sampler import create_sampler
from data.dataset import get_dataset, get_dataloader
from utils.training import train_epoch, test_epoch, get_ddpm_loss_fn
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
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{train_config.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str) 
    
    # Load model
    #model_path = os.path.join(cfg.save_dir, run_name, 'best_model.pth')
    model = create_model(**model_config)
    model = model.to(device)
    
    # Dataset 
    #dataset = get_dataset(**data_config)
    loader, dataset = get_dataloader(data_config, batch_size=train_config.batch_size)
    
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    loss_fn = partial(get_ddpm_loss_fn, sampler=sampler, dataset=dataset,  train=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    
    # Train 
    for epoch in range(train_config.epochs):
        
        train_loss = train_epoch(loss_fn, model, loader["train"], optimizer, device)
        test_loss = test_epoch(loss_fn, model, loader["val"], device)
        logger.info(f"Epoch {epoch}, train_loss: {train_loss}, test_loss: {test_loss}")
        wandb.log({"train_loss": train_loss, "test_loss": test_loss})
        pass
if __name__ == "__main__":
    train_app()