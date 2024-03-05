
#########################################CONFIG 
from functools import partial
import os
import yaml
import torch 
import logging
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
PROJECTPATH = config["paths"]["PROJECTPATH"]
DATAPATH = config["paths"]["DATAPATH"]
SAVEPATH = config["paths"]["SAVEPATH"]
DEVICE = config["paths"]["DEVICE"]
logging.basicConfig(level=logging.INFO)
from utils import *
from data.dataset import *
from utils.logger import get_logger
from model.score_model import create_model
from sampling.cond_sampler import create_sampler
import wandb
import hydra 
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def train_app(cfg: DictConfig) -> None:
    arch = cfg.train["architecture"]
    mode = cfg.train["modes"]
    n_input = cfg.train["n_input"]
    run = wandb.init(
        project="denoising-backmapping",
        config=cfg.train.__dict__['_content']
    )
    diffusion_config = cfg.train.diffusion
    model_config = cfg.train.model
    data_config = cfg.train.data
    
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str) 
    
    # Load model
    #model_path = os.path.join(cfg.save_dir, run_name, 'best_model.pth')
    model = create_model(**model_config)
    model = model.to(device)
    model.train()
    
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
    
    dataset = get_dataset(**data_config)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=True)
    
    
    # Train 
    