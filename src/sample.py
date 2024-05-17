from functools import partial
import os
import torch
from sampling.conditioning import get_conditioning_method
from sampling.measurements import get_noise, get_operator
from model.score_model import create_model
from sampling.cond_sampler import create_sampler
from data.dataset import get_dataloader
from utils.logger import get_logger
import wandb
import hydra 
import logging
from omegaconf import DictConfig
import ast

@hydra.main(version_base=None, config_path="config", config_name="base_config")
def eval(cfg: DictConfig):
    # log to wandb and  get train config 
    run_id = os.path.join("denoising-backmapping", cfg.eval['run_id'])
    api = wandb.Api()
    run = api.run(run_id)
    run_name = run.name
    model_config = ast.literal_eval(run.config["model"])
    diffusion_config = ast.literal_eval(run.config["diffusion"])
    measure_config = ast.literal_eval(run.config["measurement"])
    task_config = ast.literal_eval(run.config["conditioning"])
    data_config = ast.literal_eval(run.config["data"])
    
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{cfg.eval['gpu_id']}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    savepath = os.path.join(cfg.eval['save_dir'], run_name)
    model_path = os.path.join(savepath, 'best_model.h5')
    model_config['model_path'] = model_path
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare dataloader
    loader, dataset = get_dataloader(data_config, 
                                batch_size=1,
                                modes=('val','test'),
                                )
    
     # Prepare Operator and noise
    measure_config['operator']['n_atoms'] = 400
    measure_config['operator']['n_monomers'] = 10
    measure_config['operator']['n_polymers'] = 10
    operator = get_operator(device=device, operator=(dataset.com_matrix, dataset.atom_monomer_id), **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_method_name = 'ps'
    cond_method = get_conditioning_method(cond_method_name, operator, noiser, **task_config)
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['method']}")
   
    # Load diffusion sampler
    diffusion_config['timestep_respacing'] = 1000
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(savepath, measure_config['operator']['name']+"_"+cond_method_name+"_scale_1_opt")
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Exception) In case of inpainting, we need to generate a mask 
    #if measure_config['operator']['name'] == 'inpainting':
    #    mask_gen = mask_generator(
    #       **measure_config['mask_opt']
    #    )
    k = 0
    # Do Inference
    for i, conf in enumerate(loader['test']):
        
        logger.info(f"Inference for CG Conf {i+1}")
        fname = str(i).zfill(5) + '.pt'
        
        if i == k:
            # Exception In case of inpainging,
            #if measure_config['operator'] ['name'] == 'inpainting':
                #mask = mask_gen(ref_img)
                #mask = mask[:, 0, :, :].unsqueeze(dim=0)
                #measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
                #sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
                

                # Forward measurement model (Ax + n)
                #y = operator.forward(ref_img, mask=mask)
                #y_n = noiser(y)

            #else: 
            
            # Forward measurement model (Ax + n)
            conf = conf.to(device)
            operator.inject_std(conf.cg_std_dist)
            #y = operator.forward(conf.cg_dist / operator.cg_std_dist[None,...], conf.cg_pos)
            #y = y.to(device)
            #y_n = noiser(y)
            
            # Sampling noised initial distances
            input_conf = conf.conf
            input_cg_pos = conf.cg_pos
            x_start = torch.randn(conf.conf.shape, dtype=torch.float64 ,device=device)
            #sigma = conf.cg_std_dist.expand(x_start.shape[0])
            conf.cg_perb_dist = x_start.requires_grad_(True)
            
            sample = sample_fn(data=conf.to(device), dataset=dataset ,measurement=conf.cg_pos, record=True, save_root=out_path)

            
            torch.save(input_conf, os.path.join(out_path, 'input', fname))
            torch.save(input_cg_pos, os.path.join(out_path, 'label', fname))
            torch.save(sample.perb_pos, os.path.join(out_path, 'recon', fname))
        else:
            pass
       
       

if __name__ == '__main__':
    eval()