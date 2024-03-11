from functools import partial
import os
import torch
from sampling.conditioning import get_conditioning_method
from sampling.measurements import get_noise, get_operator
from model.score_model import create_model
from sampling.cond_sampler import create_sampler
from data.dataset import get_dataset, get_dataloader
from utils.logger import get_logger
import wandb
import hydra 
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def eval(cfg: DictConfig):
    # log to wandb and  get train config 
    run_id = os.path.join("denoising-backmapping", cfg.eval["run_id"])
    api = wandb.Api()
    run = api.run(run_id)
    run_name = run.name
    run_config = run.config
    model_config = run.config.model
    diffusion_config = run.config.diffusion
    measure_config = run.config.measurement
    
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model_path = os.path.join(cfg.save_dir, run_name, 'best_model.pth')
    model = create_model(**model_config, model_path=model_path)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(cfg.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for CG Conf {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # Exception In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

        #plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        #plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        #plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

if __name__ == '__main__':
    eval()