import os
import pickle
import datetime
import yaml
import shutil
import time
import torch
import numpy as np
import hydra
import logging
import wandb

from omegaconf import OmegaConf
from easydict import EasyDict
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

from components import create_sampler
from utils import seed_everything, get_dataset, get_model, prep_image_for_eval, save_image_to_wandb, Calc_LPIPS
from torch.func import functional_call, vjp, jvp
from torch.optim.lr_scheduler import MultiStepLR


log = logging.getLogger(__name__)


def load_config(config_file):
    configs = yaml.safe_load(open(config_file))
    return configs


def save_src_for_reproduce(configs, out_dir):
    if os.path.exists(os.path.join('outputs', out_dir, 'src')):
        shutil.rmtree(os.path.join('outputs', out_dir, 'src'))
    shutil.copytree('models', os.path.join('outputs', out_dir, 'src', 'models'))
    # dump config to yaml file
    OmegaConf.save(dict(configs), os.path.join('outputs', out_dir, 'src', 'config.yaml'))


def tint_data_with_samples(data, sample_idx, model_configs):
    """Relabel the data with given vis_label at the sample_idx indices."""
    if sample_idx is None:
        return None

    new_data = data.detach().clone()
    if model_configs.INPUT_OUTPUT.data_range == 1:
        vis_label = torch.tensor([0.75, 0.0, 0.0]).to(data.device)
    else:
        vis_label = torch.tensor([0.5, 0.0, 0.0]).to(data.device)
    if data.shape[-1] == 1:
        vis_label = vis_label[0]

    new_data[sample_idx] = torch.clamp(new_data[sample_idx] + vis_label, max=1.0)

    return new_data


def create_selected_points_image(sample_idx, H, W, C, model_configs, device):
    """Create an image showing only selected points on white background."""
    if sample_idx is None:
        return None

    # Start with white background
    background_value = 1.0  # White background for both data ranges

    new_data = torch.full((H * W, C), background_value, device=device)

    # Mark selected points with red
    if model_configs.INPUT_OUTPUT.data_range == 1:
        vis_label = torch.tensor([0.75, 0.0, 0.0]).to(device)
    else:
        vis_label = torch.tensor([0.5, 0.0, 0.0]).to(device)
    if C == 1:
        vis_label = vis_label[0]

    new_data[sample_idx] = vis_label

    return new_data


def compute_ntk_matrix_for_pixels(model, coords_subset, device):
    """
    Compute the NTK matrix for a subset of pixels using a simplified approach.

    Args:
        model: The neural network model
        coords_subset: Coordinates of the pixels to compute NTK for (shape: [n_pixels, coord_dim])
        device: Device to perform computation on

    Returns:
        ntk_matrix: NTK matrix of shape [n_pixels, n_pixels]
    """
    # Get the underlying model (handle DataParallel)
    underlying_model = model.module if hasattr(model, 'module') else model

    # Save original model state
    original_training_mode = underlying_model.training
    underlying_model.eval()  # Set to eval mode

    try:
        # Test forward pass to get dimensions
        with torch.no_grad():
            test_out = underlying_model(coords_subset)
            n_pixels, n_channels = test_out.shape

        # Compute NTK using gradient approach
        # For each pixel, compute its gradient w.r.t. parameters
        pixel_grads = []

        for pixel_idx in range(n_pixels):
            # Forward pass with gradient tracking
            coords_single = coords_subset[pixel_idx:pixel_idx+1]  # [1, coord_dim]

            underlying_model.zero_grad()
            output = underlying_model(coords_single)  # [1, n_channels]

            # Create a scalar loss from this pixel's output
            loss = output.sum()

            # Backward pass to get gradients
            loss.backward()

            # Collect gradients
            grad_vec = []
            for param in underlying_model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.detach().flatten())
            grad_vec = torch.cat(grad_vec)
            pixel_grads.append(grad_vec)

        # Compute NTK matrix as G @ G^T where G is the gradient matrix
        G = torch.stack(pixel_grads)  # [n_pixels, n_params]
        ntk_matrix = G @ G.T  # [n_pixels, n_pixels]

        return ntk_matrix.detach().cpu().numpy()

    finally:
        # Restore original model state
        underlying_model.train(original_training_mode)


def train(configs, model, dataset):
    # load configs
    train_configs = configs.TRAIN_CONFIGS
    dataset_configs = configs.DATASET_CONFIGS
    exp_configs = configs.EXP_CONFIGS
    network_configs = configs.NETWORK_CONFIGS
    model_configs = configs.model_config
    out_dir = train_configs.out_dir

    # Parse GPU devices configuration
    gpu_devices = train_configs.gpu_devices
    if gpu_devices == "all":
        available_gpus = list(range(torch.cuda.device_count()))
    else:
        available_gpus = [int(x.strip()) for x in gpu_devices.split(',') if x.strip()]

    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Using GPUs: {available_gpus}")

    # Set device for primary GPU (used for data and logging)
    if available_gpus:
        primary_device = f"cuda:{available_gpus[0]}"
    else:
        # Fallback to CUDA if available, otherwise CPU
        primary_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Primary device: {primary_device}")

    # Initialize LPIPS calculator
    lpips_calc = Calc_LPIPS(net_name="alex", device=primary_device)

    # optimizer and scheduler
    if exp_configs.optimizer_type == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=network_configs.lr)
    elif exp_configs.optimizer_type == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=network_configs.lr, momentum=0.9)
    if exp_configs.lr_scheduler_type == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=0)
    elif exp_configs.lr_scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-6)
    elif exp_configs.lr_scheduler_type == "multistep":
        # Reduce LR at 50% and 75% of training
        milestones = [train_configs.iterations // 2, int(train_configs.iterations * 0.75)]
        scheduler = MultiStepLR(opt, milestones=milestones, gamma=0.1)

    # prep model for training
    model.train()

    # Multi-GPU setup with DataParallel
    if len(available_gpus) > 1:
        print(f"Using DataParallel with {len(available_gpus)} GPUs: {available_gpus}")
        model = torch.nn.DataParallel(model, device_ids=available_gpus)
        model = model.to(primary_device)
    else:
        model = model.to(primary_device)

    # prepare training settings
    process_bar = tqdm(range(train_configs.iterations), disable=False)
    H, W, C = dataset.H, dataset.W, dataset.C
    best_psnr, best_ssim = 0, 0
    best_lpips = float('inf')  # Lower LPIPS is better
    best_pred = None

    # Local metric recorder
    metric_history = {
        'iterations': [],
        'timestamps': [],
        'psnr': [],
        'ssim': [],
        'lpips': [],
    }

    # get data
    coords, labels = dataset.get_data()
    coords, labels = coords.to(primary_device), labels.to(primary_device)
    # process training labels into ground truth image (for later use)
    ori_img = labels.view(H, W, C).cpu().detach().numpy()
    ori_img = (ori_img + 1) / 2 if model_configs.INPUT_OUTPUT.data_range == 1 else ori_img

    # Create sampler using strategy registry
    print(f"Creating sampler with strategy: {exp_configs.strategy_type}")
    sampler = create_sampler(
        exp_configs.strategy_type, 
        configs, 
        primary_device, 
        model,
        dataset
    )
    
    # Initialize strategies that need image data
    if hasattr(sampler, 'set_image'):
        sampler.set_image(labels.view(H, W, C))
    if hasattr(sampler, 'initialize_with_data'):
        sampler.initialize_with_data(labels.view(H, W, C))
    
    # sampling log
    psnr_milestone = False
    start_time = time.time()
    last_image_log_time = 0  # Track when images were last logged for time-based logging

    # train
    for step in process_bar:
        # Get samples using unified sampler interface
        sampled_coords, sampled_labels, full_preds, sampled_indices = sampler.sample(step, coords, labels, model)

        # subset inference for backprop
        sampled_preds = model(sampled_coords, None) 
        
        # MSE loss
        loss = ((sampled_preds - sampled_labels) ** 2).mean()

        # Compute and log NTK matrix at specified intervals
        if hasattr(configs.WANDB_CONFIGS, 'log_ntk') and configs.WANDB_CONFIGS.log_ntk and step % getattr(configs.WANDB_CONFIGS, 'ntk_log_interval', 100) == 0:
            try:
                # Define the two 3x3 patches by their top-left coordinates
                patches = [
                    (410, 35),   # First patch
                    (253, 175)   # Second patch
                ]

                for patch_idx, (start_y, start_x) in enumerate(patches):
                    pixel_indices = []

                    # Collect 3x3 = 9 pixel indices for this patch
                    for dy in range(3):
                        for dx in range(3):
                            y, x = start_y + dy, start_x + dx
                            if 0 <= y < H and 0 <= x < W:
                                pixel_idx = y * W + x
                                pixel_indices.append(pixel_idx)
                            else:
                                print(f"Warning: Pixel ({y},{x}) is out of bounds for patch starting at ({start_y},{start_x})")

                    if len(pixel_indices) == 9:
                        coords_subset = coords[pixel_indices].to(primary_device)
                        ntk_matrix = compute_ntk_matrix_for_pixels(model, coords_subset, primary_device)

                        print(f"\nStep {step}: NTK Matrix (9x9) for patch {patch_idx+1} at ({start_y},{start_x}):")
                        print("Matrix shape:", ntk_matrix.shape)
                        print("Matrix values:")
                        print(ntk_matrix)
                        print(".3f")
                        print(".6f")

                        # Optionally log to wandb with patch-specific names
                        if configs.WANDB_CONFIGS.use_wandb:
                            # Create a heatmap image of the NTK matrix
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(ntk_matrix, cmap='viridis', aspect='equal')
                            ax.set_title(f'NTK Matrix at Step {step} - Patch {patch_idx+1} ({start_y},{start_x})')
                            ax.set_xlabel('Pixel i')
                            ax.set_ylabel('Pixel j')
                            plt.colorbar(im, ax=ax)
                            wandb.log({f"ntk_matrix_step_{step}_patch_{patch_idx+1}": wandb.Image(fig)})
                            plt.close(fig)
                    else:
                        print(f"Warning: Could not collect 9 pixels for patch starting at ({start_y},{start_x})")

            except Exception as e:
                import traceback
                print(f"NTK computation failed at step {step}: {e}")
                print("Full traceback:")
                traceback.print_exc()

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # eval reconstruction (only if no_io is False)
        psnr_score = 0
        ssim_score = 0
        lpips_score = 0
        if not train_configs.no_io:
            # process reconstructed image for evaluation
            preds = prep_image_for_eval(full_preds, model_configs, H, W, C)

            # evaluation - compute metrics for progress bar and local logging
            # PSNR calculation for progress bar and local logging
            if configs.WANDB_CONFIGS.log_psnr or configs.METRIC_LOGGING.log_psnr:
                psnr_score = psnr_func(preds, ori_img, data_range=1)
            
            # SSIM calculation for progress bar and local logging
            if configs.WANDB_CONFIGS.log_ssim or configs.METRIC_LOGGING.log_ssim:
                ssim_score = ssim_func(preds, ori_img, channel_axis=-1, data_range=1)
            
            # Compute LPIPS for both W&B and local logging (expensive operation)
            if (configs.WANDB_CONFIGS.log_lpips or configs.METRIC_LOGGING.log_lpips) and configs.WANDB_CONFIGS.use_wandb:
                preds_tensor = torch.from_numpy(preds).permute(2, 0, 1).unsqueeze(0).float().to(primary_device)
                ori_tensor = torch.from_numpy(ori_img).permute(2, 0, 1).unsqueeze(0).float().to(primary_device)
                lpips_score = lpips_calc.compute_lpips(ori_tensor, preds_tensor)

            # (optional) squeeze image if it is GRAYSCALE
            if preds.shape[-1] == 1:
                preds = preds.squeeze(-1)

        # Local metric recording (respects METRIC_LOGGING toggles)
        if configs.METRIC_LOGGING.save_local:
            current_time = time.time() - start_time
            metric_history['iterations'].append(step)
            metric_history['timestamps'].append(current_time)
            metric_history['psnr'].append(psnr_score if configs.METRIC_LOGGING.log_psnr else None)
            metric_history['ssim'].append(ssim_score if configs.METRIC_LOGGING.log_ssim else None)
            metric_history['lpips'].append(lpips_score if configs.METRIC_LOGGING.log_lpips else None)

        # Local image logging (independent of W&B)
        current_time = time.time() - start_time
        time_interval_check = (configs.WANDB_CONFIGS.image_log_time_interval > 0 and
                             current_time - last_image_log_time >= configs.WANDB_CONFIGS.image_log_time_interval)
        
        # Save local reconstructed images
        if ((step%train_configs.save_interval==0 or time_interval_check) and 
            not train_configs.no_io and configs.WANDB_CONFIGS.log_reconstruction):
            os.makedirs(os.path.join('outputs', out_dir, 'images'), exist_ok=True)
            from PIL import Image as PILImage
            if preds.shape[-1] == 3:  # RGB
                img_pil = PILImage.fromarray((preds * 255).astype(np.uint8))
            else:  # Grayscale
                img_pil = PILImage.fromarray((preds.squeeze() * 255).astype(np.uint8))
            img_pil.save(os.path.join('outputs', out_dir, 'images', f'reconstruction_step_{step:05d}.png'))
            last_image_log_time = current_time

        # W&B logging (if enabled)
        if configs.WANDB_CONFIGS.use_wandb:
            log_dict = {
                        "loss": loss.item(),
                        "strategy": exp_configs.strategy_type
                        }
            
            # Add metrics only if enabled
            if configs.WANDB_CONFIGS.log_psnr:
                log_dict["psnr"] = psnr_score
            if configs.WANDB_CONFIGS.log_ssim:
                log_dict["ssim"] = ssim_score
            if configs.WANDB_CONFIGS.log_lpips:
                log_dict["lpips/iteration"] = lpips_score
            
            # Save ground truth image (only at 1st iteration)
            if step == 0 and not train_configs.no_io and configs.WANDB_CONFIGS.log_images and configs.WANDB_CONFIGS.log_gt_image:
                save_image_to_wandb(log_dict, ori_img, "GT", dataset_configs, H, W)

            # Save reconstructed image at save intervals or time intervals
            if ((step%train_configs.save_interval==0 or time_interval_check) and
                not train_configs.no_io and configs.WANDB_CONFIGS.log_images and configs.WANDB_CONFIGS.log_reconstruction):
                save_image_to_wandb(log_dict, preds, "Reconstruction", dataset_configs, H, W)

            # Save images showing sampled points (at save intervals or time intervals and if indices available)
            if (step%train_configs.save_interval==0 or time_interval_check) and sampled_indices is not None and configs.WANDB_CONFIGS.log_images:
                # Save tinted image showing sampled points in red (overlay on original)
                if configs.WANDB_CONFIGS.log_tinted:
                    flat_labels = labels.view(H * W, C)
                    tinted_flat = tint_data_with_samples(flat_labels, sampled_indices, model_configs)
                    tinted_img = tinted_flat.view(H, W, C)
                    tinted_img_np = prep_image_for_eval(tinted_img, model_configs, H, W, C)
                    save_image_to_wandb(log_dict, tinted_img_np, "Sampled_Points", dataset_configs, H, W)

                # Save selected points only image (no background)
                if configs.WANDB_CONFIGS.log_selected:
                    selected_flat = create_selected_points_image(sampled_indices, H, W, C, model_configs, primary_device)
                    selected_img = selected_flat.view(H, W, C)
                    selected_img_np = prep_image_for_eval(selected_img, model_configs, H, W, C)
                    save_image_to_wandb(log_dict, selected_img_np, "Selected_Points_Only", dataset_configs, H, W)

            # if PSNR > 30, log the step (only if PSNR logging is enabled)
            if configs.WANDB_CONFIGS.log_psnr and not psnr_milestone and psnr_score > 30:
                psnr_milestone = True
                wandb.log({"PSNR Threshold": step}, step=step)

            # log to wandb
            wandb.log(log_dict, step=step)
            
            # Time-based logging with elapsed time as x-axis (only if enabled)
            if configs.WANDB_CONFIGS.log_time_metrics:
                elapsed_sec = time.time() - start_time
                time_log_dict = {
                    "elapsed_sec": elapsed_sec,
                }
                if configs.WANDB_CONFIGS.log_psnr:
                    time_log_dict["psnr/time"] = psnr_score
                if configs.WANDB_CONFIGS.log_ssim:
                    time_log_dict["ssim/time"] = ssim_score
                if configs.WANDB_CONFIGS.log_lpips:
                    time_log_dict["lpips/time"] = lpips_score
                # Let elapsed_sec be the custom x-axis (defined via define_metric)
                # Do not pass a float step, which W&B expects to be int
                wandb.log(time_log_dict)

        # Save model weights if it has the best PSNR so far
        if psnr_score > best_psnr:
            best_psnr, best_ssim = psnr_score, ssim_score
            best_lpips = lpips_score  # Track LPIPS when PSNR improves
            best_pred = preds
            if not train_configs.no_io:
                torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))

        # Print progress every 100 steps
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{train_configs.iterations} | Loss: {loss.item():.4f} | PSNR: {psnr_score:.2f} | SSIM: {ssim_score*100:.2f} | Time: {elapsed:.1f}s")

        # update progress bar
        process_bar.set_description(f"[{exp_configs.strategy_type}] psnr: {psnr_score:.2f}, ssim: {ssim_score*100:.2f}, loss: {loss.item():.4f}")
    
    # wrap up training
    print("Training finished!")
    print(f"Strategy: {exp_configs.strategy_type}")
    print(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}, lpips: {best_lpips:.4f}")
    # W&B logging of final step
    if configs.WANDB_CONFIGS.use_wandb:
        if not train_configs.no_io:
            best_pred = best_pred.squeeze(-1) if dataset_configs.color_mode == 'L' else best_pred
            wandb.log(
                    {
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "best_lpips": best_lpips,
                    "best_pred": wandb.Image(Image.fromarray((best_pred*255).astype(np.uint8), mode=dataset_configs.color_mode)),
                    }, 
                step=step)
        else:
            wandb.log(
                    {
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "best_lpips": best_lpips
                    },
                step=step)
        wandb.finish()
    log.info(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}, lpips: {best_lpips:.4f}")
    
    # save model
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))

    # Save local metric history
    if configs.METRIC_LOGGING.save_local:
        os.makedirs(os.path.join('outputs', out_dir), exist_ok=True)
        # Prefer W&B run tag for filename; fallback to timestamp
        if 'run_tag' not in locals() or run_tag is None:
            from datetime import datetime
            run_tag = f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metric_file = os.path.join('outputs', out_dir, f'metrics_history_{run_tag}.pkl')
        with open(metric_file, 'wb') as f:
            pickle.dump(metric_history, f)
        print(f"Metric history saved to: {metric_file}")

    return best_psnr, best_ssim


@hydra.main(version_base=None, config_path='config', config_name='train_image')
def main(configs):
    configs = EasyDict(configs)

    # Seed python, numpy, pytorch
    seed_everything(configs.TRAIN_CONFIGS.seed)
    # Saving config and settings for reproduction
    save_src_for_reproduce(configs, configs.TRAIN_CONFIGS.out_dir)

    # model configs
    configs.model_config.INPUT_OUTPUT.data_range = configs.NETWORK_CONFIGS.data_range
    configs.model_config.INPUT_OUTPUT.coord_mode = configs.NETWORK_CONFIGS.coord_mode
    if configs.model_config.name == "FFN":
        configs.model_config.NET.rff_std = configs.NETWORK_CONFIGS.rff_std
    if hasattr(configs.NETWORK_CONFIGS, "num_layers"):
        configs.model_config.NET.num_layers = configs.NETWORK_CONFIGS.num_layers
    if hasattr(configs.NETWORK_CONFIGS, "dim_hidden"):
        configs.model_config.NET.dim_hidden = configs.NETWORK_CONFIGS.dim_hidden

    # model and dataloader
    print(f"Dataset: {configs.DATASET_CONFIGS}")
    print(f"Strategy: {configs.EXP_CONFIGS.strategy_type}")
    dataset = get_dataset(configs.DATASET_CONFIGS, configs.model_config.INPUT_OUTPUT)
    model = get_model(configs.model_config, dataset)
    print(f"Start experiment: {configs.TRAIN_CONFIGS.out_dir}")
    
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"No. of parameters: {n_params}")
    
    # wandb
    if configs.WANDB_CONFIGS.use_wandb:
        wandb.init(
            project=configs.WANDB_CONFIGS.wandb_project,
            entity=configs.WANDB_CONFIGS.wandb_entity,
            config=configs,
            group=configs.WANDB_CONFIGS.group,
            name=configs.TRAIN_CONFIGS.out_dir,
        )

        wandb.run.summary['n_params'] = n_params
        # Capture the run directory name to mirror W&B file naming
        try:
            run_tag = os.path.basename(wandb.run.dir)
        except Exception:
            run_tag = None
        
        # Define custom x-axis for time-based metrics (only if enabled)
        if configs.WANDB_CONFIGS.log_time_metrics:
            wandb.define_metric("elapsed_sec")
        if configs.WANDB_CONFIGS.log_time_metrics and configs.WANDB_CONFIGS.log_psnr:
            wandb.define_metric("psnr/time", step_metric="elapsed_sec")
        if configs.WANDB_CONFIGS.log_time_metrics and configs.WANDB_CONFIGS.log_ssim:
            wandb.define_metric("ssim/time", step_metric="elapsed_sec")
        if configs.WANDB_CONFIGS.log_time_metrics and configs.WANDB_CONFIGS.log_lpips:
            wandb.define_metric("lpips/time", step_metric="elapsed_sec")
        if configs.WANDB_CONFIGS.log_lpips:
            wandb.define_metric("lpips/iteration")

    # train
    psnr, ssim = train(configs, model, dataset)

    return psnr

if __name__=='__main__':
    main()