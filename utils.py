import random 
import wandb
import numpy as np
import torch
from dataset import *
import math


def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed) # for random partitioning
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def prep_audio_for_eval(audio, config, t, c):
    if config.INPUT_OUTPUT.data_range == 1:
        audio = audio.clamp(-1, 1).view(t, c)       # clip to [-1, 1]
    else: # data range == 0
        audio = audio.clamp(0, 1).view(t, c)       # clip to [0, 1]
        audio = audio*2 - 1                         # [0, 1] -> [-1, 1]

    audio = audio.flatten().cpu().detach().numpy()
    return audio

def prep_image_for_eval(image, config, h, w, c, reshape=True):
    if config.INPUT_OUTPUT.data_range == 1:
        image = image.clamp(-1, 1)      # clip to [-1, 1]
        image = (image + 1) / 2         # [-1, 1] -> [0, 1]
    else:
        image = image.clamp(0, 1)       # clip to [0, 1]

    if reshape:
        image = image.view(h, w, c)
    image = image.cpu().detach().numpy()

    return image


def save_image_to_wandb(wandb_dict, image, label, dataset_configs, h, w):
    wandb_img = Image.fromarray((image *255).astype(np.uint8), mode=dataset_configs.color_mode)
    if wandb_img.size[0] > 512:
        wandb_img = wandb_img.resize((512, int(512*h/w)), Image.LANCZOS)
    wandb_dict[label] = wandb.Image(wandb_img)


def compute_si_snr(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute scale-invariant signal-to-noise ratio (SI-SNR) between gt and pred.
    Inputs are 1D numpy arrays.
    """
    # Ensure 1D float arrays
    gt = np.asarray(gt, dtype=np.float32).flatten()
    pred = np.asarray(pred, dtype=np.float32).flatten()
    # Zero-mean
    gt = gt - np.mean(gt)
    pred = pred - np.mean(pred)
    # Avoid division by zero
    gt_energy = np.sum(gt ** 2) + 1e-8
    # Projection of pred onto gt
    scale = np.sum(pred * gt) / gt_energy
    s_target = scale * gt
    e_noise = pred - s_target
    # Power
    target_power = np.sum(s_target ** 2) + 1e-8
    noise_power = np.sum(e_noise ** 2) + 1e-8
    si_snr = 10.0 * math.log10(target_power / noise_power)
    return float(si_snr)


def compute_pesq(gt: np.ndarray, pred: np.ndarray, sr: int = 16000) -> float:
    """
    Compute PESQ (wideband) between gt and pred.
    Requires `pesq` package. Signals expected in range [-1, 1].
    """
    try:
        from pesq import pesq
    except Exception as e:
        raise ImportError("pesq package is required for PESQ metric. Please install `pesq`.") from e
    gt = np.asarray(gt, dtype=np.float32).flatten()
    pred = np.asarray(pred, dtype=np.float32).flatten()
    # Clip to valid range
    gt = np.clip(gt, -1.0, 1.0)
    pred = np.clip(pred, -1.0, 1.0)
    return float(pesq(sr, gt, pred, 'wb'))


def compute_stoi(gt: np.ndarray, pred: np.ndarray, sr: int = 16000, extended: bool = False) -> float:
    """
    Compute STOI between gt and pred.
    Requires `pystoi`. Set extended=False for classic STOI.
    """
    try:
        from pystoi.stoi import stoi
    except Exception as e:
        raise ImportError("pystoi package is required for STOI metric. Please install `pystoi`.") from e
    gt = np.asarray(gt, dtype=np.float32).flatten()
    pred = np.asarray(pred, dtype=np.float32).flatten()
    return float(stoi(gt, pred, sr, extended=extended))


def get_dataset(dataset_configs, input_output_configs):
    if dataset_configs.data_type == "image":
        dataset = ImageFileDataset(dataset_configs, input_output_configs)
    elif dataset_configs.data_type == "audio":
        dataset = AudioFileDataset(dataset_configs, input_output_configs) 
    elif dataset_configs.data_type == "sdf":
        dataset = MeshSDF(dataset_configs, input_output_configs)
    elif dataset_configs.data_type == "megapixel":
        dataset = BigImageFileDataset(dataset_configs, input_output_configs)
    else:
         raise NotImplementedError(f"Dataset {dataset_configs.data_type} not implemented")
    return dataset


def get_model(model_configs, dataset):
    if model_configs.name == 'SIREN':
        from models.siren import Siren
        model = Siren(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            siren_configs=model_configs
        )
    elif model_configs.name == 'FFN':
        from models.ffn import FFN
        model = FFN(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            ffn_configs=model_configs
        )
    elif model_configs.name == "WIRE":
        from models.wire import Wire
        model = Wire(
           in_features=dataset.dim_in, 
           out_features=dataset.dim_out,
           wire_configs=model_configs
        )
    elif model_configs.name == "MLP":
        from models.mlp import MLP
        model = MLP(
           dim_in=dataset.dim_in,
           dim_out=dataset.dim_out,
           mlp_configs=model_configs
        )
    elif model_configs.name == 'PEMLP':
        from models.pemlp import PEMLP
        model = PEMLP(
           dim_in=dataset.dim_in,
           dim_out=dataset.dim_out,
           pemlp_configs=model_configs
        )
    elif model_configs.name == 'FINER':
        from models.finer import Finer
        model = Finer(
           dim_in=dataset.dim_in,
           dim_out=dataset.dim_out,
           finer_configs=model_configs
        )
    elif model_configs.name == 'GAUSS':
        from models.gauss import Gauss
        model = Gauss(
           dim_in=dataset.dim_in,
           dim_out=dataset.dim_out,
           gauss_configs=model_configs
        )
    else:
        raise NotImplementedError(f"Model {model_configs.name} not implemented")
            
    return model


class Calc_LPIPS(object):
    """LPIPS (Learned Perceptual Image Patch Similarity) calculator"""
    def __init__(self, net_name="alex", device="cpu"):
        import lpips
        assert net_name in ['alex', 'vgg']
        self.lpips_net = lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

    def _norm(self, data):
        """Normalize data to [-0.5, 0.5] range"""
        _min = torch.min(data)
        _max = torch.max(data)
        data = (data - _min) / (_max - _min)  # [0,1]
        data = (data - 0.5) * 2  # [-0.5, 0.5]
        return data
    
    def compute_lpips(self, gt, pred):
        """Compute LPIPS between ground truth and prediction
        
        Args:
            gt: Ground truth tensor
            pred: Prediction tensor
            
        Returns:
            LPIPS score (scalar)
        """
        normed_gt = self._norm(gt).squeeze(0)
        normed_pred = self._norm(pred).squeeze(0)
        return self.lpips_net(normed_gt, normed_pred).detach().item()