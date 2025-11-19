import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple
from pathlib import Path
import math

from .scheduler import *
from .sampler import mt_sampler, save_samples, save_losses, hierarchical_sampler
from .strategy import strategy_factory
from torch.func import functional_call, vjp, jvp


class NINT:
    """
    Wrapper class for the NINT (NTK-Guided Implicit Neural Teaching) algorithm.
    
    NINT uses NTK-weighted importance scores to guide adaptive sampling during neural 
    network training. It combines:
    - Residual-based importance (high loss = high importance)
    - NTK-weighted importance (parameter sensitivity)
    - Random sampling (exploration)
    in a hierarchical sampling framework.

    Args:
        model (torch.nn.Module): The model to be trained.
        iters (int): The number of iterations to train the model.
        data_shape (Union[Tuple[int], List[int]]): The shape of the input data.
        scheduler (str, optional): The type of scheduler to use. Defaults to "step".
            Types: "step", "linear", "cosine", "reverse-cosine", "constant"
        strategy (str, optional): The type of strategy to use. Defaults to "dense".
            Types: "incremental", "reverse-incremental", "exponential", "dense", "void"
        starting_ratio (float, optional): The starting ratio for the NINT algorithm. Defaults to 0.2.
        top_k (bool, optional): Whether to use top-k sampling. Defaults to True.
        save_samples_path (Path, optional): The path to save the samples. Defaults to Path("logs/sampling").
        save_losses_path (Path, optional): The path to save the losses. Defaults to Path("logs/losses").
        save_name (str, optional): The name to save the samples and losses. Defaults to None.
        save_interval (int, optional): The interval to save the samples and losses. Defaults to 1000.

    Key Features:
        - NTK-weighted importance scoring using efficient VJP/JVP operations
        - Hierarchical sampling combining NTK, residual, and random components
        - Adaptive NTK computation that decays over training
        - Efficient gradient caching for multi-GPU support
    """
    def __init__(self,
                 model: torch.nn.Module,
                 iters: int,
                 data_shape: Union[Tuple[int], List[int]],
                 scheduler: str="step",
                 strategy: str="dense",
                 starting_ratio: float=0.2,
                 top_k: bool=True,
                 save_samples_path: Path=Path("logs/sampling"),
                 save_losses_path: Path=Path("logs/losses"),
                 save_name: str=None,
                 save_interval: int=1000):
        
        self.model = model
        self.scheduler = mt_scheduler_factory(scheduler)
        self.strategy = strategy_factory(strategy)
        
        self.iters = iters
        self.starting_ratio = starting_ratio
        self.ratio = starting_ratio
        self.mt_interval = None
        self.top_k = top_k
        if len(data_shape) == 3:
            self.H, self.W, self.C = data_shape
        else:
            self.H = None
            self.W = None
            self.C = None

        self.preds = None   
        self.sampled_x = None
        self.sampled_y = None
        self.sampled_indices = None

        self.save_interval = save_interval

        self.recal_NTK = True
        self.use_counter = 1
        # best: ite 10, NTK 0.3, random 0.7
        self.NTK_last_ite = 10
        self.NTK_portion_init = 0.3

        self.NTK_portion = None
        self.Residual_portion = None
        self.Random_portion = 0.7

        if save_samples_path is not None:
            self.sampling_path = Path(save_samples_path) if type(save_samples_path) is not Path else save_samples_path
            self.sampling_path.mkdir(parents=True, exist_ok=True)
        else:
            self.sampling_path = None
        if save_losses_path is not None:
            self.loss_path = Path(save_losses_path) if type(save_losses_path) is not Path else save_losses_path
            self.loss_path.mkdir(parents=True, exist_ok=True)
        else:
            self.loss_path = None
        self.save_name = f"mt{starting_ratio}_{strategy}_{scheduler}_topk{top_k}" if save_name is None else save_name

        self.save_sample_path = None
        self.save_loss_path = None
        self.save_tint_path = None

        self.sampling_history = dict()
        self.loss_history = dict()

        # Cache NTK parameter and buffer structures once to avoid per-call rebuild
        underlying_model = self.model.module if hasattr(self.model, 'module') else self.model
        self._ntk_params = {k: v for k, v in underlying_model.named_parameters() if v.requires_grad}
        self._ntk_buffers = dict(underlying_model.named_buffers())

    def sample(self, iter: int, x: torch.tensor, y: torch.tensor):
        """
        Perform the NINT sampling.

        Args:
            iter (int): The current iteration.
            x (torch.tensor): The input data.
            y (torch.tensor): The target data.
        
        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]: sampled_x, sampled_y, preds
        """
        # get the sampling ratio for this step
        self.ratio = self.scheduler(iter, self.iters, self.starting_ratio)
        # get the sampling intervals for this step, and a bool that determines whether we should sample at this step
        mt, self.mt_intervals = self.strategy(iter, self.iters)
        # perform the sampling
        if mt:
            # Forward pass: enable grads only when NTK will be refreshed; otherwise use inference_mode
            self.model.zero_grad(set_to_none=True)
            if self.recal_NTK:
                with torch.enable_grad():
                    self.preds = self.model(x)
            else:
                with torch.inference_mode():
                    self.preds = self.model(x)

            # compute NTK-weighted importance scores per sample/pixel
            residual = (self.preds - y)

            # NINT: hierarchical sampling with NTK weighting
            importance = None
            if self.recal_NTK:
                self.NTK_portion = self.NTK_portion_init * math.exp(- 2.0 * self.use_counter // self.NTK_last_ite)
                if self.NTK_portion >= 1e-3:
                    importance = self._ntk_weighted_scores_pixelwise(x, residual)
                else:
                    self.recal_NTK = False

                self.Residual_portion = 1.0 - self.NTK_portion - self.Random_portion

            self.sampled_x, self.sampled_y, idx, dif = hierarchical_sampler(
                x, y, self.preds, self.ratio, self.top_k,
                importance=importance,
                cal_NTK=self.recal_NTK,
                cached_ntk_indices=getattr(self, "_cached_ntk_indices", None),
                NTK_portion = self.NTK_portion,
                R_portion = self.Residual_portion
            )

            # Cache NTK indices when refreshed
            if self.recal_NTK:
                n_total = len(idx)
                n_ntk = int(n_total * self.NTK_portion)
                self._cached_ntk_indices = idx[:n_ntk].detach().clone()

            self.use_counter += 1
            self.recal_NTK = True if self.use_counter % self.NTK_last_ite == 0 else False
            
            # Store indices for visualization
            self.sampled_indices = idx


            if iter % self.save_interval == 0:
                # save the sampling history
                if self.sampling_path is not None:
                    self.save_sample_path = self.sampling_path / f"{self.save_name}_samples.pkl"
                    self.save_tint_path = self.sampling_path / f"{self.save_name}_tint_{iter}.png"

                    # save the samples
                    save_samples(self.sampling_history, iter, self.iters, self.sampled_x, self.save_sample_path)

                    # save the tinted image (i.e. gt data + sampled pixels highlighted)
                    if self.H is not None and self.W is not None and self.C is not None:
                        tinted_x = self._tint_data_with_samples(y, idx)
                        tinted_img = self._preprocess_img(tinted_x, self.H, self.W, self.C)
                        self._save_image(tinted_img, self.save_tint_path, self.H, self.W, color_mode="L" if self.C == 1 else "RGB")
                # save the losses/scores used for ranking
                if self.loss_path is not None:
                    self.save_loss_path = self.loss_path / f"{self.save_name}_losses.pkl"
                    save_losses(self.loss_history, iter, self.iters, dif, self.save_loss_path)

        # this condition checks if we are not doing NINT at all. If so, we just return the data as is
        elif not mt and self.mt_intervals is None:
            self.sampled_x = x
            self.sampled_y = y
            self.preds = y

        # if we are not doing NINT at this particular step (i.e. there is a non-zero mt_interval)
        # then, we reuse the samples from the previous step
        else:
            self.sampled_x = self.sampled_x
            self.sampled_y = self.sampled_y
            self.preds = self.preds

        return self.sampled_x, self.sampled_y, self.preds

    def get_ratio(self):
        return self.ratio
    
    def get_interval(self):
        return self.mt_intervals

    def get_saved_samples_path(self):
        return self.save_sample_path

    def get_saved_losses_path(self):
        return self.save_loss_path

    def get_saved_tint_path(self):
        return self.save_tint_path

    def _ntk_weighted_scores_pixelwise(self, x, residual):
        """
        Compute per-sample/pixel importance via w = K R with K = J J^T, where
        J is the Jacobian of model outputs w.r.t. parameters. We avoid forming K
        explicitly by computing w = J (J^T R) using two efficient autodiff ops:
        first a vector-Jacobian product (VJP) to get v = J^T R, then a Jacobian-
        vector product (JVP) to get w = J v.

        For multi-channel outputs, we reshape w back to [N, C] and take the
        L2 norm over channels to produce one score per sample/pixel (length N).

        Args:
            x (torch.Tensor): input batch of size [N, ...]
            residual (torch.Tensor): (pred - y) shaped [N, C] or [N]

        Returns:
            torch.Tensor: scores of shape [N]
        """
        # Use cached trainable params and buffers (handle DataParallel already)
        params = self._ntk_params
        buffers = self._ntk_buffers

        def f(p):
            underlying_model = self.model.module if hasattr(self.model, 'module') else self.model
            out = functional_call(underlying_model, (p, buffers), (x,))
            return out.reshape(-1)

        # Residual flattened to match f(p) output shape; detach to keep graph small
        R_flat = residual.reshape(-1).detach()

        # First VJP: v = J^T R
        _, pullback = vjp(f, params)
        v = pullback(R_flat)[0]

        # Then JVP: w = J v
        w_flat, _ = jvp(f, (params,), (v,))

        # Reshape w back to [N, C] and reduce over channels
        N = residual.shape[0]
        C = residual.shape[1] if residual.dim() > 1 else 1
        w = w_flat.view(N, C)
        # Squared L2 is sufficient for ranking and avoids sqrt
        scores = (w * w).sum(dim=1)

        return scores.detach()

    def _tint_data_with_samples(self, data, sample_idx, tint_color: List[float]=[0.5, 0.0, 0.0]):
        """Relabel the data with given vis_label at the sample_idx indices."""
        if sample_idx is None: 
            return None
        
        new_data = data.detach().clone()
        vis_label = torch.tensor(tint_color).to(data.device)
        if data.shape[-1] == 1:
            vis_label = vis_label[0]

        new_data[sample_idx] = torch.clamp(new_data[sample_idx] + vis_label, max=1.0)

        return new_data
    
    def _preprocess_img(self, image, h, w, c):
        """Preprocess the image for saving."""
        if torch.min(image) < 0:
            image = image.clamp(-1, 1).view(h, w, c)       # clip to [-1, 1]
            image = (image + 1) / 2                        # [-1, 1] -> [0, 1]
        else:
            image = image.clamp(0, 1).view(h, w, c)       # clip to [0, 1]

        image = image.cpu().detach().numpy()

        return image
    
    def _save_image(self, img, path, h, w, color_mode="RGB"):
        """Save the image to the given path."""
        img = Image.fromarray((img *255).astype(np.uint8), mode=color_mode)
        if img.size[0] > 512:
            img = img.resize((512, int(512*h/w)), Image.LANCZOS)
        img.save(path)
        print(f"Image saved to {path}")

