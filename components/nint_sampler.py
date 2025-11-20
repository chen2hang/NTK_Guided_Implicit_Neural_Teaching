import torch
from .base_sampler import BaseSampler
from src.nint import NINT

class NINTSampler(BaseSampler):
    """Wrapper for NINT (NTK-Guided Implicit Neural Teaching) sampling strategy"""
    
    def __init__(self, configs, device, model, dataset):
        super().__init__(configs, device, model, dataset)

        # Handle different data dimensionalities
        if hasattr(dataset, 'H') and hasattr(dataset, 'W') and hasattr(dataset, 'C'):
            # 2D image data
            self.H = dataset.H
            self.W = dataset.W
            self.C = dataset.C
            data_shape = (self.H, self.W, self.C)
        else:
            # 1D audio data or other dimensionality
            total_samples = len(dataset.get_data()[0])  # Get total number of coordinates
            data_shape = (total_samples,)
            self.H = self.W = self.C = None

        # Get all NINT parameters from the consolidated nint config section
        nint_config = configs.nint
        batch_size_scheduler = nint_config.get('batch_size_scheduler', 'step')
        mt_ratio = nint_config.get('mt_ratio', 0.2)
        top_k = nint_config.get('top_k', True)
        sample_interval = nint_config.get('sample_interval', 'dense')

        # Initialize NINT
        self.nint = NINT(
            model,
            configs.TRAIN_CONFIGS.iterations,
            data_shape,
            batch_size_scheduler,    # Now from nint config
            sample_interval,         # Now from nint config
            mt_ratio,                # Now from nint config
            top_k,                   # Now from nint config
            save_samples_path=None,
            save_losses_path=None,
            save_name=None,
            save_interval=configs.TRAIN_CONFIGS.save_interval
        )
    
    def sample(self, step, coords, labels, model):
        """Use NINT's original sampling logic"""
        sampled_x, sampled_y, preds = self.nint.sample(step, coords, labels)
        # Get indices from NINT if available, otherwise use None
        indices = getattr(self.nint, 'sampled_indices', None)
        return sampled_x, sampled_y, preds, indices
    
    def get_ratio(self):
        """Get current ratio from NINT"""
        return self.nint.get_ratio()
    
    def get_interval(self):
        """Get current interval from NINT"""
        return self.nint.get_interval()

