import torch

class BaseSampler:
    """Unified interface for all sampling strategies"""
    
    def __init__(self, configs, device, model=None, dataset=None):
        self.configs = configs
        self.device = device
        self.model = model
        self.dataset = dataset
    
    def sample(self, step, coords, labels, model):
        """
        Sample coordinates and labels for training.
        
        Args:
            step: Current training step
            coords: Full coordinate tensor
            labels: Full label tensor
            model: Neural network model
            
        Returns:
            sampled_coords: Selected coordinates
            sampled_labels: Selected labels
            full_preds: Full prediction on all coordinates (for logging)
            sampled_indices: Indices of sampled points (for visualization/tinting)
        """
        raise NotImplementedError
    
    def get_ratio(self):
        """Get current sampling ratio"""
        return self.configs.EXP_CONFIGS.mt_ratio
    
    def get_interval(self):
        """Get current interval (for strategies with variable intervals)"""
        return 1
