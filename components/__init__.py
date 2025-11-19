from .base_sampler import BaseSampler

SAMPLER_REGISTRY = {
    'nint': 'components.nint_sampler.NINTSampler',
}

def create_sampler(strategy_name, configs, device, model=None, dataset=None):
    """Factory function to create sampler"""
    if strategy_name not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(SAMPLER_REGISTRY.keys())}")
    
    module_path, class_name = SAMPLER_REGISTRY[strategy_name].rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    sampler_class = getattr(module, class_name)
    
    return sampler_class(configs, device, model, dataset)

__all__ = ['BaseSampler', 'create_sampler', 'SAMPLER_REGISTRY']
