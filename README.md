# NINT: NTK-Guided Implicit Neural Teaching

## Submission Info

- Prepared for CVPR 2026 submission #5792
- Package is anonymized and ready for open-box use as supplementary material

## Overview

This is an implementation of NINT for image reconstruction using implicit neural representations (INRs). NINT adaptively samples pixels during training by combining:

- **Residual-based importance** - prioritizes high-error regions
- **NTK-weighted importance** - weights by parameter sensitivity  
- **Random exploration** - maintains coverage
- **Hierarchical scheduling** - dynamically balances components

## Quick Start

### 1. Setup Environment

```bash
conda env create -f environment.yml
conda activate nint_env
```

### 2. Train

```bash
python train_image.py
```

The script trains a SIREN model on `images/01.png` by default and saves results to `outputs/`.

### 3. Custom Configuration

Edit `config/train_image.yaml`:

```yaml
DATASET_CONFIGS:
  file_path: path/to/your/image.png

TRAIN_CONFIGS:
  iterations: 5000
  lr: 1e-3

nint:
  mt_ratio: 0.2                    # Sampling ratio (fraction of pixels)
  scheduler_type: constant         # Options: constant, step, linear, cosine
  nmt_profile_strategy: dense      # Options: dense, incremental, reverse-incremental
```

## File Structure

```
├── src/
│   ├── nint.py              # NINT algorithm (main)
│   ├── scheduler.py         # MT ratio schedulers
│   ├── sampler.py           # Sampling utilities
│   └── strategy.py          # Sampling strategies
├── components/
│   ├── nint_sampler.py      # NINT sampler wrapper
│   ├── base_sampler.py      # Base sampler class
│   ├── lpips.py & ssim.py   # Perceptual metrics
├── models/
│   └── siren.py             # SIREN model
├── config/
│   └── train_image.yaml     # Configuration
├── train_image.py           # Training script
├── dataset.py               # Image handling
└── utils.py                 # Utilities
```

## Output

Training outputs saved to `outputs/siren_sampling/`:

```
├── model.pth                # Best model weights
├── hydra_configs/           # Configuration backups
└── src/                     # Source code snapshot
```

## Metrics

The training script logs:
- **MSE loss** - reconstruction error
- **PSNR** - Peak Signal-to-Noise Ratio
- **SSIM** - Structural Similarity
- **LPIPS** - Learned Perceptual Image Patch Similarity (optional)

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `mt_ratio` | Fraction of pixels sampled per iteration (0.0-1.0) |
| `scheduler_type` | How mt_ratio changes: constant, step, linear, or cosine |
| `nmt_profile_strategy` | When to trigger sampling: dense, incremental, or reverse-incremental |
| `top_k` | Use importance-weighted sampling (1) or random (0) |

## Advanced Configuration

### Enable W&B Logging (Optional)

```yaml
WANDB_CONFIGS:
  use_wandb: 1
  wandb_project: YOUR_PROJECT
  wandb_entity: YOUR_USERNAME
```

Then run: `wandb login && python train_image.py`

### Troubleshooting

**Out of Memory:**
```yaml
nint:
  mt_ratio: 0.1  # Reduce sampling ratio
TRAIN_CONFIGS:
  iterations: 2000  # Or reduce iterations
```

**Poor Convergence:**
```yaml
TRAIN_CONFIGS:
  iterations: 10000  # Increase training time
NETWORK_CONFIGS:
  lr: 1e-3  # Adjust learning rate
```

## Performance

Typical results on 512×512 images:
- **Training time**: 1-2 hours (GPU)
- **Final PSNR**: 28-32 dB
- **Final SSIM**: 0.85-0.95
- **Memory**: 4-6 GB

## Implementation Details

### NTK-weighted Importance

NINT computes importance scores by:
1. Computing Jacobian (parameter gradients per pixel)
2. Computing NTK (parameter sensitivity matrix)
3. Weighting importance by residual error × NTK score

This is more effective than residual-only sampling, leading to better convergence.

### Reproducibility

Results are reproducible when seed is fixed:
```yaml
TRAIN_CONFIGS:
  seed: 42
```

## License

MIT License - See LICENSE file for details.

---

**Package Size**: ~2 MB (code only, excludes sample images)  
**Core Code**: ~1,300 LOC (Python)  
**Dependencies**: PyTorch, NumPy, scikit-image, Hydra  
**Status**: Ready for open-source usage
