# NINT: NTK-Guided Implicit Neural Teaching

## Submission Info

- Prepared for CVPR 2026 submission #5792
- Package is anonymized and ready for open-box use as supplementary material

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

The script trains a SIREN model on DIV2K dataset (5 images already included in `/images`) by default and saves results to `outputs/`.

### 3. Custom Configuration

Edit `config/train_image.yaml`:

```yaml
DATASET_CONFIGS:
  file_path: path/to/your/image.png

TRAIN_CONFIGS:
  iterations: 5000

nint:
  mt_ratio: 0.2
  batch_size_scheduler: constant
  sample_interval: dense
```

## Output

Training outputs saved to `outputs/siren_sampling/`:

```
├── model.pth
├── hydra_configs/
└── src/
```

## Metrics

The training script logs:
- **MSE loss**
- **PSNR**
- **SSIM**
- **LPIPS**

## Advanced Configuration

### Enable W&B Logging (Optional)

```yaml
WANDB_CONFIGS:
  use_wandb: 1
  wandb_project: YOUR_PROJECT
  wandb_entity: YOUR_USERNAME
```

Then run: `wandb login && python train_image.py`

### Reproducibility

Results are reproducible when seed is fixed:
```yaml
TRAIN_CONFIGS:
  seed: 42
```

## License

MIT License - See LICENSE file for details.

---

**Dependencies**: PyTorch, NumPy, scikit-image, Hydra
