# [CVPR '26] NINT: NTK-Guided Implicit Neural Teaching

This is the official implementation of [**[NTK-Guided Implicit Neural Teaching]**](https://arxiv.org/pdf/2511.15487) \
[Chen Zhang*](https://scholar.google.com/citations?user=7CkE3C4AAAAJ&hl=en), [Wei Zuo*](https://scholar.google.com/citations?user=AonK3NEAAAAJ&hl=en), [Bingyang Cheng](https://scholar.google.com/citations?user=rf646k8AAAAJ&hl=en&oi=sra), Yikun Wang, Wei-Bin Kou,
[Yik-Chung Wu](https://www.eee.hku.hk/~ycwu/), [Ngai Wong](https://www.eee.hku.hk/~nwong/)

**[20-Feb-2026]** Our work is accepted by ***CVPR 2026***  🎉

---

## ✨ Intro & Key Features

🚀 **NINT** is a principled, plug-and-play sampling framework that accelerates **Implicit Neural Representation (INR)** training by leveraging the **Neural Tangent Kernel (NTK)**. 
NINT effectively identifies coordinates that maximize global functional updates by capturing both *local fitting errors* (**Self-leverage**) and *complex coordinate coupling* (**Functional Coupling**), representing a *state-of-the-art* INR training acceleration paradigm.

<p align="center">
  <img src="./NTK.png" width="350">
</p>

**⚡ State-of-the-Art Acceleration**: Reduces required training iteration and time by up to **26.58%** and **48.99%**.

**🧠 NTK-Aware Selection**: Moves beyond simple error-based sampling by scoring points via their "self-leverage" and "functional coupling".

**🔌 Plug-and-Play**: Compatible with various network architectures and data modalities (Image, Audio, 3D).

---

## 🛠️ Getting Started

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate nint_env
```

### 2. Training (Quick Start)

Train images (using 5 images from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) in `/images` as examples) with a default SIREN network architecture:

```bash
python train_image.py
```

Outputs (models and logs) are saved to `outputs/`.

### 3. Advanced Usage

Configure your training via `config/train_image.yaml`:

```yaml
DATASET_CONFIGS:
  file_path: path/to/your/image.png

TRAIN_CONFIGS:
  iterations: 5000

nint:
  mt_ratio: 0.2
  batch_size_scheduler: constant
  sample_interval: dense

WANDB_CONFIGS:
  use_wandb: 1
  wandb_project: YOUR_PROJECT
  wandb_entity: YOUR_USERNAME
```

### 4. Reproducibility

Results are reproducible when seed is fixed:
```yaml
TRAIN_CONFIGS:
  seed: 42
```

---

## 📑 Citation

```bibtex
@article{zhang2025ntk,
  title={NTK-Guided Implicit Neural Teaching},
  author={Zhang, Chen and Zuo, Wei and Cheng, Bingyang and Wang, Yikun and Kou, Wei-Bin and WU, Yik Chung and Wong, Ngai},
  journal={arXiv preprint arXiv:2511.15487},
  year={2025}
}
```

## 🏷️ Acknowledgment
We thank the authors of the following works for releasing their codebases:
- [INT](https://github.com/chen2hang/INT_NonparametricTeaching)
- [Soft Mining](https://github.com/ubc-vision/nf-soft-mining)
- [EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](https://github.com/zwx-open/EVOS-INR)
- [Expansive Supervision for Neural Radiance Fields](https://github.com/zwx-open/Expansive-Supervision)

## 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.
