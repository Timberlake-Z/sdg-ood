# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of "Learning to Learn Single Domain Generalization" (CVPR 2020) by Fengchun Qiao, Long Zhao and Xi Peng. The project implements Meta-learning Adversarial Domain Augmentation (M-ADA) for out-of-distribution (OOD) generalization using only single domain training data.

**Paper**: [https://arxiv.org/abs/2003.13216](https://arxiv.org/abs/2003.13216)

The key innovation is using adversarial training to create "fictitious" yet "challenging" populations from a single source domain, enabling the model to generalize to unseen domains with theoretical guarantees.

## Repository Structure

```
sdg-ood/
├── main_Digits.py      # Main training/testing script for digit datasets
├── main_ood.py         # Training script for CIFAR OOD detection experiments
├── models/
│   ├── ada_conv.py     # Core model architectures (ConvNet, WAE, Adversary)
│   └── wrn.py          # Wide ResNet implementation for CIFAR experiments
├── utils/
│   ├── digits_process_dataset.py  # Data loading for digit datasets
│   ├── download_and_process_mnist.py  # MNIST download script
│   └── ops.py          # Utility operations
├── data/               # Dataset storage
│   ├── mnist/          # Source domain (training)
│   ├── mnist_m/        # Target domain (evaluation)
│   ├── svhn/           # Target domain (evaluation)
│   ├── syn/            # Target domain (evaluation)
│   └── usps/           # Target domain (evaluation)
├── doc/
│   └── task_history.md # Documentation of repository changes
└── checkpoint/         # Model checkpoints (create manually)
```

## Key Commands

### Dataset Preparation
```bash
# Download and process MNIST dataset (required before training)
cd utils
python download_and_process_mnist.py
cd ..

# Download SVHN test set for evaluation
mkdir -p ./data/svhn
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat -O "./data/svhn/test_32x32.mat"
```

### Training
```bash
# Train from scratch on MNIST (digits experiment)
python main_Digits.py

# Train with custom parameters
python main_Digits.py --num_iters 10001 --batch-size 32 --lr 0.0001 --K 3 --T_adv 25

# Train on CIFAR-10 for OOD detection
python main_ood.py --dataset cifar10 --batch-size 128
```

### Testing/Evaluation
```bash
# Evaluate pretrained model on all target domains
python main_Digits.py --mode test --resume checkpoint/pretrained/ckpt_mnist.pth.tar

# Test specific checkpoint
python main_Digits.py --mode test --resume path/to/your/checkpoint.pth.tar
```

## Architecture Overview

The codebase implements a meta-learning framework for domain generalization with three main components:

1. **Main Model (ConvNet)**: A CNN classifier wrapped in a MetaNN Learner for meta-learning capabilities
   - Location: `models/ada_conv.py:5-31`
   - Architecture: Conv → MaxPool → Conv → MaxPool → FC layers → 10-class output
   - Input: 32×32×3 images
   - Features: 1024-dimensional representation

2. **Domain Augmentation (WAE)**: Wasserstein Autoencoder that generates fictitious domains
   - Location: `models/ada_conv.py:33-53`
   - Architecture: FC encoder (3072→400→20) and decoder (20→400→3072)
   - Purpose: Creates domain perturbations while preserving semantic content

3. **Adversarial Training (Adversary)**: Discriminator for adversarial domain generation
   - Location: `models/ada_conv.py:54-66`
   - Architecture: Simple MLP (20→128→1)
   - Purpose: Ensures generated domains are challenging yet valid

## Training Process

The training follows this procedure (implemented in `main_Digits.py`):

1. **WAE Pre-training** (20 epochs): Train the autoencoder to learn meaningful representations
2. **Meta-learning Loop**:
   - Generate K augmented domains using WAE
   - Perform adversarial training to make domains challenging
   - Update main model using meta-learning across all domains
   - Periodically evaluate on validation set

Key hyperparameters:
- `--K`: Number of augmented domains (default: 3)
- `--T_adv`: Adversarial training iterations (default: 25)
- `--T_min`: Interval between domain augmentation (default: 100)
- `--gamma`: Constraint coefficient (default: 1)
- `--beta`: Relaxation coefficient (default: 2000)

## Key Dependencies

- Python 3.6
- PyTorch 1.1.0
- MetaNN 0.1.5 (for meta-learning operations)
- Scipy 1.2.1
- CUDA-capable GPU (recommended)

## Data Organization

Datasets are organized under `data/` directory:

**Source Domain** (used for training):
- `mnist/`: MNIST dataset in pickle format

**Target Domains** (used for evaluation only):
- `svhn/`: Street View House Numbers (.mat format)
- `mnist_m/`: MNIST-M with background clutter (image folders)
- `syn/`: Synthetic digits dataset (.mat format)
- `usps/`: USPS digits dataset (.pkl format)

The data loaders in `utils/digits_process_dataset.py` handle different dataset formats automatically.

## Important Notes

1. **GPU Usage**: Set GPU ID with `--GPU_ID` flag
2. **Reproducibility**: Fixed random seeds (torch: 0, numpy: 0)
3. **Checkpoint Format**: Saves model state, iteration number, and best accuracy
4. **Evaluation**: Automatically tests on all available target domains
5. **Pretrained Models**: Available on [Google Drive](https://drive.google.com/open?id=19VGuIsv38JutNCkKrG3htBBau3gomhYC)

## Citation

```bibtex
@inproceedings{qiaoCVPR20learning,
  title={Learning to learn single domain generalization},
  author={Qiao, Fengchun and Zhao, Long and Peng, Xi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={12556--12565},
  year={2020}
}
```