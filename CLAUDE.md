# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of "Learning to Learn Single Domain Generalization" (CVPR 2020). The project implements Meta-learning Adversarial Domain Augmentation (M-ADA) for out-of-distribution generalization using only single domain training data.

## Key Commands

### Dataset Preparation
```bash
# Download and process MNIST dataset (required before training)
cd utils
python download_and_process_mnist.py
cd ..
```

### Training
```bash
# Train from scratch on MNIST
python main_Digits.py

# Train with custom parameters
python main_Digits.py --num_iters 10001 --batch-size 32 --lr 0.0001
```

### Testing/Evaluation
```bash
# Download SVHN test set (if not already present)
mkdir -p ./data/svhn
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat -O "./data/svhn/test_32x32.mat"

# Evaluate pretrained model
python main_Digits.py --mode test --resume checkpoint/pretrained/ckpt_mnist.pth.tar
```

## Architecture Overview

The codebase implements a meta-learning framework for domain generalization with three main components:

1. **Main Model (ConvNet)**: A CNN classifier wrapped in a MetaNN Learner for meta-learning capabilities (models/ada_conv.py:5-31)

2. **Domain Augmentation (WAE)**: Wasserstein Autoencoder that generates fictitious domains by perturbing source data (models/ada_conv.py:33-53)

3. **Adversarial Training (Adversary)**: Discriminator ensuring generated domains are challenging yet semantically valid (models/ada_conv.py:54-66)

The training process (main_Digits.py) consists of:
- Pre-training WAE on source domain
- Iterative meta-learning with adversarial domain augmentation
- Model evaluation on unseen target domains (SVHN, MNIST-M, SYN, USPS)

## Key Dependencies

- Python 3.6
- PyTorch 1.1.0
- MetaNN 0.1.5 (for meta-learning operations)
- Scipy 1.2.1

## Data Organization

Datasets are organized under `data/` directory:
- `mnist/`: Source domain (training)
- `svhn/`, `mnist_m/`, `syn/`, `usps/`: Target domains (evaluation only)

The data loaders in `utils/digits_process_dataset.py` handle different dataset formats (.pkl, .mat, image folders).