#!/usr/bin/env python
"""
Load Pre-trained Checkpoint and Evaluate OOD Detection

This script loads a pre-trained WideResNet model and evaluates its 
Out-of-Distribution (OOD) detection performance using the same configuration 
as main_ood.py.

Usage:
    python try_load_the_ckpt_and_eval.py --dataset cifar10 --model_path ./models/cifar10_wrn_pretrained_epoch_99.pt
"""

# import packages
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import numpy as np
from models.wrn import WideResNet
from metann import Learner
from ood_evaluation import evaluate_ood_detection, print_ood_results

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load checkpoint and evaluate OOD detection')
    
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset to use (cifar10 or cifar100)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--test_bs', default=200, type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--GPU_ID', default=0, type=int,
                        help='GPU device ID')
    parser.add_argument('--wrn_layers', default=40, type=int,
                        help='Number of layers for WideResNet')
    parser.add_argument('--wrn_widen_factor', default=2, type=int,
                        help='Widen factor for WideResNet')
    parser.add_argument('--droprate', default=0.0, type=float,
                        help='Dropout rate for WideResNet')
    
    return parser.parse_args()


def get_test_dataloaders(args):
    """
    Create test data loaders for ID and OOD datasets.
    
    Returns:
        tuple: (test_loader, ood_test_loaders, num_classes, ood_num_examples)
    """
    # Set normalization parameters
    if 'cifar' in args.dataset:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    else: 
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Define test transforms (no augmentation)
    test_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])

    test_transform_resize = trn.Compose([
        trn.Resize(32),
        trn.CenterCrop(32),  # Ensure uniform 32x32 dimensions
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])

    # Load ID test dataset
    if args.dataset == 'cifar10':
        test_data = dset.CIFAR10('../data/cifarpy', train=False, transform=test_transform)
        cifar_data = dset.CIFAR100('../data/cifarpy', train=False, transform=test_transform)  
        num_classes = 10
    else:
        test_data = dset.CIFAR100('../data/cifarpy', train=False, transform=test_transform)
        cifar_data = dset.CIFAR10('../data/cifarpy', train=False, transform=test_transform)
        num_classes = 100

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    # Create ID test loader
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_bs, shuffle=False, **kwargs)

    # Load OOD test datasets
    print("Loading OOD test datasets...")
    
    try:
        texture_data = dset.ImageFolder("../data/dtd/images", transform=test_transform_resize)
        texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.test_bs, shuffle=True, **kwargs)
        print(f"  ✓ Texture (DTD): {len(texture_data)} samples")
    except:
        print("  ✗ Texture (DTD): Not found")
        texture_loader = None

    try:
        places365_data = dset.ImageFolder("../data/places365/", transform=test_transform_resize)
        places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.test_bs, shuffle=True, **kwargs)
        print(f"  ✓ Places365: {len(places365_data)} samples")
    except:
        print("  ✗ Places365: Not found")
        places365_loader = None

    try:
        lsunc_data = dset.ImageFolder("../data/LSUN/", transform=test_transform_resize)
        lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.test_bs, shuffle=True, **kwargs)
        print(f"  ✓ LSUN-C: {len(lsunc_data)} samples")
    except:
        print("  ✗ LSUN-C: Not found")
        lsunc_loader = None

    try:
        lsunr_data = dset.ImageFolder("../data/LSUN_resize/LSUN_resize", transform=test_transform_resize)
        lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.test_bs, shuffle=True, **kwargs)
        print(f"  ✓ LSUN-R: {len(lsunr_data)} samples")
    except:
        print("  ✗ LSUN-R: Not found")
        lsunr_loader = None

    try:
        isun_data = dset.ImageFolder("../data/iSUN_fixed", transform=test_transform_resize)
        isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.test_bs, shuffle=True, **kwargs)
        print(f"  ✓ iSUN: {len(isun_data)} samples")
    except:
        print("  ✗ iSUN: Not found")
        isun_loader = None

    # CIFAR as OOD (opposite dataset)
    cifar_loader = torch.utils.data.DataLoader(cifar_data, batch_size=args.test_bs, shuffle=True, **kwargs)
    cifar_name = 'cifar100' if args.dataset == 'cifar10' else 'cifar10'
    print(f"  ✓ {cifar_name.upper()}: {len(cifar_data)} samples")

    # Create OOD test loaders dictionary
    ood_test_loaders = {}
    if texture_loader is not None:
        ood_test_loaders['texture'] = texture_loader
    if places365_loader is not None:
        ood_test_loaders['places365'] = places365_loader
    if lsunc_loader is not None:
        ood_test_loaders['lsunc'] = lsunc_loader
    if lsunr_loader is not None:
        ood_test_loaders['lsunr'] = lsunr_loader
    if isun_loader is not None:
        ood_test_loaders['isun'] = isun_loader
    ood_test_loaders['cifar'] = cifar_loader

    # Calculate ood_num_examples as 1/5 of test data size (same as main_ood.py)
    ood_num_examples = len(test_data) // 5
    
    print(f"\nDataset summary:")
    print(f"  ID dataset: {args.dataset.upper()} ({len(test_data)} test samples)")
    print(f"  OOD datasets: {len(ood_test_loaders)} datasets loaded")
    print(f"  Evaluation config: {ood_num_examples} OOD samples per dataset")

    return test_loader, ood_test_loaders, num_classes, ood_num_examples


def load_model(args, num_classes, device):
    """
    Load the pre-trained model from checkpoint.
    
    Args:
        args: Command line arguments
        num_classes: Number of classes for the model
        device: Device to load the model on
        
    Returns:
        Loaded model wrapped in Learner
    """
    print(f"\nLoading model...")
    print(f"  Architecture: WideResNet-{args.wrn_layers}-{args.wrn_widen_factor}")
    print(f"  Classes: {num_classes}")
    print(f"  Checkpoint: {args.model_path}")
    
    # Check if checkpoint file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    # Create model architecture
    net = WideResNet(args.wrn_layers, num_classes, args.wrn_widen_factor, dropRate=args.droprate).to(device)
    model = Learner(net)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Handle different model saving formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # Checkpoint is a dictionary with state_dict
            state_dict = checkpoint['state_dict']
        else:
            # Checkpoint is the state_dict itself
            state_dict = checkpoint
        
        # Handle DataParallel 'module.' prefix
        if 'module.' in list(state_dict.keys())[0]:
            # Remove 'module.' prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        # Load the state dict
        model.module.load_state_dict(state_dict)
        
        print("  ✓ Model loaded successfully!")
        
        # Print additional checkpoint info if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'accuracy' in checkpoint:
                print(f"  Training accuracy: {checkpoint['accuracy']:.4f}")
            if 'loss' in checkpoint:
                print(f"  Training loss: {checkpoint['loss']:.4f}")
                
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    return model


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()
    
    print("="*60)
    print("OOD Detection Evaluation with Pre-trained Checkpoint")
    print("="*60)
    
    # Set up GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    # load the dataloader (test_loader and ood_loader)
    test_loader, ood_test_loaders, num_classes, ood_num_examples = get_test_dataloaders(args)
    
    # load the pretrained model from checkpoint
    model = load_model(args, num_classes, device)
    
    # Set model to evaluation mode
    model.eval()
    
    # eval the model AUROC with same configuration as main_ood
    print(f"\n{'='*60}")
    print("Running OOD Detection Evaluation")
    print(f"{'='*60}")
    
    try:
        # Use the same evaluation function as main_ood.py
        results = evaluate_ood_detection(
            model, test_loader, ood_test_loaders, 
            device, ood_num_examples, args.test_bs
        )
        
        # Print results in the same format as main_ood.py
        print_ood_results(results, ood_num_examples)
        
        # Calculate and print summary statistics
        aurocs = [result[0] for result in results.values()]
        auprs = [result[1] for result in results.values()]
        fprs = [result[2] for result in results.values()]
        
        print(f"\nSummary Statistics:")
        print(f"  Mean AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        print(f"  Mean AUPR:  {np.mean(auprs):.4f} ± {np.std(auprs):.4f}")
        print(f"  Mean FPR95: {np.mean(fprs):.4f} ± {np.std(fprs):.4f}")
        
        print(f"\n{'='*60}")
        print("✅ Evaluation completed successfully!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()