
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data import DataLoader

from models.wrn import WideResNet
from metann import Learner
from ood_evaluation import evaluate_ood_detection, print_ood_results

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Argument parser
parser = argparse.ArgumentParser(description='Outlier Exposure Baseline for OOD Detection')
parser.add_argument('--data_dir', default='./data', type=str, help='Root directory for datasets')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
parser.add_argument('--model', default='wrn', type=str, help='Model architecture')
parser.add_argument('--wrn_layers', default=40, type=int, help='WideResNet layers')
parser.add_argument('--wrn_widen_factor', default=2, type=int, help='WideResNet widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='Dropout rate')

# Training parameters
parser.add_argument('--epochs', default=16, type=int, help='Number of training epochs')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for ID data')
parser.add_argument('--oe_batch_size', default=256, type=int, help='Batch size for auxiliary OE data')
parser.add_argument('--test_bs', default=100, type=int, help='Test batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate (same as main_ood.py)')
# === BASIC OE VERSION: Comment out SGD-specific parameters not used in main_ood.py ===
# parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
# parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay')
parser.add_argument('--oe_weight', default=0.5, type=float, help='Weight for OE loss')

# Evaluation and saving
parser.add_argument('--eval_freq', default=5, type=int, help='Evaluation frequency (epochs)')
parser.add_argument('--save', default='./checkpoints', type=str, help='Checkpoint save directory')
parser.add_argument('--pretrained_model', default='./models/cifar10_wrn_pretrained_epoch_99.pt', 
                    type=str, help='Path to pretrained model checkpoint')

# Device
parser.add_argument('--gpu', default=0, type=int, help='GPU device id')

def get_dataloaders(args):
    """Get data loaders for training and evaluation."""
    
    # CIFAR normalization parameters
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        num_classes = 10
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        num_classes = 100
    
    # Data transforms
    train_transform = trn.Compose([
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(32, padding=4),
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])
    
    test_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])
    
    # Test transform for OOD datasets (resize to 32x32)
    test_transform_resize = trn.Compose([
        trn.Resize(32),
        trn.CenterCrop(32),
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])
    
    # ID training and test data
    if args.dataset == 'cifar10':
        train_data_in = dset.CIFAR10(args.data_dir + '/cifarpy', train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_dir + '/cifarpy', train=False, transform=test_transform, download=True)
    else:
        train_data_in = dset.CIFAR100(args.data_dir + '/cifarpy', train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_dir + '/cifarpy', train=False, transform=test_transform, download=True)
    
    # Auxiliary OE training data (Tiny ImageNet-200, same as main_ood.py)
    oe_data = dset.ImageFolder(
        root="../data/tiny-imagenet-200/train/",
        transform=trn.Compose([
            trn.Resize(32),
            trn.RandomCrop(32, padding=4),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            trn.Normalize(mean, std)
        ])
    )
    
    # Create data loaders
    train_loader_in = DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=False, num_workers=4)
    train_loader_out = DataLoader(oe_data, batch_size=args.oe_batch_size, shuffle=True, num_workers=4)
    
    # OOD test datasets
    ood_test_loaders = {}
    
    # Texture (DTD)
    try:
        texture_data = dset.ImageFolder('../data/dtd/images', transform=test_transform_resize)
        texture_loader = DataLoader(texture_data, batch_size=args.test_bs, shuffle=False, num_workers=4)
        ood_test_loaders['texture'] = texture_loader
        print(f"  ✓ Texture (DTD): {len(texture_data)} samples")
    except:
        print(f"  ✗ Texture (DTD): Not found")
    
    # Places365
    try:
        places365_data = dset.ImageFolder('../data/places365_standard', transform=test_transform_resize)
        places365_loader = DataLoader(places365_data, batch_size=args.test_bs, shuffle=False, num_workers=4)
        ood_test_loaders['places365'] = places365_loader
        print(f"  ✓ Places365: {len(places365_data)} samples")
    except:
        print(f"  ✗ Places365: Not found")
    
    # LSUN-C
    try:
        lsunc_data = dset.ImageFolder('../data/LSUN', transform=test_transform_resize)
        lsunc_loader = DataLoader(lsunc_data, batch_size=args.test_bs, shuffle=False, num_workers=4)
        ood_test_loaders['lsunc'] = lsunc_loader
        print(f"  ✓ LSUN-C: {len(lsunc_data)} samples")
    except:
        print(f"  ✗ LSUN-C: Not found")
    
    # LSUN-R
    try:
        lsunr_data = dset.ImageFolder('../data/LSUN_resize', transform=test_transform_resize)
        lsunr_loader = DataLoader(lsunr_data, batch_size=args.test_bs, shuffle=False, num_workers=4)
        ood_test_loaders['lsunr'] = lsunr_loader
        print(f"  ✓ LSUN-R: {len(lsunr_data)} samples")
    except:
        print(f"  ✗ LSUN-R: Not found")
    
    # iSUN
    try:
        isun_data = dset.ImageFolder('../data/iSUN', transform=test_transform_resize)
        isun_loader = DataLoader(isun_data, batch_size=args.test_bs, shuffle=False, num_workers=4)
        ood_test_loaders['isun'] = isun_loader
        print(f"  ✓ iSUN: {len(isun_data)} samples")
    except:
        print(f"  ✗ iSUN: Not found")
    
    # CIFAR100 as OOD for CIFAR10
    if args.dataset == 'cifar10':
        cifar100_data = dset.CIFAR100(args.data_dir + '/cifarpy', train=False, transform=test_transform, download=True)
        cifar_loader = DataLoader(cifar100_data, batch_size=args.test_bs, shuffle=False, num_workers=4)
        ood_test_loaders['cifar'] = cifar_loader
        print(f"  ✓ CIFAR100: {len(cifar100_data)} samples")
    else:
        cifar10_data = dset.CIFAR10(args.data_dir + '/cifarpy', train=False, transform=test_transform, download=True)
        cifar_loader = DataLoader(cifar10_data, batch_size=args.test_bs, shuffle=False, num_workers=4)
        ood_test_loaders['cifar'] = cifar_loader
        print(f"  ✓ CIFAR10: {len(cifar10_data)} samples")
    
    # Calculate ood_num_examples as 1/5 of test data size
    ood_num_examples = len(test_data) // 5
    
    return train_loader_in, train_loader_out, test_loader, ood_test_loaders, num_classes, ood_num_examples

def load_model(args, num_classes, device):
    """Load and setup the WideResNet model."""
    print(f"Loading model...")
    print(f"  Architecture: WideResNet-{args.wrn_layers}-{args.wrn_widen_factor}")
    print(f"  Classes: {num_classes}")
    print(f"  Checkpoint: {args.pretrained_model}")
    
    # Create model wrapped with Learner (same as main_ood.py for consistency)
    base_net = WideResNet(args.wrn_layers, num_classes, args.wrn_widen_factor, dropRate=args.droprate).to(device)
    net = Learner(base_net)
    
    # Load pretrained checkpoint
    if os.path.exists(args.pretrained_model):
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'net' in checkpoint:
            state_dict = checkpoint['net']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel prefixes
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        # Load state dict into the base WideResNet model
        base_net.load_state_dict(state_dict)
        print(f"  ✓ Pretrained model loaded successfully!")
    else:
        print(f"  ✗ Pretrained model not found, training from scratch")
    
    return net

class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Compute the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(net, train_loader_in, train_loader_out, optimizer, device, args):
    """Train for one epoch using Outlier Exposure."""
    net.train()
    
    losses = AverageMeter()
    ce_losses = AverageMeter()
    oe_losses = AverageMeter()
    top1 = AverageMeter()
    
    # Ensure we iterate through both loaders properly
    train_loader_out_iter = iter(train_loader_out)
    
    for batch_idx, (data_in, target) in enumerate(train_loader_in):
        try:
            data_out, _ = next(train_loader_out_iter)
        except StopIteration:
            # Restart the auxiliary loader
            train_loader_out_iter = iter(train_loader_out)
            data_out, _ = next(train_loader_out_iter)
        
        # Move to device
        data_in, target = data_in.to(device), target.to(device)
        data_out = data_out.to(device)
        
        # Combine ID and OOD data
        data_combined = torch.cat((data_in, data_out), 0)
        batch_size_in = data_in.size(0)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = net(data_combined)
        logits_in = logits[:batch_size_in]
        logits_out = logits[batch_size_in:]
        
        # Loss calculation
        ce_loss = F.cross_entropy(logits_in, target)
        oe_loss = -(logits_out.mean(1) - torch.logsumexp(logits_out, dim=1)).mean()
        total_loss = ce_loss + args.oe_weight * oe_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        # scheduler.step()  # Commented out - not used in main_ood.py
        
        # Calculate accuracy
        prec1 = accuracy(logits_in, target)[0]
        
        # Update meters
        losses.update(total_loss.item(), batch_size_in)
        ce_losses.update(ce_loss.item(), batch_size_in)
        oe_losses.update(oe_loss.item(), batch_size_in)
        top1.update(prec1.item(), batch_size_in)
        
        if batch_idx % 100 == 0:
            print(f'Train Batch: [{batch_idx}/{len(train_loader_in)}] '
                  f'Loss: {losses.avg:.4f} CE: {ce_losses.avg:.4f} OE: {oe_losses.avg:.4f} '
                  f'Acc: {top1.avg:.2f}%')
    
    return losses.avg, top1.avg

def test(net, test_loader, device):
    """Test the model on ID test data."""
    net.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = net(data)
            loss = F.cross_entropy(output, target)
            
            prec1 = accuracy(output, target)[0]
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
    
    return losses.avg, top1.avg

def main():
    """Main training and evaluation loop."""
    args = parser.parse_args()
    
    print("="*60)
    print("Outlier Exposure Baseline for OOD Detection")
    print("="*60)
    
    # Device setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    # Create save directory
    os.makedirs(args.save, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_loader_in, train_loader_out, test_loader, ood_test_loaders, num_classes, ood_num_examples = get_dataloaders(args)
    
    print(f"\nDataset summary:")
    print(f"  ID training: {len(train_loader_in.dataset)} samples")
    print(f"  OE auxiliary: {len(train_loader_out.dataset)} samples") 
    print(f"  ID test: {len(test_loader.dataset)} samples")
    print(f"  OOD test datasets: {len(ood_test_loaders)} datasets")
    print(f"  OOD samples per eval: {ood_num_examples}")
    
    # Load model
    net = load_model(args, num_classes, device)
    
    # === BASIC OE VERSION: Using Adam optimizer like main_ood.py ===
    # Setup optimizer and scheduler (commented out advanced strategies not in main_ood.py)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader_in))
    
    # Use same optimizer as main_ood.py for fair comparison (Adam with fixed lr)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    print(f"\nTraining setup:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  OE weight: {args.oe_weight}")
    print(f"  Eval frequency: {args.eval_freq} epochs")
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        
        # Training
        train_loss, train_acc = train(net, train_loader_in, train_loader_out, optimizer, device, args)
        
        # Testing  
        test_loss, test_acc = test(net, test_loader, device)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f'Epoch {epoch+1:3d} | Time {epoch_time:5.0f}s | '
              f'Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | '
              f'Test Loss {test_loss:.4f} | Test Acc {test_acc:.2f}%')
        
        # Save checkpoint
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
        
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),  # Commented out - no scheduler in basic version
            'best_acc': best_acc,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        
        # Save regular checkpoint
        save_path = os.path.join(args.save, f'{args.dataset}_oe_epoch_{epoch+1}.pt')
        torch.save(checkpoint, save_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(args.save, f'{args.dataset}_oe_best.pt')
            torch.save(checkpoint, best_path)
            print(f'  → New best accuracy: {best_acc:.2f}%')
        
        # OOD evaluation every eval_freq epochs
        if (epoch + 1) % args.eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"OOD Evaluation - Epoch {epoch+1}")
            print(f"{'='*60}")
            
            try:
                # Use evaluation from ood_evaluation module
                results = evaluate_ood_detection(net, test_loader, ood_test_loaders, device, ood_num_examples, args.test_bs)
                print_ood_results(results, ood_num_examples)
                
                # Calculate summary statistics
                aurocs = [result[0] for result in results.values()]
                print(f"\nSummary: Mean AUROC = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
                
            except Exception as e:
                print(f"OOD evaluation failed: {e}")
            
            print(f"{'='*60}")
        
        # Remove previous checkpoint to save space (keep only last 2)
        if epoch > 1:
            old_path = os.path.join(args.save, f'{args.dataset}_oe_epoch_{epoch-1}.pt')
            if os.path.exists(old_path):
                os.remove(old_path)
    
    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()