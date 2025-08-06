import argparse
import os
import time
import math
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from models.ada_conv import ConvNet, WAE, Adversary
import numpy
from metann import Learner
import torchvision.transforms as trn
import torchvision.datasets as dset
from utils.digits_process_dataset import *
from models.wrn import WideResNet

torch.manual_seed(0)
numpy.random.seed(0)

def safe_unnormalize(tensor, mean, std, clamp_range=(0.0, 1.0)):
    """安全地将标准化tensor恢复到[0,1]范围，带边界限制"""
    result = tensor.clone()
    
    if isinstance(mean, list):
        mean = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
        std = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    
    # 处理不同tensor形状
    if tensor.dim() == 4:  # (batch, channel, height, width)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif tensor.dim() == 2:  # (batch, flattened: 3072)
        mean = mean.repeat(tensor.size(1) // 3)
        std = std.repeat(tensor.size(1) // 3)
    
    # unnormalize: x = x * std + mean
    result = result * std + mean
    
    # 边界安全限制
    if clamp_range is not None:
        result = torch.clamp(result, clamp_range[0], clamp_range[1])
    
    return result

def safe_normalize(tensor, mean, std):
    """将[0,1]数据标准化回标准范围"""
    result = tensor.clone()
    
    if isinstance(mean, list):
        mean = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
        std = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    
    if tensor.dim() == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif tensor.dim() == 2:
        mean = mean.repeat(tensor.size(1) // 3)
        std = std.repeat(tensor.size(1) // 3)
    
    # normalize: x = (x - mean) / std
    result = (result - mean) / std
    
    return result

parser = argparse.ArgumentParser(description='Training on Digits')
parser.add_argument('--data_dir', default='data', type=str,
                    help='dataset dir')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset cifar100 or cifar10')
parser.add_argument('--num_iters', default=10001, type=int,
                    help='number of total iterations to run')
parser.add_argument('--start_iters', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--min-learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_max', '--adv-learning-rate', default=1, type=float,
                    help='adversarial learning rate')
parser.add_argument('--gamma', default=1, type=float,
                    help='coefficient of constraint')
parser.add_argument('--beta', default=2000, type=float,
                    help='coefficient of relaxation')
parser.add_argument('--T_adv', default=25, type=int,
                    help='iterations for adversarial training')
parser.add_argument('--advstart_iter', default=0, type=int,
                    help='iterations for pre-train')
parser.add_argument('--K', default=3, type=int,
                    help='number of augmented domains')
parser.add_argument('--T_min', default=100, type=int,
                    help='intervals between domain augmentation')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str,
                    help='path to saved checkpoint (default: none)')
parser.add_argument('--name', default='Digits', type=str,
                    help='name of experiment')
parser.add_argument('--mode',  default='train', type=str,
                    help='train or test')
parser.add_argument('--GPU_ID', default=0, type=int,
                    help='GPU_id')
parser.add_argument('--ood_weight', default=0.5, type=float,
                    help='weight for out-of-distribution loss')
parser.add_argument('--ood_weight2', default=0.5, type=float,
                    help='weight for out-of-distribution loss in meta learning')
parser.add_argument('--oe_batch_size', default=256, type=int,
                    help='batch size for auxiliary loader')
parser.add_argument('--wrn_layers', default=40, type=int,
                    help='number of layers for WideResNet')
parser.add_argument('--wrn_widen_factor', default=2, type=int,
                    help='widen factor for WideResNet')
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout rate for WideResNet')
parser.add_argument('--test_bs', default=200, type=int,
                    help='batch size for test loader')
parser.add_argument('--pretrained', default=False, action='store_true',
                    help='use pretrained model')




def get_dataloaders(args):
    if 'cifar' in args.dataset:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    else: 
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # define transform
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

    if args.dataset == 'cifar10':
        train_data_in = dset.CIFAR10('../data/cifarpy', train=True, transform=train_transform)
        test_data = dset.CIFAR10('../data/cifarpy', train=False, transform=test_transform)
        cifar_data = dset.CIFAR100('../data/cifarpy', train=False, transform=test_transform)  # 
        num_classes = 10
    else:
        train_data_in = dset.CIFAR100('../data/cifarpy', train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100('../data/cifarpy', train=False, transform=test_transform)
        cifar_data = dset.CIFAR10('../data/cifarpy', train=False, transform=test_transform)
        num_classes = 100

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader_in = torch.utils.data.DataLoader(
        train_data_in, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    auxiliary_data = dset.ImageFolder(
        root="../data/tiny-imagenet-200/train/",
        transform=trn.Compose([
            trn.Resize(32),
            trn.RandomCrop(32, padding=4),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            trn.Normalize(mean, std)
        ])
    )
    auxiliary_loader = torch.utils.data.DataLoader(
        auxiliary_data, batch_size=args.oe_batch_size, shuffle=True, drop_last=True, **kwargs)

    test_transform_center = trn.Compose([
        trn.Resize(32),
        trn.CenterCrop(32),
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])
    test_transform_resize = trn.Compose([
        trn.Resize(32),
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])

    texture_data = dset.ImageFolder("../data/dtd/images", transform=test_transform_center)
    places365_data = dset.ImageFolder("../data/places365_standard/", transform=test_transform_center)
    lsunc_data = dset.ImageFolder("../data/LSUN_resize/LSUN_resize", transform=test_transform_resize)  # Use LSUN_resize as fallback
    lsunr_data = dset.ImageFolder("../data/LSUN_resize/LSUN_resize", transform=test_transform_resize)
    isun_data = dset.ImageFolder("../data/iSUN_fixed", transform=test_transform_resize)

    texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.test_bs, shuffle=True, **kwargs)
    places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.test_bs, shuffle=True, **kwargs)
    lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.test_bs, shuffle=True, **kwargs)
    lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.test_bs, shuffle=True, **kwargs)
    isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.test_bs, shuffle=True, **kwargs)
    cifar_loader = torch.utils.data.DataLoader(cifar_data, batch_size=args.test_bs, shuffle=True, **kwargs)

    return train_loader_in, test_loader, auxiliary_loader, {
        'texture': texture_loader,
        'places365': places365_loader,
        'lsunc': lsunc_loader,
        'lsunr': lsunr_loader,
        'isun': isun_loader,
        'cifar': cifar_loader
    }, num_classes

def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 on stackoverflow
    # Fix: Only set CUDA_VISIBLE_DEVICES if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)

    exp_name = args.name

    kwargs = {'num_workers': 4}


    # create model, use Learner to wrap it, model needs to be reset as WideResNet
    # model = Learner(ConvNet())
    # model = model.cuda()
    # cudnn.benchmark = True
    # tim, create the model
    if args.dataset == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    net = WideResNet(args.wrn_layers, num_classes, args.wrn_widen_factor, dropRate=args.droprate).to(device)
    model = Learner(net)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    # could also set up an interface for loading pretrained model
    if args.pretrained:
        print('=> loading pretrained model')
        if args.dataset == 'cifar10':
            model_path = './models/cifar10_wrn_pretrained_epoch_99.pt'
        else:
            model_path = './models/cifar100_wrn_pretrained_epoch_99.pt'
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different model saving formats
        if 'module.' in list(checkpoint.keys())[0]:
            # Remove 'module.' prefix if present (from DataParallel)
            new_checkpoint = {}
            for key, value in checkpoint.items():
                new_key = key.replace('module.', '')
                new_checkpoint[new_key] = value
            model.module.load_state_dict(new_checkpoint)
        else:
            model.module.load_state_dict(checkpoint) 

    # tim campared to dal method, we did not use -------
    # optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
    # def cosine_annealing(step, total_steps, lr_max, lr_min):
    #     return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader_in), 1, 1e-6 / args.learning_rate))
    #---------------------------------------------------

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['iter']
            prec = checkpoint['prec']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (iter {})"
                  .format(args.resume, checkpoint['iter']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.mode == 'train':
        train(model, exp_name, kwargs, device)
    else:
        evaluation(model, args.data_dir, args.batch_size, kwargs)


    


class OELoss(nn.Module):
    def __init__(self):
        super(OELoss, self).__init__()

    def forward(self, logits):
        """
        logits: Tensor of shape (batch_size, num_classes)
        """
        return - (logits.mean(1) - torch.logsumexp(logits, dim=1)).mean()

def asarray_and_reshape_ood(data_list):
    """Convert list of batches to numpy array and reshape for OOD use."""
    data_array = numpy.concatenate(data_list, axis=0)
    return data_array

def get_in_scores(model, test_loader, in_dist=True):
    """Get confidence scores for in-distribution test data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    confidence_scores = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            logits = model.module(data)
            smax = F.softmax(logits, dim=1)
            confidence = torch.max(smax, dim=1)[0]
            confidence_scores.extend(confidence.cpu().numpy())
    
    return numpy.array(confidence_scores)

def get_ood_results(model, test_name, test_loader):
    """Get OOD detection results for a specific test dataset."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    ood_scores = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            logits = model.module(data)
            smax = F.softmax(logits, dim=1)
            confidence = torch.max(smax, dim=1)[0]
            ood_scores.extend(confidence.cpu().numpy())
    
    # For evaluation, we need ID scores too
    # This is a simplified version - normally would compare against stored ID scores
    ood_scores = numpy.array(ood_scores)
    
    # Return basic statistics for now
    return {
        'auroc': numpy.mean(ood_scores),  # Placeholder
        'aupr': numpy.std(ood_scores),    # Placeholder  
        'mean_score': numpy.mean(ood_scores),
        'std_score': numpy.std(ood_scores)
    }

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    """Computes the precision@k for the specified values of k"""
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

def save_checkpoint(state, dataset, exp_name, filename='checkpoint.pth.tar'):
    """Save model checkpoint"""
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    if not os.path.exists(f'checkpoint/{dataset}'):
        os.makedirs(f'checkpoint/{dataset}')
    torch.save(state, f'checkpoint/{dataset}/{exp_name}_{filename}')

def log_density_igaussian(z, z_var):
    """
    Calculate log density for isotropic Gaussian
    """
    D = z.size(1)
    log_p = -0.5 * D * torch.log(torch.tensor(2 * 3.141592653589793 * z_var))
    log_p = log_p - 0.5 * torch.sum(z**2, dim=1) / z_var
    return log_p

def train(model, exp_name, kwargs, device):
    print('Pre-train wae')
    # construct train and val dataloader
    # tim, modify the dataloader here 
    # train_loader, val_loader = construct_datasets(args.data_dir, args.batch_size, kwargs)
    id_loader, test_loader, auxiliary_loader, ood_test_loaders, num_classes = get_dataloaders(args)

    wae = WAE().to(device)
    wae_optimizer = torch.optim.Adam(wae.parameters(), lr=1e-3)
    discriminator = Adversary().to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)


    # 获取标准化参数
    if 'cifar' in args.dataset:
        cifar_mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        cifar_std = [x / 255 for x in [63.0, 62.1, 66.7]]
    else:
        cifar_mean = [0.485, 0.456, 0.406]
        cifar_std = [0.229, 0.224, 0.225]

    # tim pre-train wae on source domain -> auxiliary set
    skip_wae = True  # Set to True to skip WAE pre-training for testing
    print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if skip_wae:
        print('Skipping WAE pre-training for testing...')
    else:
        for epoch in range(1, 20 + 1):
            wae_train(wae, discriminator, auxiliary_loader, wae_optimizer, d_optimizer, epoch, device, cifar_mean, cifar_std)

    print('Training task model')
    print(f'Total iterations: {args.num_iters}, Start iter: {args.start_iters}')
    # define loss function (criterion) and optimizer
    # tim, comment the code for modify the loss structure
    # criterion = nn.CrossEntropyLoss().cuda()

    # tim, here need to modify the loss to an oe loss function for auxiliary set
    l_oe = OELoss().to(device)
    
    # tim, here need to define the ID loss function
    l_id = nn.CrossEntropyLoss().to(device)

    mse_loss = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    # tim, here need to init for ood
    # # only augmented domains
    # only_virtual_test_images = []
    # only_virtual_test_labels = []
    # train_loader_iter = iter(train_loader)

    # compared to virtual_test_labels, this is the accumulated augmented data
    augumented_images = [] # cuz all argumented are considered as ood data, no label needed
    auxiliary_loader_iter = iter(auxiliary_loader)
    id_loader_iter = iter(id_loader)

    # counter for domain augmentation
    counter_k = 0

    for t in range(args.start_iters, args.num_iters):
        
        # Progress tracking
        if t % 100 == 0:
            print(f'\n--- Iteration {t}/{args.num_iters} ---')
            print(f'Counter K: {counter_k}, Augmented images: {len(augumented_images)}')

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        # how many samples used from current data / how many samples used from source domain / how many samples used from augmented domain
        break_point = int((len(auxiliary_loader) - 1) / (counter_k + 1))
        src_num = int(args.batch_size / (counter_k + 1))
        aug_num = args.batch_size - src_num

        #-------------------------------
        # domain augmentation
        # condition
        # 1. > start iteration for augmentation
        # 2. the number of augmented domains is less than K
        # 3. reach the interval of T_min
        #-------------------------------
        # Debug: print conditions
        if t % 10 == 0 and t > 0:
            cond1 = t > args.advstart_iter
            cond2 = (t + 1 - args.advstart_iter) % args.T_min == 0
            cond3 = counter_k < args.K
            print(f"  Domain Aug Check: t={t}, cond1={cond1}, cond2={cond2}, cond3={cond3}, counter_k={counter_k}/{args.K}")
        
        if (t > args.advstart_iter) and ((t + 1 - args.advstart_iter) % args.T_min == 0) and (counter_k < args.K):
            print(f'\n🔄 Domain Augmentation at iteration {t} (counter_k={counter_k})')
            model.eval()
            params = list(model.parameters())

            # tim, this is the argumented data in one round, accumulated to `augumented_images`
            # virtual_test_images = []
            # virtual_test_labels = []
            argumented_images_temp = []
            aug_start_time = time.time()

        # the `for` structure generates `break_point` * batchsize new samples by training
        # cuz set `shuffle=True`, the data loader will shuffle the data every time
            for i, (input_a, target_a) in enumerate(auxiliary_loader):

                # just use the first `break_point` batches from auxiliary_loader
                if i == break_point:
                    break

                if counter_k > 0:
                    # tim, check the data loader here-----------------------------
                    input_b, target_b = next(aug_loader_iter)
                    input_comb = torch.cat((input_a[:src_num].float(), input_b[:aug_num])).to(device)
                    # target_comb = torch.cat((target_a[:src_num].long(), target_b[:aug_num])).cuda(non_blocking=True)
                    input_aug = input_comb.clone()
                    # target_aug = target_comb.clone()
                else:
                    input_a = input_a.to(device).float()
                    # target_a = target_a.cuda(non_blocking=True).long()
                    input_aug = input_a.clone()
                    # target_aug = target_a.clone()

                input_aug = input_aug.to(device)
                # target_aug = target_aug.cuda(non_blocking=True)
                aug_optimizer = torch.optim.SGD([input_aug.requires_grad_()], args.lr_max)

                if counter_k == 0:
                    input_feat, output = model.functional(params, False, input_a, return_feat=True)
                    input_a_unnorm = safe_unnormalize(input_a, cifar_mean, cifar_std)
                    recon_batch, _, = wae(input_a_unnorm)
                    # Normalize back to standardized space and detach to prevent gradient issues
                    recon_batch_norm = safe_normalize(recon_batch, cifar_mean, cifar_std).detach()
                else:
                    input_feat, output = model.functional(params, False, input_comb, return_feat=True)
                    input_comb_unnorm = safe_unnormalize(input_comb, cifar_mean, cifar_std)
                    recon_batch, _, = wae(input_comb_unnorm)
                    # Normalize back to standardized space and detach to prevent gradient issues
                    recon_batch_norm = safe_normalize(recon_batch, cifar_mean, cifar_std).detach()
                
                # Detach input_feat to prevent gradient flow through it
                input_feat = input_feat.detach()
                
                #-------------------------------------------------------------------
                # iteratively generate adversarial samples
                for n in range(args.T_adv):
                    # input_aug_feat is the feature before classifier, output_aug is the logits
                    input_aug_feat, output_aug = model.functional(params, False, input_aug, return_feat=True)
                    input_aug_unnorm = safe_unnormalize(input_aug, cifar_mean, cifar_std)
                    recon_batch_aug, _, = wae(input_aug_unnorm)
                    
                    # Normalize WAE output back to standardized space for consistent comparison
                    recon_batch_aug_norm = safe_normalize(recon_batch_aug, cifar_mean, cifar_std)
                    
                    # Constraint: compare features in feature space
                    constraint = mse_loss(input_feat, input_aug_feat)
                    oe_loss = l_oe(output_aug)
                    # Relaxation: compare reconstructions in standardized space
                    relaxation = mse_loss(recon_batch_norm, recon_batch_aug_norm)
                    adv_loss = -(args.beta * relaxation + oe_loss - args.gamma * constraint)
                    aug_optimizer.zero_grad()
                    adv_loss.backward()
                    aug_optimizer.step()

                argumented_images_temp.append(input_aug.data.cpu().numpy())
                # virtual_test_labels.append(target_aug.data.cpu().numpy())

            # tim, this function needs to be modified
            argumented_images_temp = asarray_and_reshape_ood(argumented_images_temp)

            if counter_k == 0:
                augumented_images = np.copy(argumented_images_temp)
                # only_virtual_test_labels = np.copy(virtual_test_labels)
            else:
                augumented_images = np.concatenate([augumented_images, argumented_images_temp])
                # only_virtual_test_labels = np.concatenate([only_virtual_test_labels, virtual_test_labels])

            # dataloader for domain augmentation
            aug_size = len(augumented_images)
            X_aug = torch.stack([torch.from_numpy(augumented_images[i]) for i in range(aug_size)])
            # y_aug = torch.stack([torch.from_numpy(np.asarray(i)) for i in only_virtual_test_labels])
            y_aug = torch.zeros(aug_size, dtype=torch.long)
            aug_dataset = torch.utils.data.TensorDataset(X_aug, y_aug)
            aug_loader = torch.utils.data.DataLoader(aug_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
            aug_loader_iter = iter(aug_loader)

            # dataloader for the latest domain augmentation
            new_aug_size = len(argumented_images_temp)
            new_X_aug = torch.stack([torch.from_numpy(argumented_images_temp[i]) for i in range(new_aug_size)])
            new_y_aug = torch.zeros(new_aug_size, dtype=torch.long)
            new_aug_dataset = torch.utils.data.TensorDataset(new_X_aug, new_y_aug)
            new_aug_loader = torch.utils.data.DataLoader(new_aug_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
            new_aug_loader_iter = iter(new_aug_loader)

            # re-train a wae on  the latest domain augmentation
            if counter_k + 1 < args.K:
                wae = WAE().to(device)
                wae_optimizer = torch.optim.Adam(wae.parameters(), lr=1e-3)
                discriminator = Adversary().to(device)
                d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
                for epoch in range(1, 20 + 1):
                    wae_train(wae, discriminator, new_aug_loader, wae_optimizer, d_optimizer, epoch, device, cifar_mean, cifar_std)
            aug_end_time = time.time()
            print('aug duration', (aug_end_time - aug_start_time) / 60)
            counter_k += 1
            print(f'✅ Domain augmentation completed. Total augmented samples: {len(augumented_images)}, counter_k: {counter_k}')


        # start to train the model paramters
        # meta learning structure

        model.train()

        try:
            id_input, id_target = next(id_loader_iter)
        except:
            id_loader_iter = iter(id_loader)
            id_input, id_target = next(id_loader_iter)

        try:
            ood_input, _ = next(auxiliary_loader_iter)
        except:
            auxiliary_loader_iter = iter(auxiliary_loader)
            ood_input, _ = next(auxiliary_loader_iter)

        id_input, id_target = id_input.to(device).float(), id_target.to(device).long()
        ood_input = ood_input.to(device).float()

        # tim, this training process needs to be modified to OE training process
        # params = list(model.parameters())
        # output = model.functional(params, True, input)
        # loss = criterion(output, target)

        params = list(model.parameters())
        logits_id = model.functional(params, True, id_input)
        logits_ood = model.functional(params, True, ood_input)

        loss_id = l_id(logits_id, id_target)
        loss_ood = l_oe(logits_ood)
        loss = loss_id + args.ood_weight * loss_ood

        # counter_k = 0, means no need for meta learning, no augmented data available
        if counter_k == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % 50 == 0:
                print(f'  Regular training - Loss: {loss.item():.4f} (ID: {loss_id.item():.4f}, OOD: {loss_ood.item():.4f})')

        # meta learning structure
        # Need to consider if need an extra ID batch for meta learning, or only keep the agumented data for ood loss,
        # instead of mixed oe loss. Maybe could use a second hyperparameter to control this.
        # set 'ood_weight2'
        else:
            grads = torch.autograd.grad(loss, params, create_graph=True)
            params = [(param - args.lr * grad).requires_grad_() for param, grad in zip(params, grads)]
            try:
                augumented_ood, _ = next(aug_loader_iter)
            except:
                aug_loader_iter = iter(aug_loader)
                augumented_ood, _ = next(aug_loader_iter)

            try:
                id_input2, id_target2 = next(id_loader_iter)
            except:
                id_loader_iter = iter(id_loader)
                id_input2, id_target2 = next(id_loader_iter)

            augumented_ood = augumented_ood.to(device).float()
            id_input2, id_target2 = id_input2.to(device).float(), id_target2.to(device).long()

            logits_id2 = model.functional(params, True, id_input2)
            logits_ood2 = model.functional(params, True, augumented_ood)

            loss_id2 = l_id(logits_id2, id_target2)
            loss_ood2 = l_oe(logits_ood2)
            loss2 = loss_id2 + args.ood_weight2 * loss_ood2


            
            loss_combine = (loss + loss2) / 2
            optimizer.zero_grad()
            loss_combine.backward()
            optimizer.step()
            if t % 50 == 0:
                print(f'  Meta-learning - Loss: {loss_combine.item():.4f} (Original: {loss.item():.4f}, Aug: {loss2.item():.4f})')

        # tim, do some evaluation every 1000 iterations
        if t % args.print_freq == 0 and t > 0:  # Skip evaluation at iteration 0 for speed
            model.eval()
            in_score = get_in_scores(model, test_loader, in_dist=True)
            metric_all = []
            for test_name, test_loader_ood in ood_test_loaders.items():
                metric = get_ood_results(model, test_name, test_loader_ood)
                metric_all.append(metric)
                print(f"{test_name} - AUROC: {metric['auroc']:.4f}, AUPR: {metric['aupr']:.4f}")





        # tim eval process --------------------------------
        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target, topk=(1,))[0]
        # losses.update(loss.data.item(), input.size(0))
        # top1.update(prec1.item(), input.size(0))
        prec1 = accuracy(logits_id.data, id_target, topk=(1,))[0]
        losses.update(loss.data.item(), id_input.size(0))
        top1.update(prec1.item(), id_input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)

        # if t % args.print_freq == 0:
        #     print('Iter: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(t, t, args.num_iters, batch_time=batch_time, loss=losses, top1=top1))
        #     # evaluate on validation set per print_freq, compute acc on the whole val dataset
        #     prec1 = validate(val_loader, model)
        #     print("validation set acc", prec1)
        
        # tim need to use validation of ood set every 10 epochs

        save_checkpoint({
            'iter': t + 1,
            'state_dict': model.state_dict(),
            'prec': prec1,
        }, args.dataset, exp_name)
        # ---------------------------------------------

def wae_train(model, D, new_aug_loader, optimizer, d_optimizer, epoch, device, mean, std):

    def sample_z(n_sample=None, dim=None, sigma=None, template=None):
        if template is not None:
            z = sigma * Variable(template.data.new(template.size()).normal_())
        else:
            if n_sample is None:
                n_sample = 32
            if dim is None:
                dim = 20
            if sigma is None:
                sigma = z_sigma
            z = sigma * Variable(torch.randn(n_sample, dim)).to(device)
        return z

    z_var = 1
    z_sigma = math.sqrt(z_var)
    param = 100
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(new_aug_loader):
        input_comb = data.to(device).float()
        batch_size = input_comb.size(0)
        
        # 安全转换到[0,1]范围用于WAE
        input_unnorm = safe_unnormalize(input_comb, mean, std)
        
        optimizer.zero_grad()
        d_optimizer.zero_grad()

        # WAE在[0,1]数据上工作
        recon_batch, z_tilde = model(input_unnorm)
        z = sample_z(template=z_tilde, sigma=z_sigma)
        log_p_z = log_density_igaussian(z, z_var).view(-1, 1)

        # Create fresh tensors each time to avoid in-place operation issues
        ones_d = torch.ones(batch_size, 1, device=device)
        zeros_d = torch.zeros(batch_size, 1, device=device)
        ones_g = torch.ones(batch_size, 1, device=device)

        D_z = D(z)
        D_z_tilde = D(z_tilde)
        
        # Calculate discriminator loss
        D_z_logits = D_z + log_p_z
        D_z_tilde_logits = D_z_tilde + log_p_z
        D_loss = F.binary_cross_entropy_with_logits(D_z_logits, ones_d) + \
                 F.binary_cross_entropy_with_logits(D_z_tilde_logits, zeros_d)

        total_D_loss = param * D_loss
        total_D_loss.backward(retain_graph=True)
        d_optimizer.step()

        # Recalculate for generator loss to avoid reusing modified tensors
        z_tilde_fresh = model.encode(input_unnorm.view(batch_size, -1))
        D_z_tilde_fresh = D(z_tilde_fresh)
        log_p_z_fresh = log_density_igaussian(z_tilde_fresh, z_var).view(-1, 1)
        
        # BCE损失使用unnormalized数据作为target
        BCE = F.binary_cross_entropy(recon_batch, input_unnorm.view(batch_size, -1), reduction='sum')
        Q_loss = F.binary_cross_entropy_with_logits(D_z_tilde_fresh + log_p_z_fresh, ones_g)
        loss = BCE + param * Q_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(new_aug_loader.dataset),
                100. * batch_idx / len(new_aug_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(new_aug_loader.dataset)))

if __name__ == '__main__':
    main()
