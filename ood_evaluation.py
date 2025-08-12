"""
OOD Detection Evaluation Module

This module provides standard evaluation functions for Out-of-Distribution (OOD) detection,
including AUROC, AUPR, and FPR@95 metrics calculation.

Author: Adapted from standard OOD detection implementations
"""

import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as sk

# Default recall level for FPR calculation
recall_level_default = 0.95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """
    Use high precision for cumsum and check that final value matches sum.
    
    Args:
        arr: Input array
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Cumulative sum with high precision
        
    Raises:
        RuntimeError: If cumsum is unstable
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    
    # Handle empty array case
    if len(out) == 0:
        return out
    
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                         'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    """
    Calculate False Positive Rate (FPR) at a given recall level.
    
    Args:
        y_true: True binary labels (0 for negative class, 1 for positive class)
        y_score: Target scores (higher scores indicate positive class)
        recall_level: The recall level at which to compute FPR
        pos_label: Label of positive class
        
    Returns:
        FPR at the given recall level
    """
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    """
    Calculate AUROC, AUPR, and FPR@95 metrics for OOD detection.
    
    Args:
        _pos: Scores for positive class (OOD samples)
        _neg: Scores for negative class (ID samples)  
        recall_level: Recall level for FPR calculation
        
    Returns:
        tuple: (AUROC, AUPR, FPR@recall_level)
    """
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1  # OOD samples labeled as 1, ID samples as 0

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def get_ood_scores(model, loader, device, in_dist=False, ood_num_examples=2000, test_bs=200):
    """
    Calculate OOD detection scores using negative max softmax probability.
    
    Args:
        model: The neural network model
        loader: Data loader (ID or OOD)
        device: Device to run inference on
        in_dist: Whether this is in-distribution data (if False, limits sample count)
        ood_num_examples: Maximum number of OOD examples to process
        test_bs: Test batch size
        
    Returns:
        np.ndarray: OOD detection scores (higher = more likely OOD)
    """
    _score = []
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            # For OOD data, limit the number of samples
            if batch_idx >= ood_num_examples // test_bs and in_dist is False:
                break
            
            data = data.to(device)
            # output = model.module(data)  # Use model.module for Learner wrapper
            output = model(data)
            smax = F.softmax(output, dim=1)
            # Use negative max softmax probability as OOD score
            # ID samples: high confidence -> low score
            # OOD samples: low confidence -> high score
            _score.append(-torch.max(smax, dim=1)[0].cpu().numpy())
    
    scores = np.concatenate(_score)
    
    if in_dist:
        return scores  # Return all ID scores
    else:
        return scores[:ood_num_examples]  # Limit OOD scores to exact count


def evaluate_ood_detection(model, test_loader, ood_test_loaders, device, ood_num_examples, test_bs):
    """
    Complete OOD detection evaluation pipeline.
    
    Args:
        model: The neural network model
        test_loader: ID test data loader
        ood_test_loaders: Dictionary of OOD test loaders
        device: Device to run inference on
        ood_num_examples: Number of OOD examples per dataset
        test_bs: Test batch size
        
    Returns:
        dict: Results dictionary with format {dataset_name: (auroc, aupr, fpr95)}
    """
    model.eval()
    
    # Get ID (in-distribution) scores
    in_scores = get_ood_scores(model, test_loader, device, in_dist=True, 
                              ood_num_examples=ood_num_examples, test_bs=test_bs)
    
    results = {}
    
    # Evaluate each OOD dataset
    for test_name, ood_loader in ood_test_loaders.items():
        # Get OOD scores
        out_scores = get_ood_scores(model, ood_loader, device, in_dist=False,
                                   ood_num_examples=ood_num_examples, test_bs=test_bs)
        
        # Calculate metrics
        auroc, aupr, fpr95 = get_measures(out_scores, in_scores)
        
        results[test_name] = (auroc, aupr, fpr95)
    
    return results


def print_ood_results(results, ood_num_examples):
    """
    Print OOD detection results in a formatted table.
    
    Args:
        results: Results dictionary from evaluate_ood_detection
        ood_num_examples: Number of OOD examples used
    """
    print(f"\nOOD Detection Results (using {ood_num_examples} OOD samples per dataset):")
    print("-" * 65)
    print(f"{'Dataset':>12} {'AUROC':>8} {'AUPR':>8} {'FPR95':>8}")
    print("-" * 65)
    
    for dataset_name, (auroc, aupr, fpr95) in results.items():
        print(f"{dataset_name:>12} {auroc:>8.4f} {aupr:>8.4f} {fpr95:>8.4f}")
    
    print("-" * 65)