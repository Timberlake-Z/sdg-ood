"""
Test suite for OOD evaluation module.

This file contains comprehensive tests for the ood_evaluation.py module,
including unit tests for individual functions and integration tests.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ood_evaluation import (
    stable_cumsum, fpr_and_fdr_at_recall, get_measures, 
    get_ood_scores, evaluate_ood_detection, print_ood_results
)


def test_stable_cumsum():
    """Test the stable cumsum function."""
    print("Testing stable_cumsum...")
    
    # Test basic functionality
    arr = np.array([1, 2, 3, 4, 5])
    result = stable_cumsum(arr)
    expected = np.array([1, 3, 6, 10, 15])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    
    # Test empty array
    arr = np.array([])
    result = stable_cumsum(arr)
    expected = np.array([])
    assert len(result) == 0, f"Expected empty array, got {result}"
    
    print("✓ stable_cumsum tests passed")


def test_get_measures_perfect_separation():
    """Test get_measures with perfectly separable data (AUROC should be 1.0)."""
    print("Testing get_measures with perfect separation...")
    
    # OOD samples have high scores, ID samples have low scores
    pos_scores = np.array([0.8, 0.9, 0.95, 0.99])  # OOD (positive class)
    neg_scores = np.array([0.1, 0.2, 0.3, 0.4])    # ID (negative class)
    
    auroc, aupr, fpr = get_measures(pos_scores, neg_scores)
    
    assert auroc == 1.0, f"Expected AUROC=1.0, got {auroc}"
    assert aupr == 1.0, f"Expected AUPR=1.0, got {aupr}"
    assert fpr == 0.0, f"Expected FPR=0.0, got {fpr}"
    
    print(f"✓ Perfect separation: AUROC={auroc:.4f}, AUPR={aupr:.4f}, FPR={fpr:.4f}")


def test_get_measures_random():
    """Test get_measures with random data (AUROC should be ~0.5)."""
    print("Testing get_measures with random data...")
    
    np.random.seed(42)  # For reproducibility
    pos_scores = np.random.normal(0, 1, 1000)
    neg_scores = np.random.normal(0, 1, 1000)
    
    auroc, aupr, fpr = get_measures(pos_scores, neg_scores)
    
    # For random data, AUROC should be around 0.5
    assert 0.45 <= auroc <= 0.55, f"Expected AUROC~0.5, got {auroc}"
    # AUPR should be around 0.5 for balanced random data
    assert 0.45 <= aupr <= 0.55, f"Expected AUPR~0.5, got {aupr}"
    
    print(f"✓ Random data: AUROC={auroc:.4f}, AUPR={aupr:.4f}, FPR={fpr:.4f}")


def test_get_measures_biased_separation():
    """Test get_measures with biased but separable data."""
    print("Testing get_measures with biased separation...")
    
    # OOD samples centered at 1, ID samples centered at 0
    np.random.seed(42)
    pos_scores = np.random.normal(1, 0.3, 500)    # OOD
    neg_scores = np.random.normal(0, 0.3, 500)    # ID
    
    auroc, aupr, fpr = get_measures(pos_scores, neg_scores)
    
    # Should have high AUROC due to separation
    assert auroc > 0.9, f"Expected AUROC>0.9, got {auroc}"
    assert aupr > 0.9, f"Expected AUPR>0.9, got {aupr}"
    assert fpr < 0.1, f"Expected FPR<0.1, got {fpr}"
    
    print(f"✓ Biased separation: AUROC={auroc:.4f}, AUPR={aupr:.4f}, FPR={fpr:.4f}")


def test_fpr_at_different_recall_levels():
    """Test FPR calculation at different recall levels."""
    print("Testing FPR at different recall levels...")
    
    # Create data with known properties
    pos_scores = np.array([0.9, 0.8, 0.7, 0.6])  # 4 OOD samples
    neg_scores = np.array([0.5, 0.4, 0.3, 0.2])  # 4 ID samples
    
    # Test different recall levels
    _, _, fpr95 = get_measures(pos_scores, neg_scores, recall_level=0.95)
    _, _, fpr80 = get_measures(pos_scores, neg_scores, recall_level=0.80)
    
    # FPR should be lower at higher recall (more restrictive threshold)
    assert fpr95 >= fpr80, f"Expected FPR95 >= FPR80, got FPR95={fpr95}, FPR80={fpr80}"
    
    print(f"✓ FPR at different recalls: FPR95={fpr95:.4f}, FPR80={fpr80:.4f}")


class SimpleModel(nn.Module):
    """Simple model for testing purposes."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(32*32*3, num_classes)
    
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class MockLearner:
    """Mock Learner wrapper to simulate MetaNN Learner."""
    def __init__(self, model):
        self.module = model
    
    def eval(self):
        self.module.eval()


def test_get_ood_scores_format():
    """Test that get_ood_scores returns correct format and shape."""
    print("Testing get_ood_scores format...")
    
    # Create mock model and data
    model = MockLearner(SimpleModel(num_classes=10))
    device = torch.device('cpu')
    
    # Create synthetic data - ensure we have enough samples for the test
    data = torch.randn(120, 3, 32, 32)  # Increased to ensure we have enough
    targets = torch.randint(0, 10, (120,))
    dataset = TensorDataset(data, targets)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    
    # Test in-distribution scores (should use all samples)
    in_scores = get_ood_scores(model, loader, device, in_dist=True, ood_num_examples=50, test_bs=20)
    assert len(in_scores) == 120, f"Expected 120 ID scores, got {len(in_scores)}"
    
    # Test OOD scores (should be limited to ood_num_examples)
    # With 120 samples, batch_size=20, we have 6 batches
    # ood_num_examples=50, test_bs=20, so we process 50//20=2 full batches (40 samples)
    # Then take the first 50 samples, but we only have 40, so we get 40
    ood_scores = get_ood_scores(model, loader, device, in_dist=False, ood_num_examples=50, test_bs=20)
    # We expect min(50, actual_samples_processed)
    expected_len = min(50, 40)  # 2 full batches * 20 = 40 samples
    assert len(ood_scores) == expected_len, f"Expected {expected_len} OOD scores, got {len(ood_scores)}"
    
    # Check that scores are negative (since we use -max(softmax))
    assert np.all(in_scores <= 0), "Expected all scores to be negative or zero"
    assert np.all(ood_scores <= 0), "Expected all scores to be negative or zero"
    
    print(f"✓ get_ood_scores format: ID={len(in_scores)}, OOD={len(ood_scores)} samples")


def test_evaluate_ood_detection_integration():
    """Integration test for the complete evaluation pipeline."""
    print("Testing evaluate_ood_detection integration...")
    
    # Create mock model and data
    model = MockLearner(SimpleModel(num_classes=10))
    device = torch.device('cpu')
    
    # Create ID test data
    id_data = torch.randn(200, 3, 32, 32)
    id_targets = torch.randint(0, 10, (200,))
    id_dataset = TensorDataset(id_data, id_targets)
    test_loader = DataLoader(id_dataset, batch_size=20, shuffle=False)
    
    # Create multiple OOD datasets
    ood_datasets = {}
    for name in ['texture', 'places365']:
        ood_data = torch.randn(300, 3, 32, 32)
        ood_targets = torch.randint(0, 10, (300,))
        ood_dataset = TensorDataset(ood_data, ood_targets)
        ood_datasets[name] = DataLoader(ood_dataset, batch_size=20, shuffle=False)
    
    # Run evaluation
    results = evaluate_ood_detection(
        model, test_loader, ood_datasets, device, 
        ood_num_examples=100, test_bs=20
    )
    
    # Check results format
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert 'texture' in results, "Missing 'texture' in results"
    assert 'places365' in results, "Missing 'places365' in results"
    
    # Check each result has 3 metrics
    for dataset_name, metrics in results.items():
        assert len(metrics) == 3, f"Expected 3 metrics for {dataset_name}, got {len(metrics)}"
        auroc, aupr, fpr = metrics
        assert 0 <= auroc <= 1, f"AUROC out of range for {dataset_name}: {auroc}"
        assert 0 <= aupr <= 1, f"AUPR out of range for {dataset_name}: {aupr}"
        assert 0 <= fpr <= 1, f"FPR out of range for {dataset_name}: {fpr}"
    
    print("✓ Integration test passed")
    print_ood_results(results, 100)


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("Testing edge cases...")
    
    # Test with identical scores (should not crash)
    pos_scores = np.array([0.5, 0.5, 0.5, 0.5])
    neg_scores = np.array([0.5, 0.5, 0.5, 0.5])
    
    try:
        auroc, aupr, fpr = get_measures(pos_scores, neg_scores)
        # AUROC should be 0.5 for identical distributions
        assert 0.4 <= auroc <= 0.6, f"Expected AUROC~0.5 for identical scores, got {auroc}"
        print(f"✓ Identical scores handled: AUROC={auroc:.4f}")
    except Exception as e:
        print(f"✗ Error with identical scores: {e}")
        raise
    
    # Test with single sample
    pos_scores = np.array([0.8])
    neg_scores = np.array([0.2])
    
    try:
        auroc, aupr, fpr = get_measures(pos_scores, neg_scores)
        assert auroc == 1.0, f"Expected AUROC=1.0 for single sample, got {auroc}"
        print(f"✓ Single sample handled: AUROC={auroc:.4f}")
    except Exception as e:
        print(f"✗ Error with single sample: {e}")
        raise
    
    print("✓ Edge cases passed")


def run_all_tests():
    """Run all tests in sequence."""
    print("=" * 60)
    print("Running OOD Evaluation Test Suite")
    print("=" * 60)
    
    try:
        test_stable_cumsum()
        test_get_measures_perfect_separation()
        test_get_measures_random()
        test_get_measures_biased_separation()
        test_fpr_at_different_recall_levels()
        test_get_ood_scores_format()
        test_evaluate_ood_detection_integration()
        test_edge_cases()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()