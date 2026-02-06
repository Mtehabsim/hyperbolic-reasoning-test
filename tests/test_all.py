#!/usr/bin/env python3
"""
Comprehensive unit tests for the hyperbolic probing codebase.

Run with: python -m pytest tests/ -v
Or: python tests/test_all.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch


# ============================================================================
# DATA TESTS
# ============================================================================

class TestReasoningSample:
    """Test the ReasoningSample dataclass."""
    
    def test_creation(self):
        from src.data.base import ReasoningSample
        
        sample = ReasoningSample(
            id="test_001",
            prompt="Is A a B?",
            answer="Yes",
            depth=2,
            label="TRUE",
        )
        
        assert sample.id == "test_001"
        assert sample.depth == 2
        assert sample.label == "TRUE"
    
    def test_to_dict(self):
        from src.data.base import ReasoningSample
        
        sample = ReasoningSample(
            id="test_001",
            prompt="Test",
            answer="Yes",
            depth=1,
            label="TRUE",
        )
        
        d = sample.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "test_001"


class TestPrOntoQAGenerator:
    """Test PrOntoQA dataset generation."""
    
    def test_generate_sample(self):
        from src.data.prontoqa import PrOntoQAGenerator
        
        gen = PrOntoQAGenerator(seed=42)
        sample = gen.generate_sample(depth=3, label="TRUE")
        
        assert sample.depth == 3
        assert sample.label == "TRUE"
        assert len(sample.prompt) > 0
    
    def test_graph_distances(self):
        from src.data.prontoqa import PrOntoQAGenerator
        
        gen = PrOntoQAGenerator(seed=42)
        sample = gen.generate_sample(depth=3, label="TRUE")
        
        # Should have graph distances
        assert sample.graph_distances is not None
        assert sample.graph_distances.shape[0] == sample.graph_distances.shape[1]
    
    def test_train_test_split(self):
        from src.data.prontoqa import PrOntoQAGenerator
        
        gen = PrOntoQAGenerator(seed=42)
        train, test = gen.generate_train_test_split(
            n_train=10,
            n_test_true=5,
            n_test_false=5,
        )
        
        assert len(train) == 10
        assert len(test) == 10
        
        # Check label distribution
        true_count = sum(1 for s in test.samples if s.label == "TRUE")
        false_count = sum(1 for s in test.samples if s.label == "FALSE")
        assert true_count == 5
        assert false_count == 5


class TestListOpsGenerator:
    """Test ListOps dataset generation."""
    
    def test_generate_sample(self):
        from src.data.listops import ListOpsGenerator
        
        gen = ListOpsGenerator(seed=42)
        sample = gen.generate_sample(depth=2, label="TRUE")
        
        assert sample.depth == 2
        assert sample.label == "TRUE"
    
    def test_operators(self):
        from src.data.listops import ListOpsGenerator
        
        gen = ListOpsGenerator(seed=42)
        
        # Generate multiple samples to check operator variety
        samples = [gen.generate_sample(depth=2) for _ in range(20)]
        
        # Check that we got valid samples with prompts
        for s in samples:
            assert len(s.prompt) > 0
            assert s.answer is not None


# ============================================================================
# GEOMETRY TESTS
# ============================================================================

class TestEuclideanGeometry:
    """Test Euclidean geometry functions."""
    
    def test_pairwise_l2_distance(self):
        from src.geometry.euclidean import pairwise_l2_distance
        
        x = torch.randn(10, 16)
        dist = pairwise_l2_distance(x)
        
        assert dist.shape == (10, 10)
        # Diagonal should be close to zero (allow small numerical error due to float32)
        assert torch.allclose(dist.diag(), torch.zeros(10), atol=0.01)
        assert (dist >= 0).all()

    
    def test_pca_projection(self):
        from src.geometry.euclidean import pca_projection
        
        x = torch.randn(100, 64)
        # pca_projection returns (projected, components, mean) tuple
        result = pca_projection(x, n_components=8)
        projected = result[0]  # First element is projected data
        
        assert projected.shape == (100, 8)


class TestHyperbolicGeometry:
    """Test hyperbolic geometry functions."""
    
    def test_poincare_ball(self):
        from src.geometry.hyperbolic import PoincareBall
        
        ball = PoincareBall(curvature=1.0)
        
        # Test expmap0
        v = torch.randn(10, 16) * 0.1
        z = ball.expmap0(v)
        
        # Points should be inside ball
        norms = torch.norm(z, dim=-1)
        assert (norms < 1.0).all(), f"Points outside ball: max norm = {norms.max()}"
    
    def test_poincare_distance(self):
        from src.geometry.hyperbolic import PoincareBall
        
        ball = PoincareBall(curvature=1.0)
        
        # Create points inside ball
        x = torch.randn(5, 8) * 0.1
        y = torch.randn(5, 8) * 0.1
        
        x = ball.project(x)
        y = ball.project(y)
        
        dist = ball.distance(x, y)
        
        assert dist.shape == (5, 5)
        assert (dist >= 0).all()
    
    def test_lorentz_model(self):
        from src.geometry.hyperbolic import LorentzModel
        
        lorentz = LorentzModel(curvature=1.0)
        
        # Test expmap0 (tangent vectors are d-dimensional, output is d+1)
        v = torch.randn(10, 16) * 0.1
        z = lorentz.expmap0(v)
        
        assert z.shape == (10, 17)  # d+1 dimensions
        
        # Verify on hyperboloid: -t^2 + sum(x^2) = -1/c
        # For c=1: -z[0]^2 + z[1:]^2 = -1
        # Using self.c attribute
        constraint = -z[:, 0]**2 + (z[:, 1:]**2).sum(dim=-1)
        expected = -1.0 / lorentz.c
        assert torch.allclose(constraint, torch.full_like(constraint, expected), atol=1e-4)
    
    def test_mdr_rescaling(self):
        from src.geometry.hyperbolic import maximum_distance_rescaling
        
        x = torch.randn(10, 16) * 100  # Large norms
        
        rescaled = maximum_distance_rescaling(x, max_norm=15.0)
        
        norms = torch.norm(rescaled, dim=-1)
        assert (norms <= 15.0 + 1e-5).all(), f"MDR failed: max norm = {norms.max()}"


class TestMetrics:
    """Test geometric metrics."""
    
    def test_spearman(self):
        from src.geometry.metrics import compute_spearman
        
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Create distance matrices
        pred_dist = torch.abs(pred.unsqueeze(0) - pred.unsqueeze(1))
        target_dist = torch.abs(target.unsqueeze(0) - target.unsqueeze(1))
        
        result = compute_spearman(pred_dist, target_dist)
        
        # compute_spearman returns a dict with 'rho' key
        assert isinstance(result, dict)
        assert "rho" in result
        assert isinstance(result["rho"], float)
    
    def test_distortion(self):
        from src.geometry.metrics import compute_distortion
        
        pred = torch.randn(5, 5).abs()
        target = torch.randn(5, 5).abs()
        
        # Make symmetric
        pred = (pred + pred.T) / 2
        target = (target + target.T) / 2
        
        dist = compute_distortion(pred, target)
        
        assert "avg_distortion" in dist
        assert dist["avg_distortion"] >= 0



# ============================================================================
# PROBE TESTS
# ============================================================================

class TestProbes:
    """Test probe implementations."""
    
    def test_euclidean_probe(self):
        from src.probes import EuclideanPairwiseProbe
        
        probe = EuclideanPairwiseProbe(input_dim=64, output_dim=16)
        
        x = torch.randn(10, 64)
        z = probe(x)
        
        assert z.shape == (10, 16)
        
        dist = probe.pairwise_distances(z)
        assert dist.shape == (10, 10)
        assert (dist >= 0).all()
    
    def test_hyperbolic_probe(self):
        from src.probes import HyperbolicPairwiseProbe
        
        probe = HyperbolicPairwiseProbe(input_dim=64, output_dim=16)
        
        x = torch.randn(10, 64)
        z = probe(x)
        
        assert z.shape == (10, 16)
        
        # Check inside Poincare ball
        norms = torch.norm(z, dim=-1)
        assert (norms < 1.0).all(), f"Outside ball: max norm = {norms.max()}"
        
        dist = probe.pairwise_distances(z)
        assert dist.shape == (10, 10)
        assert (dist >= 0).all()
    
    def test_lorentz_probe(self):
        from src.probes import LorentzProbe
        
        probe = LorentzProbe(input_dim=64, output_dim=16)
        
        x = torch.randn(10, 64)
        z = probe(x)
        
        # Lorentz outputs d+1 dimensions
        assert z.shape == (10, 17)
        
        dist = probe.pairwise_distances(z)
        assert dist.shape == (10, 10)
        assert (dist >= 0).all()
    
    def test_create_probe_factory(self):
        from src.probes import create_probe
        
        euclidean = create_probe("euclidean", 64, 16)
        hyperbolic = create_probe("hyperbolic", 64, 16)
        lorentz = create_probe("lorentz", 64, 16)
        
        x = torch.randn(5, 64)
        
        assert euclidean(x).shape == (5, 16)
        assert hyperbolic(x).shape == (5, 16)
        assert lorentz(x).shape == (5, 17)


class TestProbeTraining:
    """Test probe training utilities."""
    
    def test_train_probe(self):
        from src.probes import create_probe, train_probe
        
        probe = create_probe("euclidean", 32, 8)
        
        hidden = torch.randn(20, 32)
        target = torch.cdist(hidden[:, :3], hidden[:, :3])  # Use first 3 dims as proxy
        
        trained_probe, history = train_probe(
            probe=probe,
            hidden_states=hidden,
            target_distances=target,
            n_epochs=10,
            verbose=False,
            device="cpu",
        )
        
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0


# ============================================================================
# UTILS TESTS
# ============================================================================

class TestReproducibility:
    """Test reproducibility utilities."""
    
    def test_set_seed(self):
        from src.utils.reproducibility import set_seed
        
        set_seed(42)
        a = torch.randn(10)
        
        set_seed(42)
        b = torch.randn(10)
        
        assert torch.allclose(a, b)


class TestTokenSelector:
    """Test token selection."""
    
    def test_find_thinking_positions(self):
        from src.model.token_selector import find_thinking_positions
        
        tokens = ["The", "answer", "is", "therefore", "yes", "."]
        positions = find_thinking_positions(tokens)
        
        assert 3 in positions  # "therefore"
    
    def test_token_selector(self):
        from src.model.token_selector import TokenSelector
        
        selector = TokenSelector(method="thinking_tokens")
        
        tokens = ["Let", "me", "think", ".", "Actually", ",", "the", "answer"]
        hidden = torch.randn(len(tokens), 64)
        
        pooled, positions, method = selector.select(tokens, hidden)
        
        assert pooled.shape[0] == 64  # Should pool to single vector
        assert isinstance(positions, list)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_data_to_probe_pipeline(self):
        """Test full pipeline from data generation to probe training."""
        from src.data.prontoqa import PrOntoQAGenerator
        from src.probes import create_probe, train_probe
        
        # Generate small dataset
        gen = PrOntoQAGenerator(seed=42)
        samples = [gen.generate_sample(depth=d) for d in [1, 2, 3, 2, 1]]
        
        # Mock hidden states (in real use, would come from model)
        hidden_states = torch.randn(len(samples), 64)
        
        # Use depth as proxy for distances
        depths = torch.tensor([s.depth for s in samples], dtype=torch.float32)
        target_distances = torch.abs(depths.unsqueeze(0) - depths.unsqueeze(1))
        
        # Create and train probe
        probe = create_probe("hyperbolic", 64, 8)
        trained_probe, history = train_probe(
            probe=probe,
            hidden_states=hidden_states,
            target_distances=target_distances,
            n_epochs=5,
            verbose=False,
            device="cpu",
        )
        
        # Verify training ran
        assert len(history["train_loss"]) >= 1
        
        # Verify probe outputs valid embeddings
        with torch.no_grad():
            embeddings = trained_probe(hidden_states)
        
        norms = torch.norm(embeddings, dim=-1)
        assert (norms < 1.0).all(), "Embeddings outside Poincaré ball"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
