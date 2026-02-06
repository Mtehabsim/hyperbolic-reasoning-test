"""
Tests for data generators.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.data.base import ReasoningSample, Dataset, DataGenerator
from src.data.prontoqa import PrOntoQAGenerator
from src.data.listops import ListOpsGenerator


class TestReasoningSample:
    """Tests for ReasoningSample dataclass."""
    
    def test_sample_creation(self):
        """Test basic sample creation."""
        sample = ReasoningSample(
            id="test_001",
            prompt="Test prompt",
            answer="Yes",
            depth=3,
            label="TRUE",
        )
        assert sample.id == "test_001"
        assert sample.depth == 3
        assert sample.label == "TRUE"
    
    def test_sample_with_distances(self):
        """Test sample with graph distances."""
        distances = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        sample = ReasoningSample(
            id="test_002",
            prompt="Test",
            answer="Yes",
            depth=2,
            label="TRUE",
            graph_distances=distances,
            node_ids=["a", "b", "c"],
        )
        assert sample.graph_distances.shape == (3, 3)
        assert len(sample.node_ids) == 3
    
    def test_to_dict_and_back(self):
        """Test serialization round-trip."""
        distances = np.array([[0, 1], [1, 0]])
        sample = ReasoningSample(
            id="test_003",
            prompt="Test",
            answer="No",
            depth=1,
            label="FALSE",
            graph_distances=distances,
            node_ids=["x", "y"],
        )
        
        d = sample.to_dict()
        restored = ReasoningSample.from_dict(d)
        
        assert restored.id == sample.id
        assert restored.label == sample.label
        np.testing.assert_array_equal(restored.graph_distances, sample.graph_distances)


class TestDataset:
    """Tests for Dataset class."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        samples = [
            ReasoningSample(id=f"s{i}", prompt="p", answer="a", depth=i % 3 + 1, label="TRUE")
            for i in range(10)
        ]
        ds = Dataset(name="test", samples=samples)
        
        assert len(ds) == 10
        assert ds.name == "test"
    
    def test_filter_by_label(self):
        """Test filtering by label."""
        samples = [
            ReasoningSample(id=f"s{i}", prompt="p", answer="a", depth=1, label="TRUE" if i < 5 else "FALSE")
            for i in range(10)
        ]
        ds = Dataset(name="test", samples=samples)
        
        true_ds = ds.filter_by_label("TRUE")
        assert len(true_ds) == 5
        assert all(s.label == "TRUE" for s in true_ds)
    
    def test_filter_by_depth(self):
        """Test filtering by depth."""
        samples = [
            ReasoningSample(id=f"s{i}", prompt="p", answer="a", depth=i + 1, label="TRUE")
            for i in range(5)
        ]
        ds = Dataset(name="test", samples=samples)
        
        filtered = ds.filter_by_depth(min_depth=2, max_depth=4)
        assert len(filtered) == 3
    
    def test_save_and_load(self):
        """Test save/load round-trip."""
        samples = [
            ReasoningSample(
                id=f"s{i}",
                prompt="p",
                answer="a",
                depth=1,
                label="TRUE",
                graph_distances=np.array([[0, 1], [1, 0]]),
                node_ids=["a", "b"],
            )
            for i in range(3)
        ]
        ds = Dataset(name="test", samples=samples, config={"seed": 42})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            ds.save(path)
            
            loaded = Dataset.load(path)
            assert len(loaded) == 3
            assert loaded.name == "test"
            assert loaded.config["seed"] == 42


class TestPrOntoQAGenerator:
    """Tests for PrOntoQA generator."""
    
    def test_generator_init(self):
        """Test generator initialization."""
        gen = PrOntoQAGenerator(seed=42)
        assert len(gen.entities) > 0
        assert len(gen.subject_names) > 0
    
    def test_generate_sample_true(self):
        """Test generating TRUE sample."""
        gen = PrOntoQAGenerator(seed=42)
        sample = gen.generate_sample(depth=3, label="TRUE")
        
        assert sample.depth == 3
        assert sample.label == "TRUE"
        assert sample.answer == "Yes"
        assert "Facts:" in sample.prompt
        assert "Question:" in sample.prompt
        assert sample.graph_distances is not None
    
    def test_generate_sample_false(self):
        """Test generating FALSE sample."""
        gen = PrOntoQAGenerator(seed=42)
        sample = gen.generate_sample(depth=2, label="FALSE")
        
        assert sample.label == "FALSE"
        assert sample.answer == "No"
    
    def test_graph_distances_shape(self):
        """Test graph distance matrix shape."""
        gen = PrOntoQAGenerator(seed=42)
        sample = gen.generate_sample(depth=3)
        
        n_nodes = len(sample.node_ids)
        assert sample.graph_distances.shape == (n_nodes, n_nodes)
        # Diagonal should be zero
        np.testing.assert_array_equal(np.diag(sample.graph_distances), 0)
    
    def test_generate_dataset(self):
        """Test generating full dataset."""
        gen = PrOntoQAGenerator(seed=42)
        ds = gen.generate_dataset(n_samples=20, depth_range=(1, 4))
        
        assert len(ds) == 20
        assert min(ds.depths) >= 1
        assert max(ds.depths) <= 4
    
    def test_balanced_depths(self):
        """Test that depths are balanced."""
        gen = PrOntoQAGenerator(seed=42)
        ds = gen.generate_dataset(n_samples=100, depth_range=(1, 5), balanced_depth=True)
        
        from collections import Counter
        depth_counts = Counter(ds.depths)
        # Should be roughly balanced (20 per depth for 5 depths)
        assert all(15 <= c <= 25 for c in depth_counts.values())
    
    def test_reproducibility(self):
        """Test that same seed produces same output."""
        gen1 = PrOntoQAGenerator(seed=42)
        gen2 = PrOntoQAGenerator(seed=42)
        
        s1 = gen1.generate_sample(depth=2)
        s2 = gen2.generate_sample(depth=2)
        
        # Reset counter for fair comparison
        assert s1.prompt == s2.prompt
        assert s1.answer == s2.answer


class TestListOpsGenerator:
    """Tests for ListOps generator."""
    
    def test_generator_init(self):
        """Test generator initialization."""
        gen = ListOpsGenerator(seed=42)
        assert len(gen.operators) == 4
    
    def test_generate_sample(self):
        """Test generating sample."""
        gen = ListOpsGenerator(seed=42)
        sample = gen.generate_sample(depth=2, label="TRUE")
        
        assert sample.depth == 2
        assert sample.label == "TRUE"
        assert "Expression:" in sample.prompt
        assert sample.graph_distances is not None
    
    def test_expression_format(self):
        """Test that expression has correct format."""
        gen = ListOpsGenerator(seed=42)
        sample = gen.generate_sample(depth=2)
        
        expr = sample.metadata["expression"]
        # Should start with [ and end with ]
        assert expr.startswith("[")
        assert expr.endswith("]")
        # Should contain an operator
        assert any(op in expr for op in gen.operators)
    
    def test_generate_dataset(self):
        """Test generating full dataset."""
        gen = ListOpsGenerator(seed=42)
        ds = gen.generate_dataset(n_samples=20, depth_range=(1, 3))
        
        assert len(ds) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
