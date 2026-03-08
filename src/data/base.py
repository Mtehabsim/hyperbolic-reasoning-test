"""
Base classes for dataset generation.

Provides abstract interfaces and common data structures for
PrOntoQA, ListOps, and other hierarchical reasoning datasets.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ReasoningSample:
    """
    Base sample for hierarchical reasoning datasets.
    
    All datasets should produce samples in this format for unified processing.
    """
    id: str
    prompt: str
    answer: str
    depth: int
    label: str  # TRUE or FALSE
    
    # Graph structure
    graph_distances: Optional[np.ndarray] = None  # Pairwise distances matrix
    node_ids: Optional[List[str]] = None  # Node identifiers for distance matrix
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling numpy arrays."""
        d = asdict(self)
        if d["graph_distances"] is not None:
            d["graph_distances"] = d["graph_distances"].tolist()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReasoningSample":
        """Create from dictionary."""
        if d.get("graph_distances") is not None:
            d["graph_distances"] = np.array(d["graph_distances"])
        return cls(**d)


@dataclass
class Dataset:
    """Collection of reasoning samples with metadata."""
    
    name: str
    samples: List[ReasoningSample]
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ReasoningSample:
        return self.samples[idx]
    
    def __iter__(self):
        return iter(self.samples)
    
    @property
    def depths(self) -> List[int]:
        """Get all depth values."""
        return [s.depth for s in self.samples]
    
    @property
    def labels(self) -> List[str]:
        """Get all labels."""
        return [s.label for s in self.samples]
    
    def filter_by_label(self, label: str) -> "Dataset":
        """Return new dataset with only specified label."""
        filtered = [s for s in self.samples if s.label == label]
        return Dataset(
            name=f"{self.name}_{label}",
            samples=filtered,
            config=self.config,
        )
    
    def filter_by_depth(self, min_depth: int = 1, max_depth: int = 10) -> "Dataset":
        """Return new dataset filtered by depth range."""
        filtered = [s for s in self.samples if min_depth <= s.depth <= max_depth]
        return Dataset(
            name=f"{self.name}_d{min_depth}-{max_depth}",
            samples=filtered,
            config=self.config,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        depths = self.depths
        labels = self.labels
        
        # Convert numpy types to native Python for JSON serialization
        if depths:
            unique_depths, depth_counts = np.unique(depths, return_counts=True)
            depth_dist = {int(k): int(v) for k, v in zip(unique_depths, depth_counts)}
        else:
            depth_dist = {}
        
        if labels:
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            label_dist = {str(k): int(v) for k, v in zip(unique_labels, label_counts)}
        else:
            label_dist = {}
        
        return {
            "n_samples": len(self.samples),
            "depth_min": int(min(depths)) if depths else 0,
            "depth_max": int(max(depths)) if depths else 0,
            "depth_mean": float(np.mean(depths)) if depths else 0,
            "depth_distribution": depth_dist,
            "label_distribution": label_dist,
        }

    
    def save(self, path: Union[str, Path]) -> Path:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "name": self.name,
            "config": self.config,
            "statistics": self.get_statistics(),
            "samples": [s.to_dict() for s in self.samples],
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Dataset":
        """Load dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        samples = [ReasoningSample.from_dict(s) for s in data["samples"]]
        return cls(
            name=data["name"],
            samples=samples,
            config=data.get("config", {}),
        )


class DataGenerator(ABC):
    """Abstract base class for dataset generators."""
    
    def __init__(self, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
            config: Generator configuration
        """
        self.seed = seed
        self.config = config or {}
        self.rng = np.random.default_rng(seed)
    
    @abstractmethod
    def generate_sample(self, depth: int, label: str = "TRUE") -> ReasoningSample:
        """Generate a single sample with specified depth and label."""
        pass
    
    def generate_dataset(
        self,
        n_samples: int,
        depth_range: Tuple[int, int] = (1, 5),
        label: str = "TRUE",
        balanced_depth: bool = True,
    ) -> Dataset:
        """
        Generate a dataset with specified parameters.
        
        Args:
            n_samples: Number of samples to generate
            depth_range: (min_depth, max_depth) inclusive
            label: Label for all samples ("TRUE" or "FALSE")
            balanced_depth: If True, balance samples across depths
            
        Returns:
            Dataset object with generated samples
        """
        min_depth, max_depth = depth_range
        depths = list(range(min_depth, max_depth + 1))
        
        samples = []
        for i in range(n_samples):
            if balanced_depth:
                depth = depths[i % len(depths)]
            else:
                depth = self.rng.choice(depths)
            
            sample = self.generate_sample(depth=depth, label=label)
            samples.append(sample)
        
        return Dataset(
            name=self.__class__.__name__,
            samples=samples,
            config={
                "seed": self.seed,
                "n_samples": n_samples,
                "depth_range": depth_range,
                "label": label,
                "generator_config": self.config,
            },
        )
    
    def generate_train_test_split(
        self,
        n_train: int,
        n_test_true: int,
        n_test_false: int,
        depth_range: Tuple[int, int] = (1, 5),
    ) -> Tuple[Dataset, Dataset]:
        """
        Generate train and test datasets.
        
        Args:
            n_train: Number of training samples (TRUE only)
            n_test_true: Number of TRUE test samples
            n_test_false: Number of FALSE test samples
            depth_range: Depth range for samples
            
        Returns:
            (train_dataset, test_dataset)
        """
        # Training: TRUE only
        train = self.generate_dataset(
            n_samples=n_train,
            depth_range=depth_range,
            label="TRUE",
            balanced_depth=True,
        )
        train.name = f"{self.__class__.__name__}_train"
        
        # Test: TRUE + FALSE
        test_true = self.generate_dataset(
            n_samples=n_test_true,
            depth_range=depth_range,
            label="TRUE",
            balanced_depth=True,
        )
        test_false = self.generate_dataset(
            n_samples=n_test_false,
            depth_range=depth_range,
            label="FALSE",
            balanced_depth=True,
        )
        
        # Combine test samples
        test = Dataset(
            name=f"{self.__class__.__name__}_test",
            samples=test_true.samples + test_false.samples,
            config={
                "n_test_true": n_test_true,
                "n_test_false": n_test_false,
                **test_true.config,
            },
        )
        
        return train, test
