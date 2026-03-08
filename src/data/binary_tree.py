"""
Binary Tree Dataset Generator.

Generates binary tree traversal tasks with explicit pairwise distances.
This dataset provides TRUE graph structure (not just depth) for testing
whether hyperbolic embeddings outperform Euclidean on hierarchical data.

Key difference from PrOntoQA:
- PrOntoQA: depth is 1D ordinal (1, 3, 5 hops)
- Binary Tree: Full pairwise tree distances (LCA-based path lengths)

This should show stronger differentiation between Euclidean and hyperbolic
probes, as hyperbolic space naturally embeds trees with low distortion.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .base import DataGenerator, Dataset, ReasoningSample


@dataclass
class BinaryTreeConfig:
    """Configuration for binary tree generation."""
    tree_depth: int = 5  # Depth of the shared tree
    n_samples: int = 500  # Number of node pairs to generate
    seed: int = 42


class BinaryTreeGenerator(DataGenerator):
    """
    Generate binary tree traversal tasks from a SINGLE SHARED TREE.
    
    All samples reference the same underlying tree structure, allowing
    computation of true pairwise distances between any two samples.
    
    Key insight: The pairwise distance matrix captures GRAPH STRUCTURE,
    not just depth. This is what should differentiate hyperbolic from
    Euclidean probes.
    """
    
    def __init__(
        self,
        tree_depth: int = 5,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize binary tree generator.
        
        Args:
            tree_depth: Depth of the binary tree (root = depth 0)
            seed: Random seed for reproducibility
            config: Optional additional configuration
        """
        super().__init__(seed=seed, config=config or {})
        self.tree_depth = tree_depth
        
        # Generate the shared tree structure
        self.nodes, self.children, self.tree_structure = self._generate_shared_tree()
        
    def _generate_shared_tree(self) -> Tuple[List[int], Dict[int, List[int]], str]:
        """
        Generate a single full binary tree that all samples will reference.
        
        Returns:
            (nodes, children_dict, edge_description_string)
        """
        # Nodes are numbered 1, 2, 3, ... (level order)
        num_nodes = (2 ** self.tree_depth) - 1
        nodes = list(range(1, num_nodes + 1))
        
        # Compute parent-child relationships
        children = {}
        for i in nodes:
            left = 2 * i
            right = 2 * i + 1
            if left <= num_nodes:
                children[i] = [left, right] if right <= num_nodes else [left]
            else:
                children[i] = []
        
        # Build edge description (shared across all prompts)
        edge_descriptions = []
        for parent, childs in children.items():
            for child in childs:
                edge_descriptions.append(f"Node {parent} connects to Node {child}")
        
        # Shuffle for variety but keep consistent (seeded)
        self.rng.shuffle(edge_descriptions)
        tree_structure = ". ".join(edge_descriptions)
        
        return nodes, children, tree_structure
    
    def tree_distance(self, node1: int, node2: int) -> int:
        """
        Compute tree distance (path length) between two nodes.
        
        Uses LCA (Lowest Common Ancestor) method:
        dist(a,b) = depth(a) + depth(b) - 2*depth(LCA(a,b))
        
        Args:
            node1: First node ID
            node2: Second node ID
            
        Returns:
            Path length between nodes
        """
        def path_to_root(n: int) -> List[int]:
            path = []
            while n >= 1:
                path.append(n)
                n = n // 2
            return path
        
        path1 = path_to_root(node1)
        path2 = path_to_root(node2)
        
        # Find LCA (lowest common ancestor)
        set1 = set(path1)
        lca = None
        for n in path2:
            if n in set1:
                lca = n
                break
        
        if lca is None:
            return 999  # Should never happen for valid nodes
        
        # Distance = depth(node1) - depth(lca) + depth(node2) - depth(lca)
        dist1 = path1.index(lca)
        dist2 = path2.index(lca)
        return dist1 + dist2
    
    def node_depth(self, node: int) -> int:
        """Compute depth of a node (root = depth 0)."""
        depth = 0
        while node > 1:
            node = node // 2
            depth += 1
        return depth
    
    def generate_sample(self, depth: int = None, label: str = "TRUE") -> ReasoningSample:
        """
        Generate a single sample (node pair query).
        
        Args:
            depth: Ignored for binary tree (depth is derived from node selection)
            label: Ignored for binary tree (all samples are valid traversals)
            
        Returns:
            ReasoningSample with prompt and metadata
        """
        # Sample random pair from the shared tree
        node1, node2 = self.rng.choice(self.nodes, size=2, replace=False)
        # Convert numpy int64 to native Python int for JSON serialization
        node1, node2 = int(node1), int(node2)
        dist = self.tree_distance(node1, node2)
        depth1 = self.node_depth(node1)
        depth2 = self.node_depth(node2)
        
        # Create traversal prompt with the shared tree structure
        prompt = (
            f"Binary tree structure: {self.tree_structure}. "
            f"Find the path from Node {node1} to Node {node2}."
        )
        
        # The "answer" is the path length
        answer = f"Path length: {dist}"
        
        return ReasoningSample(
            id=f"bt_{node1}_{node2}",
            prompt=prompt,
            answer=answer,
            depth=dist,  # Use tree distance as "depth" for compatibility
            label="TRUE",  # All valid traversals
            metadata={
                "node1": node1,
                "node2": node2,
                "depth1": depth1,
                "depth2": depth2,
                "tree_distance": dist,
                "tree_depth": self.tree_depth,
            }
        )
    
    def generate_dataset(
        self,
        n_samples: int = 500,
        depth_range: Tuple[int, int] = None,  # Ignored
        label: str = "TRUE",  # Ignored
        balanced_depth: bool = False,  # Ignored
    ) -> Dataset:
        """
        Generate dataset of node pairs with pairwise tree distances.
        
        This overrides the base class to compute the full pairwise
        distance matrix, which is the key advantage of this dataset.
        
        Args:
            n_samples: Number of node pair queries to generate
            depth_range: Ignored (depth determined by tree structure)
            label: Ignored (all samples are valid)
            balanced_depth: Ignored
            
        Returns:
            Dataset with samples AND pairwise distance matrix
        """
        samples = []
        for i in range(n_samples):
            sample = self.generate_sample()
            sample.id = f"bt_{i:04d}"
            samples.append(sample)
        
        # Compute pairwise distance matrix (key feature!)
        distance_matrix = self.compute_pairwise_distances(samples)
        
        # Attach distance matrix to each sample's metadata
        for i, sample in enumerate(samples):
            sample.graph_distances = distance_matrix
            # Convert to native Python int for JSON serialization
            sample.node_ids = [int(s.metadata["node1"]) for s in samples]
        
        return Dataset(
            name="BinaryTree",
            samples=samples,
            config={
                "seed": self.seed,
                "n_samples": n_samples,
                "tree_depth": self.tree_depth,
                "n_nodes": len(self.nodes),
                "generator_config": self.config,
            },
        )
    
    def compute_pairwise_distances(self, samples: List[ReasoningSample]) -> np.ndarray:
        """
        Compute the TRUE pairwise distance matrix between all samples.
        
        For each pair of samples (i, j), computes the tree distance
        between the primary nodes (node1 of sample i vs node1 of sample j).
        
        This gives the GRAPH STRUCTURE signal that should differentiate
        hyperbolic from Euclidean probes.
        
        Args:
            samples: List of ReasoningSample objects
            
        Returns:
            [N, N] distance matrix
        """
        N = len(samples)
        distance_matrix = np.zeros((N, N), dtype=np.float32)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    # Distance between the queried nodes
                    dist = self.tree_distance(
                        samples[i].metadata["node1"],
                        samples[j].metadata["node1"]
                    )
                    distance_matrix[i, j] = dist
        
        return distance_matrix


def generate_binary_tree_datasets(
    output_dir: Path,
    n_samples: int = 500,
    tree_depth: int = 5,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """
    Generate and save binary tree datasets.
    
    Args:
        output_dir: Directory to save datasets
        n_samples: Number of samples per split
        tree_depth: Depth of the binary tree
        seed: Random seed
        
    Returns:
        Tuple of (train_path, test_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create generator
    generator = BinaryTreeGenerator(
        tree_depth=tree_depth,
        seed=seed,
    )
    
    # Generate datasets
    train_dataset = generator.generate_dataset(n_samples=n_samples)
    train_dataset.name = "BinaryTree_train"
    
    # Test uses different seed but same tree structure
    test_generator = BinaryTreeGenerator(
        tree_depth=tree_depth,
        seed=seed + 1000,
    )
    test_dataset = test_generator.generate_dataset(n_samples=n_samples // 2)
    test_dataset.name = "BinaryTree_test"
    
    # Save
    train_path = train_dataset.save(output_dir / "binary_tree_train.json")
    test_path = test_dataset.save(output_dir / "binary_tree_test.json")
    
    print(f"Generated binary tree datasets:")
    print(f"  Train: {len(train_dataset)} samples -> {train_path}")
    print(f"  Test: {len(test_dataset)} samples -> {test_path}")
    print(f"  Tree depth: {tree_depth}, Nodes: {len(generator.nodes)}")
    
    return train_path, test_path
