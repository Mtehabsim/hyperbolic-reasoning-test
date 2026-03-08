"""
ListOps dataset generator.

Based on Nangia & Bowman (2018) and the Long Range Arena benchmark.
Generates nested operator expressions with controllable depth.

Example:
    [MAX [MIN 4 2] [SUM 3 1 2]] -> 6
    depth = 2 (two levels of nesting)
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .base import DataGenerator, Dataset, ReasoningSample


class ListOpsGenerator(DataGenerator):
    """
    Generator for ListOps-style nested operator expressions.
    
    Produces samples like:
        "[MAX [MIN 4 2] [SUM 3 1 2]]"
        Answer: 6 (MAX(MIN(4,2), SUM(3,1,2)) = MAX(2,6) = 6)
    """
    
    # Available operators
    OPERATORS = ["MAX", "MIN", "SUM", "MED"]  # MED = median
    
    # Value range
    MIN_VALUE = 0
    MAX_VALUE = 9
    
    def __init__(
        self,
        seed: int = 42,
        operators: Optional[List[str]] = None,
        value_range: Tuple[int, int] = (0, 9),
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ListOps generator.
        
        Args:
            seed: Random seed
            operators: List of operators to use
            value_range: (min, max) for integer values
            config: Additional configuration
        """
        super().__init__(seed=seed, config=config)
        
        self.operators = operators or self.OPERATORS.copy()
        self.value_range = value_range
        self._sample_counter = 0
        self._node_counter = 0
    
    def _reset_node_counter(self):
        """Reset node counter for new sample."""
        self._node_counter = 0
    
    def _get_node_id(self) -> str:
        """Get unique node ID."""
        self._node_counter += 1
        return f"n{self._node_counter}"
    
    def _generate_expression(
        self,
        depth: int,
        graph: nx.DiGraph,
        parent_id: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Recursively generate a nested expression.
        
        Args:
            depth: Current remaining depth
            graph: Graph to add nodes to
            parent_id: Parent node ID
            
        Returns:
            (expression_string, computed_value)
        """
        if depth == 0:
            # Base case: generate a value
            val = int(self.rng.integers(self.value_range[0], self.value_range[1] + 1))
            node_id = self._get_node_id()
            graph.add_node(node_id, type="value", value=val, depth=depth)
            if parent_id:
                graph.add_edge(parent_id, node_id)
            return str(val), val
        
        # Recursive case: generate an operator with children
        op = self.rng.choice(self.operators)
        node_id = self._get_node_id()
        graph.add_node(node_id, type="operator", op=op, depth=depth)
        
        if parent_id:
            graph.add_edge(parent_id, node_id)
        
        # Generate 2-4 children
        n_children = int(self.rng.integers(2, 5))
        children_exprs = []
        children_vals = []
        
        for _ in range(n_children):
            # Randomly decide if child should be deeper or a value
            if depth > 1 and self.rng.random() > 0.3:
                child_depth = depth - 1
            else:
                child_depth = 0
            
            expr, val = self._generate_expression(child_depth, graph, node_id)
            children_exprs.append(expr)
            children_vals.append(val)
        
        # Compute result
        if op == "MAX":
            result = max(children_vals)
        elif op == "MIN":
            result = min(children_vals)
        elif op == "SUM":
            result = sum(children_vals)
        elif op == "MED":
            result = int(np.median(children_vals))
        else:
            result = children_vals[0]  # Fallback
        
        # Build expression string
        children_str = " ".join(children_exprs)
        expr = f"[{op} {children_str}]"
        
        return expr, result
    
    def _compute_graph_distances(self, G: nx.DiGraph) -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise distances in the parse tree."""
        nodes = list(G.nodes())
        n = len(nodes)
        
        G_undirected = G.to_undirected()
        
        distances = np.zeros((n, n))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i == j:
                    distances[i, j] = 0
                else:
                    try:
                        distances[i, j] = nx.shortest_path_length(G_undirected, u, v)
                    except nx.NetworkXNoPath:
                        distances[i, j] = float('inf')
        
        return distances, nodes
    
    def generate_sample(self, depth: int, label: str = "TRUE") -> ReasoningSample:
        """
        Generate a single ListOps sample.
        
        Args:
            depth: Maximum nesting depth
            label: "TRUE" for correct answer, "FALSE" for incorrect
            
        Returns:
            ReasoningSample with parse tree and distances
        """
        self._sample_counter += 1
        self._reset_node_counter()
        
        # Build parse tree
        G = nx.DiGraph()
        expression, correct_answer = self._generate_expression(depth, G)
        
        # For FALSE samples, modify the answer
        if label == "FALSE":
            # Add or subtract to make it wrong
            offset = int(self.rng.choice([-2, -1, 1, 2]))
            wrong_answer = correct_answer + offset
            answer = str(wrong_answer)
        else:
            answer = str(correct_answer)
        
        # Build prompt
        prompt = f"Expression: {expression}\n\nWhat is the result?"
        
        # Compute graph distances
        graph_distances, node_ids = self._compute_graph_distances(G)
        
        return ReasoningSample(
            id=f"listops_{self._sample_counter:06d}",
            prompt=prompt,
            answer=answer,
            depth=depth,
            label=label,
            graph_distances=graph_distances,
            node_ids=node_ids,
            metadata={
                "expression": expression,
                "correct_answer": correct_answer,
                "n_nodes": len(node_ids),
            },
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ListOpsGenerator":
        """Create generator from configuration dictionary."""
        return cls(
            seed=config.get("seed", 42),
            operators=config.get("operators"),
            value_range=tuple(config.get("value_range", (0, 9))),
            config=config,
        )


def generate_listops_datasets(
    output_dir: str,
    n_train: int = 1500,
    n_test: int = 500,
    depth_range: Tuple[int, int] = (1, 6),
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Generate and save ListOps train/test datasets.
    
    Args:
        output_dir: Directory to save datasets
        n_train: Number of training samples
        n_test: Number of test samples
        depth_range: (min_depth, max_depth)
        seed: Random seed
        
    Returns:
        (train_dataset, test_dataset)
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = ListOpsGenerator(seed=seed)
    
    # Generate train (TRUE only)
    train = generator.generate_dataset(
        n_samples=n_train,
        depth_range=depth_range,
        label="TRUE",
        balanced_depth=True,
    )
    train.name = "ListOps_train"
    
    # Generate test (TRUE only for this dataset)
    test = generator.generate_dataset(
        n_samples=n_test,
        depth_range=depth_range,
        label="TRUE",
        balanced_depth=True,
    )
    test.name = "ListOps_test"
    
    # Save datasets
    train.save(output_dir / "listops_train.json")
    test.save(output_dir / "listops_test.json")
    
    return train, test
