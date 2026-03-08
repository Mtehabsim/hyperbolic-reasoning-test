"""
PrOntoQA dataset generator.

Based on Saparov & He (NeurIPS 2022):
"Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought"

Generates formal deductive reasoning samples with:
- Fictional entities (wumpus, zumpus, etc.)
- Explicit reasoning chains with controllable depth
- Ground-truth proof graphs with pairwise distances
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .base import DataGenerator, Dataset, ReasoningSample


class PrOntoQAGenerator(DataGenerator):
    """
    Generator for PrOntoQA-style formal reasoning samples.
    
    Produces samples like:
        Facts: "Every wumpus is a zumpus. Every zumpus is a dumpus. Alex is a wumpus."
        Query: "Is Alex a dumpus?"
        Answer: "Yes" (depth=2)
    """
    
    # Fictional entity categories (from original PrOntoQA paper)
    DEFAULT_ENTITIES = [
        "wumpus", "yumpus", "zumpus", "dumpus", "rompus",
        "impus", "tumpus", "vumpus", "jompus", "numpus",
        "bompus", "lorpus", "sterpus", "grimpus", "flumpus",
    ]
    
    # Entity names
    DEFAULT_SUBJECT_NAMES = ["Alex", "Sam", "Max", "Pat", "Chris"]
    
    def __init__(
        self,
        seed: int = 42,
        entities: Optional[List[str]] = None,
        subject_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PrOntoQA generator.
        
        Args:
            seed: Random seed for reproducibility
            entities: List of fictional entity types (default: wumpus, zumpus, etc.)
            subject_names: List of subject names (default: Alex, Sam, etc.)
            config: Additional configuration
        """
        super().__init__(seed=seed, config=config)
        
        self.entities = entities or self.DEFAULT_ENTITIES.copy()
        self.subject_names = subject_names or self.DEFAULT_SUBJECT_NAMES.copy()
        self._sample_counter = 0
    
    def _select_entities(self, n: int) -> List[str]:
        """Select n unique entities for a reasoning chain."""
        if n > len(self.entities):
            raise ValueError(f"Requested {n} entities but only {len(self.entities)} available")
        
        indices = self.rng.choice(len(self.entities), size=n, replace=False)
        return [self.entities[i] for i in indices]
    
    def _build_chain(self, depth: int) -> Tuple[List[str], List[str], str, str, nx.DiGraph]:
        """
        Build a reasoning chain with specified depth.
        
        Args:
            depth: Number of reasoning hops (1-5 typical)
            
        Returns:
            (facts, entity_chain, subject, target, proof_graph)
        """
        # Select entities for the chain (need depth + 1 entities for depth hops)
        entity_chain = self._select_entities(depth + 1)
        
        # Select subject name
        subject = self.rng.choice(self.subject_names)
        
        # Build facts: "Every X is a Y"
        facts = []
        for i in range(depth):
            facts.append(f"Every {entity_chain[i]} is a {entity_chain[i+1]}.")
        
        # Base fact: "Subject is a X"
        facts.append(f"{subject} is a {entity_chain[0]}.")
        
        # Shuffle facts (except keep structure for graph)
        shuffled_facts = facts.copy()
        self.rng.shuffle(shuffled_facts)
        
        # Target entity (what we're asking about)
        target = entity_chain[depth]
        
        # Build proof graph
        G = nx.DiGraph()
        
        # Add nodes with depth
        G.add_node(subject, depth=0, type="subject")
        for i, entity in enumerate(entity_chain):
            G.add_node(entity, depth=i, type="entity")
        
        # Add edges (reasoning steps)
        G.add_edge(subject, entity_chain[0], relation="is_a")
        for i in range(depth):
            G.add_edge(entity_chain[i], entity_chain[i+1], relation="subsumes")
        
        return shuffled_facts, entity_chain, subject, target, G
    
    def _inject_contradiction(
        self,
        facts: List[str],
        entity_chain: List[str],
        depth: int,
    ) -> List[str]:
        """
        Inject a contradiction into the facts to create a FALSE sample.
        
        Strategy: Negate one of the subsumption relations.
        """
        modified_facts = facts.copy()
        
        # Find a subsumption fact to negate
        pattern = re.compile(r"Every (\w+) is a (\w+)\.")
        
        for i, fact in enumerate(modified_facts):
            match = pattern.match(fact)
            if match:
                entity_a, entity_b = match.groups()
                # Replace with negation or different entity
                if self.rng.random() < 0.5:
                    # Negation: "Every X is not a Y" -> makes chain invalid
                    modified_facts[i] = f"Every {entity_a} is not a {entity_b}."
                else:
                    # Replace with unrelated entity
                    available = [e for e in self.entities if e not in entity_chain]
                    if available:
                        new_entity = self.rng.choice(available)
                        modified_facts[i] = f"Every {entity_a} is a {new_entity}."
                break  # Only modify one fact
        
        return modified_facts
    
    def _compute_graph_distances(self, G: nx.DiGraph) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise shortest path distances in the proof graph.
        
        Returns:
            (distance_matrix, node_list)
        """
        nodes = list(G.nodes())
        n = len(nodes)
        
        # Use undirected graph for distances
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
        Generate a single PrOntoQA sample.
        
        Args:
            depth: Reasoning depth (number of hops)
            label: "TRUE" for valid reasoning, "FALSE" for contradiction
            
        Returns:
            ReasoningSample with proof graph and distances
        """
        self._sample_counter += 1
        
        # Build chain
        facts, entity_chain, subject, target, G = self._build_chain(depth)
        
        # For FALSE samples, inject contradiction
        if label == "FALSE":
            facts = self._inject_contradiction(facts, entity_chain, depth)
        
        # Build prompt
        facts_text = " ".join(facts)
        query = f"Is {subject} a {target}?"
        prompt = f"Facts: {facts_text}\n\nQuestion: {query}"
        
        # Answer
        answer = "Yes" if label == "TRUE" else "No"
        
        # Compute graph distances
        graph_distances, node_ids = self._compute_graph_distances(G)
        
        return ReasoningSample(
            id=f"prontoqa_{self._sample_counter:06d}",
            prompt=prompt,
            answer=answer,
            depth=depth,
            label=label,
            graph_distances=graph_distances,
            node_ids=node_ids,
            metadata={
                "subject": subject,
                "target": target,
                "entity_chain": entity_chain,
                "n_facts": len(facts),
                "facts": facts,
            },
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PrOntoQAGenerator":
        """Create generator from configuration dictionary."""
        return cls(
            seed=config.get("seed", 42),
            entities=config.get("entities"),
            subject_names=config.get("subject_names"),
            config=config,
        )


def generate_prontoqa_datasets(
    output_dir: str,
    n_train: int = 1500,
    n_test_true: int = 300,
    n_test_false: int = 300,
    depth_range: Tuple[int, int] = (1, 5),
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Generate and save PrOntoQA train/test datasets.
    
    Args:
        output_dir: Directory to save datasets
        n_train: Number of training samples (TRUE only)
        n_test_true: Number of TRUE test samples
        n_test_false: Number of FALSE test samples
        depth_range: (min_depth, max_depth)
        seed: Random seed
        
    Returns:
        (train_dataset, test_dataset)
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = PrOntoQAGenerator(seed=seed)
    
    train, test = generator.generate_train_test_split(
        n_train=n_train,
        n_test_true=n_test_true,
        n_test_false=n_test_false,
        depth_range=depth_range,
    )
    
    # Save datasets
    train.save(output_dir / "prontoqa_train.json")
    test.save(output_dir / "prontoqa_test.json")
    
    return train, test
