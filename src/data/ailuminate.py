"""AILuminate (MLCommons) harm-taxonomy dataset -- SYNTHETIC STAND-IN.

This mirrors src/data/prontoqa.py in structure, but the "hierarchy" here is a
real BRANCHING harm taxonomy (family -> hazard) rather than PrOntoQA's 1-D
reasoning-depth chain. That distinction is the whole point of the H1.5 experiment:
a branching taxonomy is where hyperbolic geometry can genuinely beat Euclidean
(a tree needs exponential room; flat space can't give it), whereas a depth ruler
fits both geometries equally.

WHAT THIS IS / IS NOT (be honest):
  * The TAXONOMY STRUCTURE is faithful to AILuminate's published 12-hazard set,
    grouped into 3 families -- so real prompts drop in later with no code change:
    just replace the placeholder `prompt` strings (and, if you have finer
    sub-categories, extend `label_path`). Everything downstream (label_path,
    tree distance, probes) already works.
  * The PROMPTS are placeholders (templated per hazard). This lets the full
    pipeline (extract -> H1 -> H1.5) run end-to-end NOW; it is NOT real harm data
    and results on it are pipeline validation, not science.

IMPORTANT CAVEAT (from the geometry): this taxonomy is only 2 levels deep
(family -> hazard). Shallow trees embed fine in Euclidean space too, so hyperbolic
may NOT show an advantage here even on real data -- deep/bushy taxonomies are where
it wins. That is an empirical question H1.5 answers, not an assumption. If you have
AILuminate's finer sub-hazards, add them as a 3rd path level to deepen the tree.

Each sample stores its taxonomy route in ``metadata["label_path"]`` = a list of
integer node ids from root -> leaf, e.g. [family_idx, hazard_idx]. H1.5 reads this
to build the tree-distance target via shared-prefix tree distance.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .base import Dataset, ReasoningSample

# AILuminate v1.0 hazard taxonomy: 3 families -> 12 hazard categories.
# (Grouping is the standard physical / non-physical / contextual split.)
AILUMINATE_TAXONOMY: Dict[str, List[str]] = {
    "physical_hazards": [
        "violent_crimes",
        "sex_related_crimes",
        "child_sexual_exploitation",
        "suicide_self_harm",
        "indiscriminate_weapons_cbrne",
    ],
    "non_physical_hazards": [
        "intellectual_property",
        "defamation",
        "non_violent_crimes",
        "hate",
        "privacy",
    ],
    "contextual_hazards": [
        "specialized_advice",
        "sexual_content",
    ],
}

# A couple of neutral prompt templates per hazard (placeholders; swap for real
# AILuminate prompts later). Deliberately generic so this file carries no actual
# harmful content -- it is a structural stand-in only.
_TEMPLATES = [
    "This is a placeholder prompt in the {hazard} category (sample {k}).",
    "Category={hazard}: synthetic stand-in prompt number {k} for pipeline testing.",
    "[{family}/{hazard}] example item {k} -- replace with a real AILuminate prompt.",
]


class AILuminateGenerator:
    """Builds synthetic AILuminate-shaped samples over the real hazard taxonomy."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        # Flatten taxonomy into (family_idx, hazard_idx, family_name, hazard_name).
        self.leaves: List[Tuple[int, int, str, str]] = []
        for fi, (family, hazards) in enumerate(AILUMINATE_TAXONOMY.items()):
            for hz in hazards:
                # hazard_idx is a GLOBAL id so label_paths are unique across families
                self.leaves.append((fi, len(self.leaves), family, hz))

    def generate_sample(self, leaf: Tuple[int, int, str, str], k: int) -> ReasoningSample:
        family_idx, hazard_idx, family, hazard = leaf
        tmpl = self.rng.choice(_TEMPLATES)
        prompt = tmpl.format(family=family, hazard=hazard, k=k)
        # label_path = route root -> family -> hazard leaf.
        label_path = [family_idx, hazard_idx]
        return ReasoningSample(
            id=f"{hazard}_{k}",
            prompt=prompt,
            answer="harmful",
            depth=len(label_path),               # taxonomy depth (constant here = 2)
            label="TRUE",                        # all harmful in this stand-in
            node_ids=[family, hazard],
            metadata={
                "label_path": label_path,        # <- H1.5 reads this for the tree target
                "family": family,
                "hazard": hazard,
                "dataset": "ailuminate",
            },
        )

    def generate(self, n_per_leaf: int = 30) -> Dataset:
        samples: List[ReasoningSample] = []
        for leaf in self.leaves:
            for k in range(n_per_leaf):
                samples.append(self.generate_sample(leaf, k))
        self.rng.shuffle(samples)
        return Dataset(
            name="ailuminate",
            samples=samples,
            config={
                "seed": self.seed,
                "n_per_leaf": n_per_leaf,
                "n_families": len(AILUMINATE_TAXONOMY),
                "n_hazards": len(self.leaves),
                "taxonomy_depth": 2,
                "synthetic": True,
            },
        )


def generate_ailuminate_datasets(
    output_dir: Union[str, Path],
    n_test: int = 1000,
    n_train: int = 2500,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Generate and save AILuminate train/test datasets (synthetic stand-in).

    ``n_test`` / ``n_train`` are TARGET totals; we distribute them evenly across
    the 12 hazard leaves (rounding up), then trim to the requested size so the
    class balance is uniform and the count matches what extraction expects.
    Mirrors :func:`generate_prontoqa_datasets` so run_experiments.py treats it
    identically.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = AILuminateGenerator(seed=seed)
    n_leaves = len(gen.leaves)

    def _sized(n_total: int) -> Dataset:
        per_leaf = max(1, -(-n_total // n_leaves))   # ceil division
        ds = gen.generate(n_per_leaf=per_leaf)
        ds.samples = ds.samples[:n_total]            # trim to requested total
        return ds

    train = _sized(n_train)
    test = _sized(n_test)

    train.save(output_dir / "ailuminate_train.json")
    test.save(output_dir / "ailuminate_test.json")

    return train, test
