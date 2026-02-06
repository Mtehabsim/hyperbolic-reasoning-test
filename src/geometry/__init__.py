# Geometry module
from .euclidean import (
    pairwise_l2_distance,
    pairwise_cosine_distance,
    pairwise_squared_distance,
    pca_projection,
)
from .hyperbolic import (
    maximum_distance_rescaling,
    PoincareBall,
    LorentzModel,
    poincare_pairwise_distance,
    lorentz_pairwise_distance,
)
from .metrics import (
    compute_spearman,
    compute_distortion,
    compute_map_at_k,
    compute_stress,
    compute_all_metrics,
)

# Convenience aliases
poincare_distance = poincare_pairwise_distance
lorentz_distance = lorentz_pairwise_distance

__all__ = [
    # Euclidean
    "pairwise_l2_distance",
    "pairwise_cosine_distance",
    "pairwise_squared_distance",
    "pca_projection",
    # Hyperbolic
    "maximum_distance_rescaling",
    "PoincareBall",
    "LorentzModel",
    "poincare_pairwise_distance",
    "lorentz_pairwise_distance",
    "poincare_distance",  # alias
    "lorentz_distance",  # alias
    # Metrics
    "compute_spearman",
    "compute_distortion",
    "compute_map_at_k",
    "compute_stress",
    "compute_all_metrics",
]
