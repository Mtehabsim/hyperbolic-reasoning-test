"""
Weights & Biases integration for experiment tracking.

Optional module - gracefully degrades if wandb is not installed.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import wandb, gracefully handle if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class WandBLogger:
    """
    Optional W&B logger for experiment tracking.
    
    Usage:
        logger = WandBLogger(project="hyperbolic-probes", run_name="deepseek_h1")
        
        # In training loop
        logger.log({"loss": 0.5, "spearman_rho": 0.85}, step=epoch)
        
        # Log sample-level predictions
        logger.log_samples(sample_ids, predictions, targets)
        
        # Finish run
        logger.finish()
    """
    
    def __init__(
        self,
        project: str = "hyperbolic-reasoning-probes",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            run_name: Name for this run (auto-generated if None)
            config: Configuration dict to log
            enabled: Whether to enable logging (can be disabled for fast runs)
            tags: Tags for organizing runs
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        
        if self.enabled:
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=tags or [],
                reinit=True,
            )
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        if self.enabled and self.run:
            wandb.log(metrics, step=step)
    
    def log_samples(
        self,
        sample_ids: List[int],
        predictions: List[float],
        targets: List[float],
        labels: Optional[List[str]] = None,
    ) -> None:
        """
        Log sample-level predictions for analysis.
        
        Creates a W&B Table for detailed inspection.
        """
        if not self.enabled or not self.run:
            return
        
        data = []
        for i, (sid, pred, tgt) in enumerate(zip(sample_ids, predictions, targets)):
            row = {
                "sample_id": sid,
                "prediction": pred,
                "target": tgt,
                "error": abs(pred - tgt),
            }
            if labels:
                row["label"] = labels[i] if i < len(labels) else None
            data.append(row)
        
        table = wandb.Table(
            columns=list(data[0].keys()),
            data=[list(row.values()) for row in data]
        )
        wandb.log({"sample_predictions": table})
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log summary metrics at end of run."""
        if self.enabled and self.run:
            for key, value in summary.items():
                wandb.run.summary[key] = value
    
    def log_artifact(self, path: Path, name: str, artifact_type: str = "model") -> None:
        """Log a file or directory as an artifact."""
        if not self.enabled or not self.run:
            return
        
        artifact = wandb.Artifact(name, type=artifact_type)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        wandb.log_artifact(artifact)
    
    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled and self.run:
            wandb.finish()
            self.run = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.finish()


def is_wandb_available() -> bool:
    """Check if W&B is available."""
    return WANDB_AVAILABLE


def create_run_name(model_name: str, experiment: str, layer: Optional[int] = None) -> str:
    """
    Create a descriptive run name.
    
    Args:
        model_name: e.g., "deepseek_7b"
        experiment: e.g., "h1", "h2", "dim_ablation"
        layer: Optional layer number
        
    Returns:
        Run name like "deepseek_7b_h1_L23"
    """
    parts = [model_name, experiment]
    if layer is not None:
        parts.append(f"L{layer}")
    return "_".join(parts)
