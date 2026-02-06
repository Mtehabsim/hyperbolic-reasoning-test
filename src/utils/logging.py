"""
Professional logging infrastructure for experiment tracking.

Provides structured JSON logging for reproducible experiments,
with per-sample and per-experiment schemas.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Configure root logger
_LOGGER_NAME = "hyperbolic_probes"


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_prefix: str = "experiment",
) -> logging.Logger:
    """
    Setup logging with file and console handlers.
    
    Args:
        log_dir: Directory for log files (None for console only)
        level: Logging level
        console: Whether to log to console
        file_prefix: Prefix for log filename
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{file_prefix}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to: {log_file}")
    
    return logger


def get_logger() -> logging.Logger:
    """Get the experiment logger instance."""
    return logging.getLogger(_LOGGER_NAME)


class ExperimentLogger:
    """
    Structured experiment logger for JSON results.
    
    Logs per-sample and per-experiment results in a reproducible format.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize experiment logger.
        
        Args:
            output_dir: Directory for result files
            experiment_name: Name of the experiment
            config: Experiment configuration dict
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.config = config or {}
        self.start_time = datetime.now()
        
        # Results storage
        self.samples: list = []
        self.metrics: Dict[str, Any] = {}
        
        # Create result file path
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.result_file = self.output_dir / f"{experiment_name}_{timestamp}.json"
        
        self._logger = get_logger()
        self._logger.info(f"ExperimentLogger initialized: {experiment_name}")
    
    def log_sample(self, sample_data: Dict[str, Any]) -> None:
        """
        Log a single sample result.
        
        Args:
            sample_data: Dictionary with sample-level results
        """
        # Add timestamp
        sample_data["logged_at"] = datetime.now().isoformat()
        self.samples.append(sample_data)
    
    def log_metric(self, name: str, value: Any, **metadata) -> None:
        """
        Log an experiment-level metric.
        
        Args:
            name: Metric name
            value: Metric value
            **metadata: Additional metadata
        """
        self.metrics[name] = {
            "value": value,
            "logged_at": datetime.now().isoformat(),
            **metadata,
        }
        self._logger.info(f"Metric | {name}: {value}")
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value)
    
    def save(self) -> Path:
        """
        Save all results to JSON file.
        
        Returns:
            Path to the saved result file
        """
        end_time = datetime.now()
        
        result = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - self.start_time).total_seconds(),
            "n_samples": len(self.samples),
            "metrics": self.metrics,
            "samples": self.samples,
        }
        
        with open(self.result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        
        self._logger.info(f"Results saved to: {self.result_file}")
        return self.result_file
    
    @classmethod
    def load(cls, result_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Load results from a JSON file.
        
        Args:
            result_file: Path to the result file
            
        Returns:
            Dictionary with experiment results
        """
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
