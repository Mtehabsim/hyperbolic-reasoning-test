"""
Configuration utilities using Hydra/OmegaConf.

Provides config loading, path resolution, and access helpers.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/utils to project root
    return Path(__file__).parent.parent.parent


def get_config_dir() -> Path:
    """Get the config directory path."""
    return get_project_root() / "config"


def get_output_dir() -> Path:
    """Get the outputs directory path."""
    return get_project_root() / "outputs"


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: config/config.yaml)
        overrides: Dictionary of config overrides
        
    Returns:
        OmegaConf DictConfig object
    """
    if config_path is None:
        config_path = get_config_dir() / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Apply overrides
    if overrides:
        override_conf = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_conf)
    
    return config


def save_config(config: DictConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        OmegaConf.save(config, f)


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf config to plain dictionary."""
    return OmegaConf.to_container(config, resolve=True)


def resolve_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to base or project root.
    
    Args:
        path: Path string or Path object
        base: Base directory (default: project root)
        
    Returns:
        Resolved absolute Path
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    if base is None:
        base = get_project_root()
    
    return (base / path).resolve()
