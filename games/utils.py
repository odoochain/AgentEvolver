# -*- coding: utf-8 -*-
"""Common utility functions shared by all games."""
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file with Hydra inheritance support.
    
    Uses Hydra's compose API to load configuration with defaults inheritance.
    The config file should use Hydra's `defaults` list to specify base configs.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config_dir = config_path.parent.resolve()
    config_name = config_path.stem  # filename without extension
    
    # Initialize Hydra config directory and compose the config
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    
    # Convert OmegaConf DictConfig to regular Python dict
    return OmegaConf.to_container(cfg, resolve=True)


__all__ = ["load_config"]

