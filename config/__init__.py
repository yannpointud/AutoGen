"""
Module config - Gestion de la configuration de la plateforme.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
                    (par défaut: config/default_config.yaml)
    
    Returns:
        Dict: Configuration chargée
    """
    if config_path is None:
        # Utiliser la configuration par défaut
        config_path = Path(__file__).parent / "default_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Remplacer les variables d'environnement
    _replace_env_vars(config)
    
    return config


def _replace_env_vars(config: Any) -> None:
    """
    Remplace récursivement les variables d'environnement dans la config.
    
    Args:
        config: Configuration (dict, list ou str)
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, "")
            else:
                _replace_env_vars(value)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            if isinstance(item, str) and item.startswith("${") and item.endswith("}"):
                env_var = item[2:-1]
                config[i] = os.getenv(env_var, "")
            else:
                _replace_env_vars(item)


# Charger la configuration par défaut au démarrage
default_config = load_config()

__all__ = ['load_config', 'default_config']
