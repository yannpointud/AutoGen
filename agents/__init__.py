"""
Package des agents de la plateforme multi-agents.
Phase A : Import des agents v2 avec prompts dynamiques.
"""

# Import de la classe de base
from .base_agent import BaseAgent

# Import des agents v2 (Phase A)
from .analyst import Analyst
from .developer import Developer

# Import du superviseur
from .supervisor import Supervisor


__all__ = [
    'BaseAgent',
    'Supervisor',
    'Analyst',
    'Developer'
]

# Version du package
__version__ = '2.0.0'  # Phase A
