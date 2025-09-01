"""
Module de gestion du contexte projet global.
Permet aux services utilitaires d'accéder au nom du projet actuel
pour diriger leurs logs LLM vers le bon dossier.
"""

from typing import Optional

# Variable globale simple pour projet actuel
_CURRENT_PROJECT: Optional[str] = None


def set_current_project(project_name: str) -> None:
    """
    Définit le projet actuellement actif.
    
    Args:
        project_name: Nom du projet à activer
    """
    global _CURRENT_PROJECT
    _CURRENT_PROJECT = project_name


def get_current_project() -> Optional[str]:
    """
    Récupère le nom du projet actuellement actif.
    
    Returns:
        Nom du projet actuel ou None si aucun projet actif
    """
    return _CURRENT_PROJECT


def clear_current_project() -> None:
    """
    Remet à zéro le contexte projet (pour nettoyage/tests).
    """
    global _CURRENT_PROJECT
    _CURRENT_PROJECT = None