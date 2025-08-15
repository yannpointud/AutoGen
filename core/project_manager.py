"""
Gestionnaire de projets responsable de la création, du chargement
et de la gestion de la structure des répertoires pour chaque projet.
Assure une organisation cohérente des artefacts sur le disque.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from utils.logger import setup_logger


class ProjectManager:
    """
    Gère la création et la structure des projets.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialise le gestionnaire de projets.
        
        Args:
            base_path: Chemin de base pour les projets (par défaut: ./projects)
        """
        self.base_path = base_path or Path.cwd() / "projects"
        self.logger = setup_logger(self.__class__.__name__)
        
        # Créer le dossier de base s'il n'existe pas
        self.base_path.mkdir(exist_ok=True)
    
    def create_project_structure(self, project_name: str) -> Optional[Path]:
        """
        Crée la structure complète d'un nouveau projet.
        
        Args:
            project_name: Nom du projet
            
        Returns:
            Path: Chemin du projet créé, ou None si échec
        """
        project_path = self.base_path / project_name
        
        try:
            # Vérifier si le projet existe déjà
            if project_path.exists():
                self.logger.warning(f"Le projet '{project_name}' existe déjà")
                # Demander confirmation pour écraser
                response = input(f"Le projet '{project_name}' existe déjà. Écraser ? (o/N): ")
                if response.lower() != 'o':
                    return None
                shutil.rmtree(project_path)
            
            # Créer la structure de dossiers
            directories = [
                "checkpoints",
                "config", 
                "docs",
                "logs",
                "src",
                "src/tests",
                "data",
                "data/rag"
            ]
            
            for directory in directories:
                (project_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Créer les fichiers de base
            self._create_readme(project_path, project_name)
            self._create_project_config(project_path, project_name)
            self._create_gitignore(project_path)
            
            self.logger.info(f"Structure du projet '{project_name}' créée avec succès")
            return project_path
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du projet: {str(e)}")
            return None
    
    def _create_readme(self, project_path: Path, project_name: str) -> None:
        """
        Crée le fichier README.md du projet.
        
        Args:
            project_path: Chemin du projet
            project_name: Nom du projet
        """
        readme_content = f"""# {project_name}

## Description
Projet généré par la plateforme Multi-Agents IA.

## Structure
```
{project_name}/
├── checkpoints/     # États sauvegardés aux jalons
├── config/          # Configuration du projet et des agents
├── docs/            # Documentation générée
├── logs/            # Journaux des interactions
├── src/             # Code source
│   └── tests/       # Tests unitaires
├── data/            # Données du projet
│   └── rag/         # Index RAG
└── README.md        # Ce fichier
```

## Statut
- **Créé le**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Phase**: Initialisation

## Agents
Les agents suivants seront utilisés dans ce projet:
- Superviseur: Coordination générale
- Analyste: Analyse des besoins
- Architecte: Conception technique
- Lead Dev: Qualité du code
- Développeur: Implémentation
- Testeur: Tests et validation

## Logs
Les logs détaillés sont disponibles dans le dossier `logs/`.
"""
        
        readme_path = project_path / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')
    
    def _create_project_config(self, project_path: Path, project_name: str) -> None:
        """
        Crée le fichier de configuration du projet.
        
        Args:
            project_path: Chemin du projet
            project_name: Nom du projet
        """
        config = {
            "project": {
                "name": project_name,
                "created_at": datetime.now().isoformat(),
                "version": "0.1.0",
                "status": "initialized"
            },
            "agents": {
                "supervisor": {
                    "enabled": True,
                    "personality": "calme, stratégique, orienté résultat"
                },
                "analyst": {
                    "enabled": True,
                    "personality": "curieux, synthétique"
                },
                "architect": {
                    "enabled": True,
                    "personality": "pragmatique, rigoureux"
                },
                "lead_dev": {
                    "enabled": True,
                    "personality": "perfectionniste, pédagogue, attentif"
                },
                "developer": {
                    "enabled": True,
                    "personality": "professionnel et rigoureux"
                },
                "tester": {
                    "enabled": True,
                    "personality": "méthodique, pointilleux"
                }
            },
            "llm": {
                "default_model": "gpt-3.5-turbo",
                "temperature": 0.7,
            },
            "rag": {
                "enabled": True,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }
        
        config_path = project_path / "config" / "project_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _create_gitignore(self, project_path: Path) -> None:
        """
        Crée le fichier .gitignore du projet.
        
        Args:
            project_path: Chemin du projet
        """
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# Conda
.conda/
*.conda
environment.yml

# Logs
logs/
*.log

# Checkpoints
checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
data/rag/*.index
data/rag/*.pkl
"""
        
        gitignore_path = project_path / ".gitignore"
        gitignore_path.write_text(gitignore_content, encoding='utf-8')
    
    def load_project(self, project_name: str) -> Optional[Dict[str, Any]]:
        """
        Charge la configuration d'un projet existant.
        
        Args:
            project_name: Nom du projet
            
        Returns:
            Dict: Configuration du projet, ou None si non trouvé
        """
        project_path = self.base_path / project_name
        config_path = project_path / "config" / "project_config.json"
        
        if not config_path.exists():
            self.logger.error(f"Configuration du projet '{project_name}' non trouvée")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            return None
    
    def list_projects(self) -> list[str]:
        """
        Liste tous les projets existants.
        
        Returns:
            list: Liste des noms de projets
        """
        projects = []
        for item in self.base_path.iterdir():
            if item.is_dir() and (item / "config" / "project_config.json").exists():
                projects.append(item.name)
        return sorted(projects)
