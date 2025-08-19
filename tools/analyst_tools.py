"""
Outils spécifiques à l'agent Analyst.
"""

from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from agents.base_agent import ToolResult


def tool_create_document(agent, parameters: Dict[str, Any]) -> ToolResult:
    """Crée un document markdown."""
    try:
        filename = parameters.get('filename', 'document')
        content = parameters.get('content', '')
        
        # Nettoyer le nom de fichier
        filename = filename.replace('.md', '')
        
        # Limiter la taille
        max_length = agent.tools_config.get('specific', {}).get('create_document', {}).get('max_length', 10000)
        if len(content) > max_length:
            content = content[:max_length] + "\n\n[Document tronqué]"
        
        # Ajouter l'en-tête
        header = f"""# {filename.replace('_', ' ').title()}

**Généré par**: {agent.name}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Projet**: {agent.project_name}

---

"""
        full_content = header + content
        
        # Sauvegarder
        file_path = agent.docs_path / f"{filename}.md"
        file_path.write_text(full_content, encoding='utf-8')
        
        # Indexer dans le RAG
        if agent.rag_engine:
            agent.rag_engine.index_document(
                full_content,
                {
                    'type': 'project_file',
                    'source': str(file_path.relative_to(Path("projects") / agent.project_name)),
                    'agent_name': agent.name,
                    'milestone': agent.current_milestone_id,
                    'preserve': True
                }
            )
        
        return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_generate_architecture_diagrams(agent, parameters: Dict[str, Any]) -> ToolResult:
    """Génère des diagrammes d'architecture."""
    try:
        diagram_type = parameters.get('diagram_type', 'flowchart')
        content = parameters.get('content', '')
        
        # Valider le type
        valid_types = ['flowchart', 'sequence', 'erd', 'class', 'state', 'gantt']
        if diagram_type not in valid_types:
            diagram_type = 'flowchart'
        
        # Créer le document avec les diagrammes
        doc_content = f"""# Diagrammes d'Architecture

## {diagram_type.title()} Diagram

```mermaid
{content}
```

## Légende

Ce diagramme représente l'architecture du système {agent.project_name}.
"""
        
        # Sauvegarder
        filename = f"architecture_{diagram_type}_diagram"
        file_path = agent.docs_path / f"{filename}.md"
        
        # Ajouter l'en-tête
        header = f"""# Architecture - {diagram_type.title()} Diagram

**Généré par**: {agent.name}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Type**: {diagram_type}

---

"""
        full_content = header + doc_content
        file_path.write_text(full_content, encoding='utf-8')
        
        return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_generate_configuration_files(agent, parameters: Dict[str, Any]) -> ToolResult:
    """Génère des fichiers de configuration."""
    try:
        config_type = parameters.get('config_type', '').lower()
        language = parameters.get('language', 'python').lower()
        content = parameters.get('content', '')
        
        # Mapper le type de config au nom de fichier
        config_files = {
            'editorconfig': '.editorconfig',
            'gitignore': '.gitignore',
            'eslint': '.eslintrc.json',
            'prettier': '.prettierrc',
            'flake8': '.flake8',
            'pytest': 'pytest.ini',
            'jest': 'jest.config.js',
            'tsconfig': 'tsconfig.json',
            'pyproject': 'pyproject.toml',
            'package': 'package.json',
            'requirements': 'requirements.txt',
            'dockerfile': 'Dockerfile',
            'dockercompose': 'docker-compose.yml',
            'makefile': 'Makefile',
            'precommit': '.pre-commit-config.yaml'
        }
        
        # Déterminer le nom du fichier
        filename = config_files.get(config_type)
        if not filename:
            # Si non reconnu, utiliser le type comme nom
            filename = f"{config_type}.config"
        
        # Déterminer le chemin
        if filename.startswith('.') or filename in ['requirements.txt', 'setup.py', 'pyproject.toml', 
                                                    'package.json', 'Dockerfile', 'docker-compose.yml', 
                                                    'Makefile']:
            # Fichiers à la racine du projet
            file_path = agent.config_path / filename
        else:
            # Fichiers dans le dossier config
            file_path = agent.config_path / "config" / filename
            file_path.parent.mkdir(exist_ok=True)
        
        # Si pas de contenu fourni, utiliser un template par défaut
        if not content:
            content = _get_default_config_content(config_type, language)
        
        # Sauvegarder
        file_path.write_text(content, encoding='utf-8')
        
        return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def _get_default_config_content(config_type: str, language: str) -> str:
    """Retourne un contenu par défaut pour les configs communes."""
    templates = {
        'editorconfig': """root = true

[*]
indent_style = space
indent_size = 4
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

[*.{js,jsx,ts,tsx,json,yml,yaml}]
indent_size = 2

[*.md]
trim_trailing_whitespace = false
""",
        'gitignore': """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project
*.log
.coverage
htmlcov/
dist/
build/
*.egg-info/
""",
        'flake8': """[flake8]
max-line-length = 88
extend-ignore = E203, E266, E501, W503
max-complexity = 10
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist,venv
""",
        'prettier': """{
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false,
  "semi": true,
  "singleQuote": true,
  "trailingComma": "es5",
  "bracketSpacing": true,
  "arrowParens": "avoid"
}"""
    }
    
    return templates.get(config_type, f"# Configuration {config_type} pour {language}")