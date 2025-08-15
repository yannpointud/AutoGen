"""
Agent Analyste utilisant une architecture orientée outils.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from agents.base_agent import BaseAgent, Tool, ToolResult
from core.llm_connector import LLMFactory
from config import default_config


class Analyst(BaseAgent):
    """
    Agent Analyste qui utilise des outils pour créer des documents et configurations.
    """
    
    def __init__(self, project_name: str, supervisor: BaseAgent, rag_engine: Optional[Any] = None):
        """
        Initialise l'agent Analyste avec ses outils spécifiques.
        """
        analyst_config = default_config['agents']['analyst']
        llm_config = {
            'model': analyst_config.get('model', default_config['llm']['default_model']),
            'temperature': 0.7,
        }
        
        super().__init__(
            name="Analyst",
            role=analyst_config['role'],
            personality=analyst_config['personality'],
            llm_config=llm_config,
            project_name=project_name,
            supervisor=supervisor,
            rag_engine=rag_engine
        )
        
        # Chemins
        self.docs_path = Path("projects") / project_name / "docs"
        self.config_path = Path("projects") / project_name
        self.docs_path.mkdir(parents=True, exist_ok=True)
        
        # Enregistrer les outils spécifiques
        self._register_analyst_tools()
        
        self.logger.info(f"Analyste initialisé avec {len(self.tools)} outils")
    
    def _register_analyst_tools(self) -> None:
        """Enregistre les outils spécifiques à l'analyste."""
        
        # create_document
        self.register_tool(
            Tool(
                "create_document",
                "Crée un document markdown pour les humains (specs, guides, etc.)",
                {
                    "filename": "Nom du fichier (sans extension)",
                    "content": "Contenu du document en markdown"
                }
            ),
            self._tool_create_document
        )
        
        # generate_architecture_diagrams
        self.register_tool(
            Tool(
                "generate_architecture_diagrams",
                "Génère des diagrammes d'architecture en Mermaid",
                {
                    "diagram_type": "Type de diagramme (flowchart/sequence/erd/class)",
                    "content": "Code Mermaid du diagramme"
                }
            ),
            self._tool_generate_architecture_diagrams
        )
        
        # generate_configuration_files
        self.register_tool(
            Tool(
                "generate_configuration_files",
                "Génère des fichiers de configuration pour le projet",
                {
                    "config_type": "Type de config (editorconfig/eslint/prettier/etc)",
                    "language": "Langage principal du projet",
                    "content": "Contenu du fichier de configuration"
                }
            ),
            self._tool_generate_configuration_files
        )
    
    def _tool_create_document(self, parameters: Dict[str, Any]) -> ToolResult:
        """Crée un document markdown."""
        try:
            filename = parameters.get('filename', 'document')
            content = parameters.get('content', '')
            
            # Nettoyer le nom de fichier
            filename = filename.replace('.md', '')
            
            # Limiter la taille
            max_length = self.tools_config.get('specific', {}).get('create_document', {}).get('max_length', 10000)
            if len(content) > max_length:
                content = content[:max_length] + "\n\n[Document tronqué]"
            
            # Ajouter l'en-tête
            header = f"""# {filename.replace('_', ' ').title()}

**Généré par**: {self.name}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Projet**: {self.project_name}

---

"""
            full_content = header + content
            
            # Sauvegarder
            file_path = self.docs_path / f"{filename}.md"
            file_path.write_text(full_content, encoding='utf-8')
            
            # Indexer dans le RAG
            if self.rag_engine:
                self.rag_engine.index_document(
                    full_content,
                    {
                        'type': 'project_file',
                        'source': str(file_path.relative_to(Path("projects") / self.project_name)),
                        'agent_name': self.name,
                        'milestone': self.current_milestone_id,
                        'preserve': True
                    }
                )
            
            return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _tool_generate_architecture_diagrams(self, parameters: Dict[str, Any]) -> ToolResult:
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

Ce diagramme représente l'architecture du système {self.project_name}.
"""
            
            # Sauvegarder
            filename = f"architecture_{diagram_type}_diagram"
            file_path = self.docs_path / f"{filename}.md"
            
            # Ajouter l'en-tête
            header = f"""# Architecture - {diagram_type.title()} Diagram

**Généré par**: {self.name}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Type**: {diagram_type}

---

"""
            full_content = header + doc_content
            file_path.write_text(full_content, encoding='utf-8')
            
            return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _tool_generate_configuration_files(self, parameters: Dict[str, Any]) -> ToolResult:
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
                file_path = self.config_path / filename
            else:
                # Fichiers dans le dossier config
                file_path = self.config_path / "config" / filename
                file_path.parent.mkdir(exist_ok=True)
            
            # Si pas de contenu fourni, utiliser un template par défaut
            if not content:
                content = self._get_default_config_content(config_type, language)
            
            # Sauvegarder
            file_path.write_text(content, encoding='utf-8')
            
            return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _get_default_config_content(self, config_type: str, language: str) -> str:
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
    
    def communicate(self, message: str, recipient: Optional[BaseAgent] = None) -> str:
        """
        Communication orientée analyse et conception.
        """
        self.update_state(status='communicating')
        
        # L'analyste privilégie les réponses structurées
        response_prompt = f"""Tu es {self.name}, {self.role}.

Question: {message}

Réponds de manière claire et structurée.
Si c'est une question technique, fournis des recommandations concrètes.
"""
        
        try:
            response = self.generate_with_context(
                prompt=response_prompt,
                temperature=0.6
            )
        
        except Exception as e: 
            self.logger.error(f"Erreur lors de la communication de l'Analyste: {e}", exc_info=True)
            response = f"Je rencontre une erreur technique et ne peux pas répondre pour le moment. L'erreur a été enregistrée."        

        
        self.log_interaction('communicate', {
            'message': message,
            'response': response,
            'recipient': str(recipient)
        })
        
        return response