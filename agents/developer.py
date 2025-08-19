"""
Agent Developer utilisant une architecture orientée outils.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from agents.base_agent import BaseAgent, Tool, ToolResult
from core.llm_connector import LLMFactory
from config import default_config
from tools.developer_tools import (
    tool_implement_code,
    tool_create_tests,
    tool_create_project_file
)


class Developer(BaseAgent):
    """
    Agent Developer qui utilise des outils pour implémenter le code et les tests.
    """
    
    def __init__(self, project_name: str, supervisor: BaseAgent, rag_engine: Optional[Any] = None):
        """
        Initialise l'agent Developer avec ses outils spécifiques.
        """
        developer_config = default_config['agents']['developer']
        
        # Utiliser un modèle spécialisé pour le code
        model = developer_config.get('model', default_config['llm'].get('model_preferences', {}).get('code_generation', 'codestral-latest'))
        
        llm_config = {
            'model': model,
            'temperature': 0.3,
        }
        
        super().__init__(
            name="Developer",
            role=developer_config['role'],
            personality=developer_config['personality'],
            llm_config=llm_config,
            project_name=project_name,
            supervisor=supervisor,
            rag_engine=rag_engine
        )
        
        # Chemins
        self.src_path = Path("projects") / project_name / "src"
        self.tests_path = self.src_path / "tests"
        
        for path in [self.src_path, self.tests_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Enregistrer les outils spécifiques
        self._register_developer_tools()
        
        self.logger.info(f"Developer initialisé avec {len(self.tools)} outils")
    
    def _register_developer_tools(self) -> None:
        """Enregistre les outils spécifiques au développeur."""
        
        # implement_code
        self.register_tool(
            Tool(
                "implement_code",
                "À utiliser pour TOUT fichier de code source (.py, .js, etc.). Les fichiers seront automatiquement placés dans le dossier 'src/'.",
                {
                    "filename": "Nom du fichier (ex: 'main.py', 'utils.py')",
                    "description": "Description de ce que doit faire le code",
                    "language": "Langage de programmation",
                    "code": "Code source à implémenter"
                }
            ),
            lambda params: tool_implement_code(self, params)
        )
        
        # create_tests
        self.register_tool(
            Tool(
                "create_tests",
                "À utiliser pour TOUT fichier de test. Les fichiers seront automatiquement placés dans le dossier 'src/tests/'.",
                {
                    "filename": "Nom du fichier de test (ex: 'test_main.py')",
                    "target_file": "Fichier à tester",
                    "test_framework": "Framework de test (pytest/jest/unittest)",
                    "code": "Code des tests"
                }
            ),
            lambda params: tool_create_tests(self, params)
        )
        
        # create_project_file
        self.register_tool(
            Tool(
                "create_project_file",
                "À utiliser UNIQUEMENT pour les fichiers de configuration ou de documentation qui seront placés automatiquement à la racine du projet (ex: README.md, package.json, .gitignore). NE PAS UTILISER pour le code source ou les tests.",
                {
                    "filename": "Nom du fichier avec chemin relatif depuis la racine du projet",
                    "content": "Contenu du fichier"
                }
            ),
            lambda params: tool_create_project_file(self, params)
        )


    
    def communicate(self, message: str, recipient: Optional[BaseAgent] = None) -> str:
        """
        Communication technique orientée développement.
        """
        self.update_state(status='communicating')
        
        response_prompt = f"""Tu es {self.name}, {self.role}.

Question: {message}

Fournis une réponse technique claire.
Inclus des exemples de code si pertinent.
Propose des solutions concrètes.
"""
        
        try:
            response = self.generate_with_context(
                prompt=response_prompt,
                temperature=0.5
            )
        except Exception as e:
            self.logger.error(f"Erreur lors de la communication: {str(e)}")
            response = f"Je rencontre une erreur technique: {str(e)}. Je vais analyser cette question et vous proposer une solution."
        
        self.log_interaction('communicate', {
            'message': message,
            'response': response,
            'recipient': str(recipient)
        })
        
        return response