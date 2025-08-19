"""
Agent Analyste utilisant une architecture orientée outils.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from agents.base_agent import BaseAgent, Tool, ToolResult
from core.llm_connector import LLMFactory
from config import default_config
from tools.analyst_tools import (
    tool_create_document,
    tool_generate_architecture_diagrams,
    tool_generate_configuration_files
)


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
            lambda params: tool_create_document(self, params)
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
            lambda params: tool_generate_architecture_diagrams(self, params)
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
            lambda params: tool_generate_configuration_files(self, params)
        )
    
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