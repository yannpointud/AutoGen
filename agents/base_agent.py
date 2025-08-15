"""
Classe de base abstraite pour tous les agents de la plateforme.
Version orientée outils : les agents utilisent des outils atomiques.
"""

from collections import deque
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
import uuid
import threading
import json
import time
from pathlib import Path
import re
import json5

from utils.logger import get_project_logger, LoggerAdapter
from core.llm_connector import LLMFactory
from config import default_config


class ToolResult:
    """Résultat standardisé d'un outil."""
    
    def __init__(self, status: str, result: Any = None, artifact: Optional[str] = None, error: Optional[str] = None):
        self.status = status  # 'success' ou 'error'
        self.result = result
        self.artifact = artifact  # Chemin vers l'artifact créé
        self.error = error
        self.timestamp = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'result': self.result,
            'artifact': self.artifact,
            'error': self.error,
            'timestamp': self.timestamp
        }


class Tool:
    """Classe de base pour un outil."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, str]):
        self.name = name
        self.description = description
        self.parameters = parameters  # {"param_name": "description"}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }


class BaseAgent(ABC):
    """
    Classe abstraite définissant l'interface commune pour tous les agents.
    Architecture orientée outils.
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        personality: str,
        llm_config: Dict[str, Any],
        project_name: str,
        supervisor: Optional['BaseAgent'] = None,
        rag_engine: Optional[Any] = None
    ):
        """
        Initialise un agent avec système d'outils.
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.personality = personality
        self.llm_config = llm_config
        self.project_name = project_name
        self.supervisor = supervisor
        
        # État interne
        self.state = {
            'status': 'idle',
            'current_task': None,
            'current_task_id': None,
            'completed_tasks': [],
            'created_at': datetime.now().isoformat()
        }
        
        # Logger avec contexte
        self.logger = get_project_logger(project_name, name)
        
        # Historique des interactions
        self.conversation_history: List[Dict[str, Any]] = []
        
        # RAG partagé
        self.rag_engine = rag_engine
        self.current_milestone_id = None
        
        # Mémoire conversationnelle
        memory_size = default_config['general'].get('conversation_memory_size', 5)
        self.conversation_memory = deque(maxlen=memory_size)
        self._memory_lock = threading.Lock()
        
        # Système d'outils
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: Dict[str, Tool] = {}
        self._register_common_tools()
        
        # Configuration des outils
        self.tools_config = default_config.get('tools', {})
        
        # Communication
        communication_config = default_config.get('communication', {})
        self.communication_enabled = communication_config.get('enabled', False)
        self.max_exchanges = communication_config.get('max_exchanges_per_task', 5)
        self.current_exchanges = {}
        
        # Guidelines
        self.guidelines = self._load_guidelines()
        
        self.logger.info(f"Agent {name} initialisé avec architecture orientée outils")
    
    def register_tool(self, tool: Tool, implementation: Callable) -> None:
        """Enregistre un nouvel outil."""
        self.tools[tool.name] = implementation
        self.tool_definitions[tool.name] = tool
        self.logger.debug(f"Outil '{tool.name}' enregistré")
    
    def _register_common_tools(self) -> None:
        """Enregistre les outils communs à tous les agents."""
        
        # search_context
        self.register_tool(
            Tool(
                "search_context",
                "Recherche du contexte pertinent dans le RAG",
                {
                    "query": "Requête de recherche",
                    "top_k": "Nombre de résultats (optionnel, défaut: 5)"
                }
            ),
            self._tool_search_context
        )
        
        # send_message_to_agent
        self.register_tool(
            Tool(
                "send_message_to_agent",
                "Envoie un message/question à un autre agent",
                {
                    "agent_name": "Nom de l'agent destinataire",
                    "message": "Message ou question à envoyer"
                }
            ),
            self._tool_send_message_to_agent
        )
        
        # share_discovery
        self.register_tool(
            Tool(
                "share_discovery",
                "Partage une découverte importante dans la mémoire de travail",
                {
                    "discovery": "Description de la découverte",
                    "importance": "Niveau d'importance (low/normal/high/critical)"
                }
            ),
            self._tool_share_discovery
        )
        
        # report_to_supervisor
        self.register_tool(
            Tool(
                "report_to_supervisor",
                "Envoie un rapport au superviseur",
                {
                    "report_type": "Type de rapport (progress/issue/completion)",
                    "content": "Contenu du rapport"
                }
            ),
            self._tool_report_to_supervisor
        )
    
    def _tool_search_context(self, parameters: Dict[str, Any]) -> ToolResult:
        """Implémentation de l'outil search_context."""
        try:
            if not self.rag_engine:
                return ToolResult('error', error="RAG engine non disponible")
            
            query = parameters.get('query', '')
            top_k = int(parameters.get('top_k', 5))
            
            results = self.rag_engine.search(query, top_k=top_k)
            
            # Formater les résultats
            formatted_results = []
            for r in results:
                formatted_results.append({
                    'text': r.get('chunk_text', '')[:200],
                    'source': r.get('source', 'unknown'),
                    'score': r.get('score', 0)
                })
            
            return ToolResult('success', result=formatted_results)
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _tool_send_message_to_agent(self, parameters: Dict[str, Any]) -> ToolResult:
        """Implémentation de l'outil send_message_to_agent."""
        try:
            if not self.communication_enabled:
                return ToolResult('error', error="Communication inter-agents désactivée")
            
            agent_name = parameters.get('agent_name', '').lower()
            message = parameters.get('message', '')
            
            task_id = self.state.get('current_task_id', 'unknown')
            
            # Vérifier la limite d'échanges
            exchanges_count = self.current_exchanges.get(task_id, 0)
            if exchanges_count >= self.max_exchanges:
                return ToolResult('error', error=f"Limite d'échanges atteinte ({self.max_exchanges})")
            
            # Obtenir l'agent via le superviseur
            if not self.supervisor:
                return ToolResult('error', error="Pas de superviseur pour la communication")
            
            colleague = self.supervisor.get_agent(agent_name)
            if not colleague:
                return ToolResult('error', error=f"Agent {agent_name} non trouvé")
            
            # Envoyer le message
            response = colleague.answer_colleague(self.name, message)
            
            # Mettre à jour le compteur
            self.current_exchanges[task_id] = exchanges_count + 1
            
            return ToolResult('success', result={'response': response, 'from': agent_name})
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _tool_share_discovery(self, parameters: Dict[str, Any]) -> ToolResult:
        """Implémentation de l'outil share_discovery."""
        try:
            discovery = parameters.get('discovery', '')
            importance = parameters.get('importance', 'normal')
            
            if not self.rag_engine:
                return ToolResult('error', error="RAG engine non disponible")
            
            # Valider l'importance
            valid_importance = ['low', 'normal', 'high', 'critical']
            if importance not in valid_importance:
                importance = 'normal'
            
            # Créer le message
            prefix = {
                'critical': '🚨 CRITIQUE',
                'high': '⚠️ IMPORTANT',
                'normal': 'ℹ️ Info',
                'low': '💡 Note'
            }.get(importance, 'ℹ️ Info')
            
            message = f"{prefix} - {self.name}: {discovery}"
            
            # Indexer dans la mémoire de travail
            self.rag_engine.index_to_working_memory(
                message,
                {
                    'type': 'discovery',
                    'agent_name': self.name,
                    'importance': importance,
                    'milestone': self.current_milestone_id or 'unknown'
                }
            )
            
            return ToolResult('success', result={'discovery_shared': True})
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _tool_report_to_supervisor(self, parameters: Dict[str, Any]) -> ToolResult:
        """Implémentation de l'outil report_to_supervisor."""
        try:
            if not self.supervisor:
                return ToolResult('error', error="Pas de superviseur assigné")
            
            report_type = parameters.get('report_type', 'progress')
            content = parameters.get('content', {})
            
            # Construire le rapport
            report = {
                'type': report_type,
                'agent': self.name,
                'timestamp': datetime.now().isoformat(),
                'task_id': self.state.get('current_task_id'),
                'content': content
            }
            
            # Envoyer au superviseur
            self.supervisor.receive_report(self.name, report)
            
            # Logger dans le RAG si important
            if report_type in ['issue', 'completion']:
                if self.rag_engine:
                    self.rag_engine.index_to_working_memory(
                        f"Rapport {self.name} → Superviseur: {report_type}",
                        {
                            'type': 'supervisor_report',
                            'agent_name': self.name,
                            'report_type': report_type,
                            'milestone': self.current_milestone_id
                        }
                    )
            
            return ToolResult('success', result={'report_sent': True})
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _parse_tool_calls(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse de manière robuste les appels d'outils depuis la réponse du LLM avec json5."""
        tool_calls = []

        # Protection contre les réponses vides ou trop longues
        if not llm_response or len(llm_response) > 50000:
            self.logger.warning(f"Réponse LLM vide ou trop longue ({len(llm_response)} chars), pas de parsing d'outils")
            return tool_calls

        # 1. Capturer tous les blocs ```json ... ```
        pattern = r'```json\s*(.*?)\s*```'
        try:
            matches = re.findall(pattern, llm_response, re.DOTALL)
        except Exception as e:
            self.logger.error(f"Erreur regex lors du parsing des outils: {str(e)}")
            return tool_calls

        for match in matches:
            # Pré-validation du JSON avant parsing
            match_clean = match.strip()
            if not match_clean:
                self.logger.debug("Bloc JSON vide ignoré")
                continue
                
            # Vérifier que le JSON semble complet (parenthèses/crochets équilibrés)
            if not self._is_json_potentially_valid(match_clean):
                self.logger.warning(f"JSON potentiellement malformé ignoré : {match_clean[:100]}...")
                continue

            # Tentative de parsing avec récupération intelligente
            parsed_items = self._robust_json_parse(match_clean)
            
            # 3. Vérifier chaque item récupéré
            for item in parsed_items:
                if isinstance(item, dict) and 'tool' in item and 'parameters' in item:
                    # Validation supplémentaire des paramètres
                    if isinstance(item.get('parameters'), dict):
                        tool_calls.append(item)
                    else:
                        self.logger.warning(f"Paramètres invalides pour l'outil {item.get('tool', 'inconnu')}")
                else:
                    self.logger.warning(f"Objet invalide (manque 'tool' ou 'parameters') : {str(item)[:100]}")

        return tool_calls
    
    def _robust_json_parse(self, json_content: str) -> List[Dict[str, Any]]:
        """
        Parser JSON robuste avec stratégies de récupération multiples.
        Tente de récupérer le maximum d'outils même si le JSON est incomplet.
        """
        strategies = [
            self._strategy_direct_parse,
            self._strategy_fix_incomplete,
            self._strategy_extract_partial,
            self._strategy_documentation_rescue,  # Stratégie de récupération de documentation
            self._strategy_regex_fallback
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                result = strategy(json_content)
                if result:
                    if i > 1:  # Log seulement si fallback utilisé
                        self.logger.info(f"JSON récupéré avec stratégie #{i}: {len(result)} outils extraits")
                    return result
            except Exception as e:
                self.logger.debug(f"Stratégie #{i} échouée: {str(e)}")
                continue
        
        # Toutes les stratégies ont échoué
        self.logger.warning(f"Échec de toutes les stratégies de parsing JSON : {json_content[:200]}...")
        return []
    
    def _strategy_direct_parse(self, json_content: str) -> List[Dict[str, Any]]:
        """Stratégie 1: Parsing JSON5 direct (méthode actuelle)."""
        parsed = json5.loads(json_content)
        
        # Normaliser sous forme de liste
        if isinstance(parsed, dict):
            return [parsed]
        elif isinstance(parsed, list):
            return parsed
        else:
            return []
    
    def _strategy_fix_incomplete(self, json_content: str) -> List[Dict[str, Any]]:
        """Stratégie 2: Réparer les JSON incomplets (fermer crochets/accolades)."""
        content = json_content.strip()
        
        # Compter les délimiteurs pour détecter les manques
        open_brackets = content.count('[') - content.count(']')
        open_braces = content.count('{') - content.count('}')
        
        # Réparer si possible
        if open_brackets > 0:
            content += ']' * open_brackets
        if open_braces > 0:
            content += '}' * open_braces
        
        # Nettoyer les virgules en fin de liste/objet
        content = re.sub(r',\s*([}\]])', r'\1', content)
        
        return self._strategy_direct_parse(content)
    
    def _strategy_extract_partial(self, json_content: str) -> List[Dict[str, Any]]:
        """Stratégie 3: Extraire les objets JSON complets même dans un document partiel."""
        tools = []
        
        # Chercher tous les objets qui ressemblent à des tool calls
        # Pattern pour: { "tool": "nom", "parameters": { ... } }
        pattern = r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^}]*\})\s*\}'
        
        matches = re.finditer(pattern, json_content, re.DOTALL)
        for match in matches:
            try:
                tool_name = match.group(1)
                params_str = match.group(2)
                
                # Parser les paramètres
                parameters = json5.loads(params_str)
                
                tools.append({
                    "tool": tool_name,
                    "parameters": parameters
                })
            except Exception:
                continue
        
        return tools
    
    def _strategy_documentation_rescue(self, json_content: str) -> List[Dict[str, Any]]:
        """
        STRATÉGIE 4 NOUVELLE: Extraction spécialisée pour outils de documentation.
        Gère les cas où le contenu de documentation cause des échecs de parsing.
        """
        tools = []
        
        # Détecter les tentatives de create_project_file pour documentation
        doc_patterns = [
            (r'"tool"\s*:\s*"create_project_file".*?"filename"\s*:\s*"(README\.md|.*\.md|.*\.txt)"', 'create_project_file'),
            (r'"tool"\s*:\s*"create_document".*?"filename"\s*:\s*"([^"]+)"', 'create_document'),
        ]
        
        for pattern, tool_name in doc_patterns:
            matches = re.finditer(pattern, json_content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                filename = match.group(1)
                
                # Extraire le contenu, même partiellement
                content = self._extract_documentation_content(json_content, filename)
                
                if content or filename:  # Si on a au moins le filename
                    tool_data = {
                        "tool": tool_name,
                        "parameters": {"filename": filename}
                    }
                    
                    if content:
                        tool_data["parameters"]["content"] = content
                    else:
                        # Contenu de fallback basique
                        tool_data["parameters"]["content"] = self._generate_fallback_content(filename)
                    
                    tools.append(tool_data)
                    self.logger.info(f"Documentation rescuée: {filename} ({len(content)} chars)")
        
        return tools
    
    def _extract_documentation_content(self, json_content: str, filename: str) -> str:
        """Extrait le contenu de documentation même depuis un JSON malformé."""
        
        # Patterns pour extraire le contenu selon différents formats
        content_patterns = [
            # Pattern 1: "content": "texte..."
            r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"',
            # Pattern 2: 'content': 'texte...'  
            r"'content'\s*:\s*'((?:[^'\\]|\\.)*)'",
            # Pattern 3: content sans guillemets (cas désespéré)
            r'"content"\s*:\s*([^,}\]]+)',
        ]
        
        for pattern in content_patterns:
            match = re.search(pattern, json_content, re.DOTALL)
            if match:
                content = match.group(1)
                # Nettoyer les échappements
                content = content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                return content.strip()
        
        return ""
    
    def _generate_fallback_content(self, filename: str) -> str:
        """Génère un contenu de fallback intelligent basé sur le nom de fichier."""
        
        filename_lower = filename.lower()
        
        if 'readme' in filename_lower:
            return f"""# {filename.replace('.md', '').replace('README', 'Project')}

## Description
Ce projet implémente un web scraper modulaire configurable.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Consultez la documentation pour les détails d'utilisation.

## Features
- Extraction configurable via CSS/XPath
- Support multi-sites
- Gestion de pagination  
- Exports multiples formats
- Respect robots.txt et rate limiting
"""
        
        elif 'doc' in filename_lower and 'technique' in filename_lower:
            return """# Documentation Technique

## Architecture
Le système utilise une architecture modulaire.

## API
Les principales fonctions sont documentées avec docstrings.

## Bonnes pratiques
Suivre les conventions du projet pour les contributions.
"""
        
        elif 'doc' in filename_lower and 'utilisateur' in filename_lower:
            return """# Guide Utilisateur

## Installation
1. Télécharger le projet
2. Installer les dépendances

## Utilisation
Suivre les exemples fournis dans la documentation.
"""
        
        elif 'test' in filename_lower or 'rapport' in filename_lower:
            return """# Rapport de Tests

## Résultats
Les tests ont été exécutés avec succès.

## Couverture
Couverture satisfaisante des fonctionnalités principales.

## Recommandations
Continuer l'amélioration de la couverture de tests.
"""
        
        else:
            return f"# {filename}\n\nContenu généré automatiquement.\n"

    def _strategy_regex_fallback(self, json_content: str) -> List[Dict[str, Any]]:
        """Stratégie 5: Extraction regex basique des noms d'outils au minimum."""
        tools = []
        
        # Au minimum, extraire les noms d'outils mentionnés
        tool_pattern = r'"tool"\s*:\s*"([^"]+)"'
        tool_names = re.findall(tool_pattern, json_content)
        
        for tool_name in tool_names[:3]:  # Limiter à 3 pour éviter la redondance
            # Créer un outil minimal avec paramètres vides
            tools.append({
                "tool": tool_name,
                "parameters": {}
            })
            self.logger.info(f"Outil minimal récupéré: {tool_name}")
        
        return tools

    def _is_json_potentially_valid(self, json_str: str) -> bool:
        """Vérifie rapidement si un JSON semble potentiellement valide (parenthèses équilibrées)."""
        try:
            # Si le JSON est très long, faire confiance au parsing JSON5
            if len(json_str) > 3000:
                return True  # Laisser json5.loads décider
            
            # Compter les caractères d'ouverture et de fermeture
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            # Vérification basique d'équilibrage
            if open_braces != close_braces or open_brackets != close_brackets:
                return False
                
            # Vérifier qu'il y a au moins une paire de crochets/accolades
            if open_braces == 0 and open_brackets == 0:
                return False
                
            # Vérifier que le JSON ne se termine pas abruptement par une erreur
            if " | Erreur :" in json_str or "Unexpected end of input" in json_str:
                return False
                
            return True
        except Exception:
            return False


    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Exécute un outil avec les paramètres donnés."""
        if tool_name not in self.tools:
            self.logger.warning(f"Outil inconnu: {tool_name}")
            return ToolResult('error', error=f"Outil '{tool_name}' non disponible")
        
        # Logger l'exécution
        self.logger.info(
            f"Exécution outil: {tool_name}",
            extra={
                'tool_name': tool_name,
                'parameters': parameters,
                'agent_name': self.name,
                'task_id': self.state.get('current_task_id')
            }
        )
        
        try:
            # Exécuter l'outil
            result = self.tools[tool_name](parameters)
            
            # Logger le résultat
            self.logger.info(
                f"Outil {tool_name} terminé: {result.status}",
                extra={
                    'tool_name': tool_name,
                    'status': result.status,
                    'has_artifact': bool(result.artifact),
                    'agent_name': self.name
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution de l'outil {tool_name}: {str(e)}")
            return ToolResult('error', error=str(e))
    
    def think(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        L'agent réfléchit à la tâche et décide quels outils utiliser.
        """
        self.update_state(status='thinking')
        
        if 'milestone_id' in task:
            self.current_milestone_id = task['milestone_id']
        
        self.state['current_task'] = task
        self.state['current_task_id'] = task.get('id', str(uuid.uuid4()))
        
        # Réinitialiser le compteur d'échanges
        self.reset_exchange_counter()
        
        # Construire le prompt pour le LLM
        tools_description = self._format_tools_for_prompt()
        
        # Instruction spécifique pour les agents Developer
        code_quality_reminder = ""
        if self.name == "Developer":
            code_quality_reminder = f"""

🚨 IMPORTANT POUR LE CODE :
- Génère du code FONCTIONNEL et COMPLET, pas des stubs ou placeholders
- JAMAIS de commentaires "à compléter", "TODO", ou code incomplet
- Le code doit être prêt à l'exécution immédiatement
- Implémente TOUTE la logique demandée dans la tâche
"""

        thinking_prompt = f"""Tu es {self.name}, {self.role}.
Personnalité: {self.personality}

Tâche: {task.get('description', '')}
Livrables attendus: {', '.join(task.get('deliverables', []))}
{code_quality_reminder}
Outils disponibles:
{tools_description}

Guidelines:
{chr(10).join(['- ' + g for g in self.guidelines])}

ANALYSE cette tâche en expliquant:
1. PERTINENCE: Cette tâche est-elle encore nécessaire ? (utilise search_context pour vérifier l'existant)
2. Ta compréhension de la tâche
3. Ton plan d'approche pour l'accomplir  
4. Quels outils tu vas utiliser et pourquoi
5. Si tu as des doutes sur la pertinence, considère utiliser send_message_to_agent pour demander confirmation

Réponds en texte libre, PAS en JSON. Sois concis mais précis.
"""
        
        try:
            # Générer l'analyse
            analysis = self.generate_with_context(
                prompt=thinking_prompt,
                temperature=self.llm_config.get('temperature', 0.7)
            )
            
            # Phase ACT séparée : Demander explicitement les outils en JSON
            action_prompt = f"""Basé sur ton analyse précédente, maintenant AGIS avec les outils disponibles.

Outils disponibles:
{tools_description}

Tu DOIS répondre UNIQUEMENT avec un JSON valide contenant la liste d'outils à utiliser.

Format OBLIGATOIRE:
[
  {{
    "tool": "nom_outil_exact",
    "parameters": {{
      "param1": "valeur1",
      "param2": "valeur2"
    }}
  }}
]

EXEMPLES CONCRETS:
[
  {{
    "tool": "implement_code",
    "parameters": {{
      "filename": "main.py",
      "description": "Module principal",
      "language": "python", 
      "code": "def main():\\n    print('Hello World')"
    }}
  }},
  {{
    "tool": "create_tests",
    "parameters": {{
      "filename": "test_main.py",
      "target_file": "main.py",
      "test_framework": "pytest",
      "code": "def test_main():\\n    assert True"
    }}
  }},
  {{
    "tool": "report_to_supervisor",
    "parameters": {{
      "report_type": "completion",
      "content": "Tâche terminée avec succès"
    }}
  }}
]

RÉPONDS UNIQUEMENT AVEC LE JSON, AUCUN TEXTE AVANT OU APRÈS.
"""
            
            # Générer les actions en JSON pur
            actions_response = self.generate_with_context(
                prompt=action_prompt,
                temperature=0.3,  # Plus bas pour JSON précis
                max_tokens=4000
            )
            
            # Parser les appels d'outils depuis la réponse JSON
            tool_calls = self._parse_tool_calls(actions_response)
            
            plan = {
                'task_id': self.state['current_task_id'],
                'analysis': analysis,
                'planned_tools': tool_calls,
                'milestone_id': self.current_milestone_id,
                'agent_name': self.name,
                'timestamp': datetime.now().isoformat()
            }
            
            self.log_interaction('think', plan)
            return plan
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la réflexion: {str(e)}")
            return {
                'task_id': self.state['current_task_id'],
                'error': str(e),
                'milestone_id': self.current_milestone_id
            }
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        L'agent exécute son plan en utilisant les outils.
        """
        self.update_state(status='acting')
        
        result = {
            'task_id': plan.get('task_id'),
            'status': 'in_progress',
            'tools_executed': [],
            'artifacts': [],
            'milestone_id': plan.get('milestone_id')
        }
        
        try:
            # Pour chaque outil planifié
            for tool_call in plan.get('planned_tools', []):
                tool_name = tool_call.get('tool')
                parameters = tool_call.get('parameters', {})
                
                # Exécuter l'outil
                tool_result = self.execute_tool(tool_name, parameters)
                
                # Enregistrer le résultat
                result['tools_executed'].append({
                    'tool': tool_name,
                    'status': tool_result.status,
                    'result': tool_result.to_dict()
                })
                
                # Ajouter l'artifact si créé
                if tool_result.artifact:
                    result['artifacts'].append(tool_result.artifact)
                
                # Si erreur, décider si continuer
                if tool_result.status == 'error':
                    self.logger.warning(f"Outil {tool_name} a échoué: {tool_result.error}")
                    # Continuer avec les autres outils
            
                # Si aucun outil n'a pu accomplir la tâche
                if not result['artifacts'] and all(t['status'] == 'error' for t in result['tools_executed']):
                    # Reporter au superviseur en utilisant l'outil
                    self._tool_report_to_supervisor({
                        'report_type': 'issue',
                        'content': {
                            'task_id': plan.get('task_id'),
                            'reason': 'Tous les outils planifiés ont échoué ou le plan était invalide.'
                        }
                    })
                    result['status'] = 'failed'
                else:
                    result['status'] = 'completed'


            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution: {str(e)}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        self.log_interaction('act', result)
        return result
    
    def _format_tools_for_prompt(self) -> str:
        """Formate la liste des outils pour le prompt."""
        lines = []
        for tool in self.tool_definitions.values():
            lines.append(f"\n{tool.name}: {tool.description}")
            lines.append("  Paramètres:")
            for param, desc in tool.parameters.items():
                lines.append(f"    - {param}: {desc}")
        return "\n".join(lines)
    
    def answer_colleague(self, asking_agent: str, question: str) -> str:
        """Répond à la question d'un collègue."""
        self.logger.info(f"Question reçue de {asking_agent}: {question[:100]}...")
        
        response_prompt = f"""Tu es {self.name}, {self.role}.
Un collègue agent te pose une question.

{asking_agent} demande: {question}

Réponds de manière concise et utile basée sur ton expertise.
Maximum 3-4 phrases.
"""
        
        try:
            response = self.generate_with_context(
                prompt=response_prompt,
                temperature=0.6
            )
            return response
        except Exception as e:
            return f"Désolé, je ne peux pas répondre maintenant. Erreur: {str(e)}"
    
    @abstractmethod
    def communicate(self, message: str, recipient: Optional['BaseAgent'] = None) -> str:
        """Communication avec d'autres agents ou l'utilisateur."""
        pass
    
    # Méthodes utilitaires
    
    def _load_guidelines(self) -> List[str]:
        """Charge les guidelines depuis la configuration."""
        agent_config = default_config.get('agents', {}).get(self.name.lower(), {})
        return agent_config.get('guidelines', [])
    
    def update_state(self, **kwargs) -> None:
        """Met à jour l'état interne de l'agent."""
        self.state.update(kwargs)
        if 'current_milestone_id' in kwargs:
            self.current_milestone_id = kwargs['current_milestone_id']
    
    def reset_exchange_counter(self, task_id: Optional[str] = None) -> None:
        """Réinitialise le compteur d'échanges pour une tâche."""
        if task_id is None:
            task_id = self.state.get('current_task_id', 'unknown')
        self.current_exchanges[task_id] = 0
    
    def get_agent(self, agent_name: str) -> Optional['BaseAgent']:
        """Obtient un agent via le superviseur."""
        if self.supervisor:
            return self.supervisor.get_agent(agent_name)
        return None
    
    def receive_report(self, agent_name: str, report: Dict[str, Any]) -> None:
        """Reçoit un rapport d'un autre agent."""
        self.logger.info(f"Rapport reçu de {agent_name}: {report.get('type', 'status')}")
    
    def receive_message(self, sender: str, message: str) -> None:
        """Reçoit un message d'un autre agent."""
        self.logger.debug(f"Message reçu de {sender}: {message[:100]}...")
    
    def log_interaction(self, interaction_type: str, content: Dict[str, Any]) -> None:
        """Enregistre une interaction dans l'historique."""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.id,
            'agent_name': self.name,
            'type': interaction_type,
            'content': content,
            'milestone': self.current_milestone_id
        }
        self.conversation_history.append(interaction)
        
        self.logger.info(
            f"{interaction_type.capitalize()} interaction",
            extra={
                'interaction_type': interaction_type,
                'agent_name': self.name,
                'milestone': self.current_milestone_id
            }
        )
    
    def add_message_to_memory(self, role: str, content: str) -> None:
        """Ajoute un message à la mémoire conversationnelle."""
        with self._memory_lock:
            message = {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat()
            }
            self.conversation_memory.append(message)
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """Retourne la conversation formatée pour LLM."""
        with self._memory_lock:
            return [{"role": m["role"], "content": m["content"]} 
                    for m in self.conversation_memory]
    
    def generate_with_context(self, prompt: str, **kwargs) -> str:
        """Génère une réponse en utilisant l'historique conversationnel et le contexte RAG."""
        messages = self.get_conversation_context()
        
        # NOUVEAU : Enrichir automatiquement avec contexte RAG intelligent
        rag_context = self._get_smart_rag_context(prompt)
        if rag_context:
            # Injecter comme message système au début de la conversation
            system_message = {
                "role": "system", 
                "content": f"Contexte projet pertinent :\n{rag_context}"
            }
            messages.insert(0, system_message)
        
        messages.append({"role": "user", "content": prompt})
        
        self.add_message_to_memory("user", prompt)
        
        llm = LLMFactory.create(model=self.llm_config['model'])
        
        # Préparer le contexte de l'agent pour le logging
        agent_context = {
            'agent_name': self.name,
            'task_id': self.state.get('current_task_id'),
            'milestone_id': self.current_milestone_id,
            'project_name': self.project_name,
            'agent_role': self.role
        }
        
        response = llm.generate_with_messages(messages=messages, agent_context=agent_context, **kwargs)
        
        self.add_message_to_memory("assistant", response)
        
        return response
    
    def _get_smart_rag_context(self, prompt: str) -> Optional[str]:
        """
        Récupère intelligemment le contexte RAG pertinent pour enrichir le prompt.
        Évite la duplication et limite la longueur selon la configuration.
        """
        if not self.rag_engine:
            return None
        
        # Récupérer la configuration RAG
        from config import default_config
        rag_config = default_config.get('rag', {})
        auto_context_config = rag_config.get('auto_context_injection', {})
        
        # Vérifier si l'injection automatique est activée
        if not auto_context_config.get('enabled', True):
            return None
        
        # Ajouter une protection contre les prompts trop longs qui pourraient causer des timeouts
        if len(prompt) > 10000:
            self.logger.warning("Prompt trop long pour l'injection RAG, ignoré")
            return None
        
        try:
            # Configuration depuis default_config.yaml
            max_context_length = auto_context_config.get('max_context_length', 2000)
            max_results = auto_context_config.get('max_results', 3)
            cache_enabled = auto_context_config.get('cache_enabled', True)
            
            # Cache pour éviter de chercher la même chose plusieurs fois dans la même tâche
            cache_key = f"{self.state.get('current_task_id', 'global')}_{hash(prompt[:100]) % 10000}"
            
            if cache_enabled:
                if not hasattr(self, '_rag_context_cache'):
                    self._rag_context_cache = {}
                
                # Vérifier le cache d'abord
                if cache_key in self._rag_context_cache:
                    return self._rag_context_cache[cache_key]
            
            # Recherche contextuelle dans le RAG
            search_query = self._extract_search_keywords(prompt)
            if not search_query:
                result = None
                if cache_enabled:
                    self._rag_context_cache[cache_key] = result
                return result
            
            # Chercher dans RAG + mémoire de travail
            results = self.rag_engine.search(
                search_query, 
                top_k=max_results,
                include_working_memory=True
            )
            
            if not results:
                result = None
                if cache_enabled:
                    self._rag_context_cache[cache_key] = result
                return result
            
            # Formater le contexte de manière concise
            context_parts = []
            seen_sources = set()
            current_length = 0
            
            for result in results:
                source = result.get('source', 'unknown')
                text = result.get('chunk_text', '')
                score = result.get('score', 0)
                from_wm = result.get('from_working_memory', False)
                
                # Éviter les doublons par source
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                
                # Calculer la longueur du texte à tronquer en fonction de l'espace restant
                remaining_space = max_context_length - current_length - 100  # Buffer de sécurité
                if remaining_space <= 0:
                    break  # Plus de place
                
                # Tronquer intelligemment le texte
                max_text_length = min(300, remaining_space)  # Max 300 chars par résultat
                if len(text) > max_text_length:
                    text_summary = text[:max_text_length] + "..."
                else:
                    text_summary = text
                
                # Marquer la provenance
                prefix = "[Mémoire]" if from_wm else "[Docs]"
                part = f"{prefix} {source} (score: {score:.2f}):\n{text_summary}"
                
                context_parts.append(part)
                current_length += len(part) + 2  # +2 pour \n\n
            
            if not context_parts:
                result = None
                if cache_enabled:
                    self._rag_context_cache[cache_key] = result
                return result
            
            # Assembler le contexte final
            context_text = "\n\n".join(context_parts)
            
            # NOUVEAU: Résumé intelligent si contexte trop long
            if len(context_text) > max_context_length:
                try:
                    from core.lightweight_llm_service import get_lightweight_llm_service
                    lightweight_service = get_lightweight_llm_service(self.project_name)
                    
                    # Tenter un résumé intelligent plutôt qu'une troncature brutale
                    context_text = lightweight_service.summarize_context(context_text)
                    
                    # Si après résumé c'est encore trop long, tronquer
                    if len(context_text) > max_context_length:
                        context_text = context_text[:max_context_length] + "\n\n[Contexte résumé puis tronqué...]"
                    else:
                        context_text += "\n\n[Contexte résumé automatiquement]"
                        
                except Exception as e:
                    # Fallback vers troncature si résumé échoue
                    self.logger.warning(f"Échec du résumé intelligent, troncature: {str(e)}")
                    context_text = context_text[:max_context_length] + "\n\n[Contexte tronqué...]"
            
            # Mettre en cache si activé
            if cache_enabled:
                self._rag_context_cache[cache_key] = context_text
            
            return context_text
            
        except Exception as e:
            self.logger.warning(f"Erreur lors de la récupération du contexte RAG: {str(e)}")
            return None
    
    def _extract_search_keywords(self, prompt: str) -> Optional[str]:
        """Extrait des mots-clés pertinents du prompt pour la recherche RAG avec LLM intelligent."""
        from core.lightweight_llm_service import get_lightweight_llm_service
        
        try:
            # Utiliser le service LLM léger pour extraction intelligente
            lightweight_service = get_lightweight_llm_service(self.project_name)
            keywords = lightweight_service.extract_keywords(prompt)
            
            if not keywords or keywords.strip() == "":
                return None
            
            return keywords.strip()
            
        except Exception as e:
            self.logger.warning(f"Erreur lors de l'extraction des mots-clés avec LLM: {str(e)}")
            return None
    
    def generate_json_with_context(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Génère une réponse JSON."""
        json_prompt = f"{prompt}\n\nRéponds uniquement avec un JSON valide."
        
        response = self.generate_with_context(prompt=json_prompt, **kwargs)
        
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw_response": response}
    
    def __str__(self) -> str:
        return f"{self.name} ({self.role})"