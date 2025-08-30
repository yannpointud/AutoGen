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
from core.lightweight_llm_service import get_lightweight_llm_service
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
        self.conversation_memory = deque()  # Pas de limite, compression gère la taille
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
        
        # CYCLE COGNITIF HYBRIDE - Service léger pour phase d'alignement
        self.lightweight_service = get_lightweight_llm_service(self.project_name)
        
        self.logger.info(f"Agent {name} initialisé")
    
    def register_tool(self, tool: Tool, implementation: Callable) -> None:
        """Enregistre un nouvel outil."""
        self.tools[tool.name] = implementation
        self.tool_definitions[tool.name] = tool
        self.logger.debug(f"Outil '{tool.name}' enregistré")
    
    def _register_common_tools(self) -> None:
        """Enregistre les outils communs à tous les agents."""
        from tools.base_tools import (
            tool_search_context,
            tool_send_message_to_agent,
            tool_share_discovery,
            tool_report_to_supervisor
        )
        
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
            lambda params: tool_search_context(self, params)
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
            lambda params: tool_send_message_to_agent(self, params)
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
            lambda params: tool_share_discovery(self, params)
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
            lambda params: tool_report_to_supervisor(self, params)
        )
    
    def _parse_tool_calls(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse de manière robuste les appels d'outils depuis la réponse du LLM avec json5."""
        tool_calls = []

        # Protection contre les réponses vides ou trop longues
        if not llm_response or len(llm_response) > 50000:
            self.logger.warning(f"Réponse LLM vide ou trop longue ({len(llm_response)} chars), pas de parsing d'outils")
            return tool_calls

        # 1. Capturer tous les blocs de code (```json ou ``` génériques contenant JSON)
        patterns = [
            r'```json\s*(.*?)\s*```',  # Blocs explicitement marqués JSON
            r'```\s*(\[.*?\])\s*```',  # Blocs génériques contenant des arrays JSON
            r'```\s*(\{.*?\})\s*```'   # Blocs génériques contenant des objets JSON
        ]
        
        matches = []
        try:
            for pattern in patterns:
                pattern_matches = re.findall(pattern, llm_response, re.DOTALL)
                matches.extend(pattern_matches)
                if pattern_matches:
                    self.logger.debug(f"Trouvé {len(pattern_matches)} match(es) avec pattern {pattern}")
        except Exception as e:
            self.logger.error(f"Erreur regex lors du parsing des outils: {str(e)}")
            return tool_calls

        for match in matches:
            # Nettoyage basique seulement
            match_clean = match.strip()
            if not match_clean:
                self.logger.debug("Bloc JSON vide ignoré")
                continue

            # Parsing direct sans pré-validation - plus fiable
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
        DÉLÉGUÉ vers le parser centralisé pour éviter la duplication de code.
        """
        from core.json_parser import get_json_parser
        
        parser = get_json_parser(f"{self.project_name}.{self.name}")
        return parser.parse_tool_array(json_content)

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
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
    
    def _strategy_progressive_parse(self, json_content: str) -> List[Dict[str, Any]]:
        """
        STRATÉGIE 3: Parsing progressif pour code long.
        Parse chaque outil individuellement pour éviter les échecs sur gros JSON.
        """
        import re
        tools = []
        
        # Pattern pour capturer l'objet outil COMPLET (tous les champs)
        # Utilise un compteur de niveau pour les accolades imbriquées
        tool_pattern = r'\{\s*"tool"\s*:\s*"[^"]+"\s*,(?:[^{}]|\{[^{}]*\})*\}'
        
        # Chercher tous les objets outils complets
        matches = re.finditer(tool_pattern, json_content, re.DOTALL)
        
        for match in matches:
            try:
                # Parser l'objet JSON complet au lieu de le reconstruire
                tool_json = match.group(0)
                
                try:
                    # Utiliser le parser JSON centralisé robuste
                    from core.json_parser import get_json_parser
                    parser = get_json_parser(f"{self.project_name}.{self.name}")
                    tool_obj = parser.parse_universal(tool_json, return_type='dict')
                    
                    # Vérifier que c'est un objet outil valide
                    if isinstance(tool_obj, dict) and "tool" in tool_obj:
                        tools.append(tool_obj)
                        self.logger.debug(f"Outil parsé progressivement: {tool_obj.get('tool', 'unknown')}")
                except Exception as parse_error:
                    self.logger.debug(f"Échec parsing outil individuel: {str(parse_error)}")
                    continue
                
            except Exception as e:
                self.logger.debug(f"Échec parsing outil individuel: {str(e)}")
                continue
        
        # Si ça n'a pas marché, essayer un pattern plus spécialisé pour gros code avec balancing
        if not tools:
            # Utiliser un parser avec équilibrage des accolades pour gros code
            def find_balanced_objects(text, start_pos=0):
                """Trouve des objets JSON équilibrés dans le texte."""
                objects = []
                pos = start_pos
                
                while pos < len(text):
                    # Chercher le début d'un objet outil
                    tool_start = text.find('{"tool":', pos)
                    if tool_start == -1:
                        break
                    
                    # Compter les accolades pour trouver la fin
                    brace_count = 0
                    end_pos = tool_start
                    
                    for i in range(tool_start, len(text)):
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    if brace_count == 0:  # Objet complet trouvé
                        try:
                            obj_json = text[tool_start:end_pos]
                            import json5
                            obj = json5.loads(obj_json)
                            if isinstance(obj, dict) and "tool" in obj:
                                objects.append(obj)
                                self.logger.debug(f"Objet équilibré parsé: {obj.get('tool', 'unknown')}")
                        except Exception as e:
                            self.logger.debug(f"Échec parsing objet équilibré: {str(e)}")
                    
                    pos = tool_start + 1
                
                return objects
            
            tools.extend(find_balanced_objects(json_content))
        
        return tools
    
    def _strategy_extract_partial(self, json_content: str) -> List[Dict[str, Any]]:
        """Stratégie 4: Extraire les objets JSON complets même dans un document partiel."""
        tools = []
        
        # Utiliser une approche plus robuste avec équilibrage des accolades
        def extract_complete_objects(text):
            """Extrait tous les objets JSON complets qui commencent par 'tool'."""
            objects = []
            pos = 0
            
            while pos < len(text):
                # Chercher le début d'un objet potentiel
                start_patterns = [
                    text.find('{"tool":', pos),
                    text.find('{ "tool":', pos),
                    text.find('{\n  "tool":', pos),
                    text.find('{\n    "tool":', pos)
                ]
                start_positions = [p for p in start_patterns if p != -1]
                
                if not start_positions:
                    break
                
                start_pos = min(start_positions)
                
                # Équilibrer les accolades pour trouver la fin
                brace_count = 0
                end_pos = start_pos
                in_string = False
                escape_next = False
                
                for i in range(start_pos, len(text)):
                    char = text[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                
                if brace_count == 0:  # Objet complet trouvé
                    try:
                        obj_json = text[start_pos:end_pos]
                        import json5
                        obj = json5.loads(obj_json)
                        if isinstance(obj, dict) and "tool" in obj:
                            objects.append(obj)
                            self.logger.debug(f"Objet partiel extrait: {obj.get('tool', 'unknown')}")
                    except Exception as e:
                        self.logger.debug(f"Échec parsing objet partiel: {str(e)}")
                
                pos = start_pos + 1
            
            return objects
        
        tools.extend(extract_complete_objects(json_content))
        return tools
    
    def _strategy_documentation_rescue(self, json_content: str) -> List[Dict[str, Any]]:
        """
        STRATÉGIE 5: Extraction spécialisée pour outils de documentation.
        Essaie d'abord de parser l'objet complet, puis fait du rescue si nécessaire.
        """
        tools = []
        
        # D'abord essayer de trouver des objets JSON complets pour la documentation
        doc_tool_patterns = [
            r'\{\s*"tool"\s*:\s*"create_project_file"[^}]*\}',
            r'\{\s*"tool"\s*:\s*"create_document"[^}]*\}',
            r'\{\s*"tool"\s*:\s*"generate_architecture_diagrams"[^}]*\}',
            r'\{\s*"tool"\s*:\s*"generate_configuration_files"[^}]*\}'
        ]
        
        for pattern in doc_tool_patterns:
            matches = re.finditer(pattern, json_content, re.DOTALL)
            for match in matches:
                try:
                    # Essayer de parser l'objet complet d'abord
                    obj_json = match.group(0)
                    
                    # Utiliser l'équilibrage des accolades pour capturer l'objet complet
                    start_pos = match.start()
                    brace_count = 0
                    end_pos = start_pos
                    
                    for i in range(start_pos, len(json_content)):
                        if json_content[i] == '{':
                            brace_count += 1
                        elif json_content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    if brace_count == 0:
                        obj_json = json_content[start_pos:end_pos]
                        
                        try:
                            import json5
                            tool_obj = json5.loads(obj_json)
                            if isinstance(tool_obj, dict) and "tool" in tool_obj:
                                tools.append(tool_obj)
                                self.logger.debug(f"Outil documentation parsé: {tool_obj.get('tool', 'unknown')}")
                                continue
                        except:
                            pass  # Passe au rescue
                    
                    # Si le parsing complet échoue, faire du rescue manuel
                    tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', obj_json)
                    filename_match = re.search(r'"filename"\s*:\s*"([^"]*)"', obj_json)
                    if tool_match and filename_match:
                        tool_data = {
                            "tool": tool_match.group(1),
                            "parameters": {"filename": filename_match.group(1)}
                        }
                        
                        # Extraire le contenu si possible
                        content = self._extract_documentation_content(obj_json, filename_match.group(1))
                        if content:
                            tool_data["parameters"]["content"] = content
                        
                        tools.append(tool_data)
                        self.logger.debug(f"Documentation rescuée: {filename_match.group(1)}")
                
                except Exception as e:
                    self.logger.debug(f"Échec rescue documentation: {str(e)}")
                    continue
        
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
        """Stratégie 6: Extraction regex basique - essaie de récupérer le maximum de champs."""
        tools = []
        
        # Chercher tous les outils avec leurs contextes
        tool_context_pattern = r'"tool"\s*:\s*"([^"]+)"[^}]*'
        tool_matches = re.finditer(tool_context_pattern, json_content, re.DOTALL)
        
        for match in tool_matches[:3]:  # Limiter à 3 pour éviter la redondance
            tool_name = match.group(1)
            context = match.group(0)
            
            tool_data = {"tool": tool_name, "parameters": {}}
            
            # Extraire les paramètres de base
            param_patterns = [
                (r'"filename"\s*:\s*"([^"]*)"', "filename"),
                (r'"content"\s*:\s*"([^"]*)"', "content"),
                (r'"language"\s*:\s*"([^"]*)"', "language"),
                (r'"description"\s*:\s*"([^"]*)"', "description"),
                (r'"config_type"\s*:\s*"([^"]*)"', "config_type"),
                (r'"diagram_type"\s*:\s*"([^"]*)"', "diagram_type"),
            ]
            
            for pattern, param_name in param_patterns:
                param_match = re.search(pattern, context)
                if param_match:
                    tool_data["parameters"][param_name] = param_match.group(1)
            
            tools.append(tool_data)
            self.logger.info(f"Outil fallback récupéré: {tool_name}")
        
        return tools



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
        CYCLE COGNITIF HYBRIDE - Réflexion en deux phases d'alignement + raisonnement.
        
        Phase 1 (Lightweight LLM): Extraction rapide des contraintes critiques du Project Charter
        Phase 2 (Main LLM): Raisonnement complexe guidé par les contraintes
        """
        self.update_state(status='thinking')
        
        if 'milestone_id' in task:
            self.current_milestone_id = task['milestone_id']
        
        self.state['current_task'] = task
        self.state['current_task_id'] = task.get('id', str(uuid.uuid4()))
        
        # Réinitialiser le compteur d'échanges
        self.reset_exchange_counter()
        
        # Mesurer les performances du cycle
        cycle_start = time.time()
        
        try:
            # ========== PHASE 1: ALIGNEMENT  ==========
            self.logger.info("🔄 Alignement agent Projet")
            alignment_start = time.time()
            
            # Récupérer le Project Charter depuis le RAG
            project_charter = self._get_project_charter_from_file()
            
            # Utiliser le Project Charter complet au lieu d'un résumé
            if project_charter:
                critical_constraints = project_charter
                self.logger.debug(f"Project Charter complet transmis ({len(critical_constraints)} chars): {critical_constraints[:100]}...")
            else:
                critical_constraints = f"ATTENTION: Aucun Project Charter trouvé pour {self.project_name}. Procéder avec prudence."
                self.logger.warning("Project Charter non trouvé, contraintes par défaut appliquées")
            
            alignment_duration = time.time() - alignment_start
            #self.logger.info(f"✅ Phase d'Alignement terminée en {alignment_duration:.2f}s")
            
            # ========== PHASE 2: RAISONNEMENT (Main LLM) ==========
            self.logger.info("🧠 Think Agent -> ")
            reasoning_start = time.time()
            
            # Construire le prompt pour le LLM principal avec contraintes injectées
            
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

            # Architecture de prompt à deux niveaux - Charter séparé de l'historique
            clean_prompt = f"""🎯 TA MISSION TACTIQUE:
{task.get('description', '')}

📦 LIVRABLES ATTENDUS: 
{', '.join(task.get('deliverables', []))}
{code_quality_reminder}

ANALYSE cette tâche en gardant STRICTEMENT en tête les contraintes du projet:
1. ALIGNEMENT: Cette tâche respecte-t-elle les contraintes critiques identifiées ?
2. PERTINENCE: Cette tâche est-elle encore nécessaire ? (utilise search_context pour vérifier l'existant)
3. Ta compréhension de la tâche dans le contexte du projet
4. Ton plan d'approche pour respecter les contraintes ET accomplir la tâche
5. Quels outils tu vas utiliser et pourquoi
6. Si tu as des doutes sur la pertinence ou l'alignement, considère utiliser send_message_to_agent

Réponds en texte libre, PAS en JSON. Sois concis mais précis.
"""
            
            # Générer l'analyse avec Charter injecté temporairement
            analysis = self.generate_with_context_enriched(
                clean_prompt=clean_prompt,
                strategic_context=critical_constraints,
                temperature=self.llm_config.get('temperature', 0.7)
            )
            

            self.logger.info(f"🏃 -> Act Agent")
            # Phase ACT - faire confiance à la mémoire conversationnelle
            action_prompt = f"""Basé sur l'analyse que tu viens de fournir, traduis ton plan en un JSON d'appels d'outils.

Utilise les outils disponibles dans ton prompt système.

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
    "tool": "search_context",
    "parameters": {{
      "query": "recherche de contexte pertinent",
      "top_k": 5
    }}
  }},
  {{
    "tool": "create_document",
    "parameters": {{
      "filename": "specifications",
      "content": "## Spécifications\\n\\nContenu du document"
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
            actions_response = self.generate_with_context_enriched(
                clean_prompt=action_prompt,
                strategic_context=critical_constraints,  # Project Charter injecté
                temperature=0.3,  # Plus bas pour JSON précis
            )
            
            # Parser les appels d'outils depuis la réponse JSON
            tool_calls = self._parse_tool_calls(actions_response)
            
            reasoning_duration = time.time() - reasoning_start
            cycle_total = time.time() - cycle_start
            
            #self.logger.info(f"✅ Phase de Raisonnement terminée en {reasoning_duration:.2f}s")
            #self.logger.info(f"🎯 CYCLE COGNITIF HYBRIDE complet en {cycle_total:.2f}s (Alignement: {alignment_duration:.2f}s, Raisonnement: {reasoning_duration:.2f}s)")
            
            plan = {
                'task_id': self.state['current_task_id'],
                'analysis': analysis,
                'planned_tools': tool_calls,
                'deliverables': task.get('deliverables', []),
                'milestone_id': self.current_milestone_id,
                'agent_name': self.name,
                'timestamp': datetime.now().isoformat(),
                # Métriques du cycle cognitif
                'cognitive_cycle_metrics': {
                    'total_duration': cycle_total,
                    'alignment_duration': alignment_duration,
                    'reasoning_duration': reasoning_duration,
                    'project_charter_found': project_charter is not None,
                    'constraints_extracted': len(critical_constraints)
                },
                'critical_constraints': critical_constraints
            }
            
            self.log_interaction('think', plan)
            return plan
            
        except Exception as e:
            self.logger.error(f"💥 Erreur dans le CYCLE COGNITIF HYBRIDE: {str(e)}")
            return {
                'task_id': self.state['current_task_id'],
                'error': str(e),
                'milestone_id': self.current_milestone_id,
                'cognitive_cycle_metrics': {
                    'total_duration': time.time() - cycle_start,
                    'error': str(e)
                }
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
            
            # Après la boucle : évaluer le résultat global
            if not result['artifacts'] and all(t['status'] == 'error' for t in result['tools_executed']):
                # Reporter au superviseur en utilisant l'outil
                self.tools['report_to_supervisor']({
                    'report_type': 'issue',
                    'content': {
                        'task_id': plan.get('task_id'),
                        'reason': 'Tous les outils planifiés ont échoué ou le plan était invalide.'
                    }
                })
                result['status'] = 'failed'
            else:
                result['status'] = 'completed'
                
                # PHASE 2: Génération du rapport structuré pour les tâches réussies
                if result['status'] == 'completed':
                    structured_report = self._generate_structured_report(plan, result)
                    result['structured_report'] = structured_report
                    self.logger.info(f"Rapport structuré généré: {structured_report.get('self_assessment', 'unknown')}")
                    
                    # AJOUT : Envoi systématique du rapport au supervisor
                    try:
                        # Déterminer le type de rapport basé sur l'auto-évaluation
                        assessment = structured_report.get('self_assessment', 'unknown')
                        if assessment == 'compliant':
                            report_type = 'completion'
                        elif assessment == 'partial':
                            report_type = 'progress'
                        else:  # 'failed' ou 'unknown'
                            report_type = 'issue'
                        
                        self.tools['report_to_supervisor']({
                            'report_type': report_type,
                            'content': structured_report
                        })
                        self.logger.debug(f"Rapport structuré envoyé au supervisor: {report_type} ({assessment})")
                    except Exception as e:
                        self.logger.warning(f"Échec envoi rapport au supervisor: {e}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution: {str(e)}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        self.log_interaction('act', result)
        return result
    
    def _generate_structured_report(self, plan: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        NOUVEAU SYSTÈME: Génère un rapport structuré basé sur l'évaluation sémantique intelligente.
        Remplace complètement l'ancienne logique de validation par correspondance exacte.
        """
        try:
            # Collecter les faits bruts de la mission
            artifacts_created = result.get('artifacts', [])
            tools_executed = result.get('tools_executed', [])
            failed_tools = [t for t in tools_executed if t.get('status') == 'error']
            issues = [t.get('result', {}).get('error', 'Erreur inconnue') for t in failed_tools]
            
            # L'analyse de la phase 'think' est la meilleure source pour l'objectif de la mission
            task_objective = plan.get('analysis', 'Objectif non défini.')
            
            # Appeler le service léger pour obtenir une évaluation sémantique
            evaluation_response = self.lightweight_service.self_evaluate_mission(
                objective=task_objective,
                artifacts=artifacts_created,
                issues=issues
            )
            
            # Construire le rapport final en se basant sur cette évaluation intelligente
            structured_report = {
                'artifacts_created': artifacts_created,
                'decisions_made': f"Mission évaluée par l'agent lui-même via compréhension sémantique.",
                'issues_encountered': issues,
                'self_assessment': evaluation_response.get('assessment', 'unknown'),
                'confidence_level': evaluation_response.get('confidence', 0.5),
                'assessment_reason': evaluation_response.get('reason', 'Raison non fournie.'),
                'milestone_id': plan.get('milestone_id', 'unknown'),
                'agent_name': self.name,
                'completion_timestamp': datetime.now().isoformat()
            }
            
            return structured_report
            
        except Exception as e:
            self.logger.error(f"Erreur majeure dans la génération du rapport structuré: {e}")
            return {
                'self_assessment': 'failed',
                'assessment_reason': f'Erreur critique durant la génération du rapport: {e}',
                'artifacts_created': result.get('artifacts', []),
                'issues_encountered': [f"Erreur système: {e}"],
                'confidence_level': 0.1,
                'decisions_made': "Rapport d'erreur généré automatiquement",
                'milestone_id': plan.get('milestone_id', 'unknown'),
                'agent_name': self.name,
                'completion_timestamp': datetime.now().isoformat()
            }
    
    def _format_tools_for_prompt(self) -> str:
        """Formate la liste des outils pour le prompt."""
        lines = []
        for tool in self.tool_definitions.values():
            lines.append(f"\n{tool.name}: {tool.description}")
            lines.append("  Paramètres:")
            for param, desc in tool.parameters.items():
                lines.append(f"    - {param}: {desc}")
            
            # Ajouter des conseils d'usage pour les outils de création
            if tool.name in ['create_document', 'generate_architecture_diagrams', 'implement_code', 'create_tests', 'generate_configuration_files', 'create_project_file']:
                lines.append("  💡 Conseil: Créé des fichiers de qualité qui répondent aux besoins identifiés dans l'analyse de la tâche.")
                lines.append("  L'évaluation de la réussite se base sur la cohérence entre les livrables demandés et les artefacts produits.")
        
        return "\n".join(lines)
    
    def answer_colleague(self, asking_agent: str, question: str) -> str:
        """Répond à la question d'un collègue."""
        self.logger.info(f"Question reçue de {asking_agent}: {question[:100]}...")
        
        response_prompt = f"""Un collègue agent te pose une question.

{asking_agent} demande: {question}

Réponds de manière concise et utile basée sur ton expertise.
Maximum 3-4 phrases.
"""
        
        try:
            response = self.generate_with_context_enriched(
                clean_prompt=response_prompt,
                strategic_context=self._get_project_charter_from_file(),
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
    
    def get_agent_context(self) -> Dict[str, Any]:
        """
        Retourne le contexte de l'agent pour enrichir les logs LLM.
        
        Returns:
            Dict contenant les informations contextuelles de l'agent
        """
        return {
            'agent_name': self.name,
            'agent_role': self.role,
            'project_name': self.project_name,
            'agent_type': self.__class__.__name__,
            'current_milestone': getattr(self, 'current_milestone_id', None),
            'task_id': self.state.get('current_task_id', None)
        }
    
    def generate_with_context(self, prompt: str, **kwargs) -> str:
        """Génère une réponse en utilisant l'historique conversationnel et le contexte RAG."""
        messages = self.get_conversation_context()
        
        # NOUVEAU : Créer prompt système avec identité agent + guidelines + outils
        guidelines_text = '\n'.join(['- ' + g for g in self.guidelines]) if self.guidelines else ""
        tools_description = self._format_tools_for_prompt()
        agent_system_prompt = f"""Tu es {self.name}, {self.role}.
Personnalité: {self.personality}

Guidelines comportementales:
{guidelines_text}

🛠️ OUTILS DISPONIBLES:
{tools_description}"""
        system_message = {
            "role": "system",
            "content": agent_system_prompt
        }
        messages.insert(0, system_message)
        
        # Enrichir automatiquement avec contexte RAG intelligent dans le prompt user
        rag_context = self._get_smart_rag_context(prompt)
        if rag_context:
            enriched_prompt = f"{prompt}\n\n--- CONTEXTE DYNAMIQUE PERTINENT (RAG) ---\n{rag_context}\n--- FIN CONTEXTE DYNAMIQUE ---"
        else:
            enriched_prompt = prompt
        
        messages.append({"role": "user", "content": enriched_prompt})
        
        # COMPRESSION INTELLIGENTE : Vérifier si le prompt total dépasse le seuil
        from config import default_config
        compression_threshold = default_config['general']['conversation_compression_threshold']
        total_prompt_size = self._calculate_final_prompt_size(messages, None)  # rag_context déjà inclus dans messages
        
        self.logger.debug(f"Prompt size: {total_prompt_size} chars, threshold: {compression_threshold}, history: {len(self.conversation_history)} msgs, memory: {len(self.conversation_memory)} msgs")
        
        if total_prompt_size > compression_threshold:
            memory_size = default_config['general']['conversation_memory_size']
            
            # Isoler la mémoire à court terme (N derniers messages à protéger de la compression)
            short_term_memory = list(self.conversation_memory)[-memory_size:] if len(self.conversation_memory) > memory_size else list(self.conversation_memory)
            
            # Messages à compresser = conversation_history moins les N derniers (protégés)
            if len(self.conversation_history) > memory_size:
                history_to_compress = self.conversation_history[:-memory_size]
                
                if history_to_compress:
                    # Concaténer les anciens messages en texte
                    old_text = "\n\n".join([
                        f"[{msg.get('timestamp', '')}] {msg.get('role', '')}: {msg.get('content', '')}"
                        for msg in history_to_compress
                    ])
                    
                    try:
                        # Compression via lightweight_llm_service (méthode dédiée conversation)
                        compressed_summary = self.lightweight_service.summarize_conversation(old_text)
                        
                        # Reconstruction : créer des messages compressés 
                        compressed_messages = []
                        # Ajouter le prompt système agent + guidelines
                        guidelines_text = '\n'.join(['- ' + g for g in self.guidelines]) if self.guidelines else ""
                        agent_system_prompt = f"""Tu es {self.name}, {self.role}.
Personnalité: {self.personality}

Guidelines comportementales:
{guidelines_text}"""
                        system_message = {
                            "role": "system",
                            "content": agent_system_prompt
                        }
                        compressed_messages.append(system_message)
                        
                        # Créer un message résumé qui remplace les anciens messages
                        if compressed_summary.strip():
                            summary_message = {
                                "role": "assistant",
                                "content": f"[Résumé des échanges précédents : {compressed_summary}]"
                            }
                            compressed_messages.append(summary_message)
                        
                        # Ajouter la mémoire court terme intacte
                        for msg in short_term_memory:
                            compressed_messages.append({"role": msg["role"], "content": msg["content"]})
                        
                        # Ajouter le nouveau prompt avec RAG si disponible
                        if rag_context:
                            final_prompt = f"{prompt}\n\n--- CONTEXTE DYNAMIQUE PERTINENT (RAG) ---\n{rag_context}\n--- FIN CONTEXTE DYNAMIQUE ---"
                        else:
                            final_prompt = prompt
                        compressed_messages.append({"role": "user", "content": final_prompt})
                        
                        # Utiliser les messages compressés
                        messages = compressed_messages
                        
                        # Calculer la taille après compression (ne pas re-passer rag_context car déjà inclus dans messages)
                        final_prompt_size = self._calculate_final_prompt_size(messages, None)
                        
                        self.logger.info(f"⚡ Compression appliquée : {total_prompt_size} chars -> {final_prompt_size} chars ({final_prompt_size - total_prompt_size:+d})")
                    
                    except Exception as e:
                        self.logger.warning(f"Échec de la compression, prompt non modifié: {str(e)}")
        
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
        
        # Correction robuste: gérer le format de réponse structuré du modèle magistral
        if isinstance(response, list):
            # Extraire le contenu "text" de la réponse structurée
            text_content = None
            for item in response:
                if isinstance(item, str) and item.startswith('text="'):
                    # Format: text="contenu réel..."
                    text_content = item[6:]  # Enlever 'text="'
                    if text_content.endswith('"'):
                        text_content = text_content[:-1]  # Enlever '"' final
                    break
                elif isinstance(item, str) and 'text=' in item:
                    # Autre format possible
                    text_start = item.find('text="') + 6
                    text_end = item.rfind('"')
                    if text_start > 5 and text_end > text_start:
                        text_content = item[text_start:text_end]
                        break
            
            if text_content:
                response = text_content
                self.logger.info(f"Réponse structurée extraite: {len(response)} caractères")
            else:
                # Fallback: joindre tous les éléments
                response = '\n'.join(str(item) for item in response)
                self.logger.warning(f"Réponse liste non structurée, jointure: {len(response)} caractères")
        elif hasattr(response, 'text'):
            # Format objet avec attribut text
            response = response.text
            self.logger.debug("Extraction text depuis objet réponse")
        elif not isinstance(response, str):
            # Forcer la conversion en chaîne pour tous les autres types
            response = str(response)
            self.logger.warning(f"Type inattendu {type(response)}, conversion string")
        
        self.add_message_to_memory("assistant", response)
        
        return response
    
    def _parse_json_from_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse intelligent de JSON depuis réponses LLM (gère markdown et formats divers).
        DÉLÉGUÉ vers le parser centralisé pour éviter la duplication de code.
        
        Args:
            response: Réponse LLM potentiellement contenant du JSON
            
        Returns:
            Dict contenant le JSON parsé, ou dict vide si échec
        """
        from core.json_parser import get_json_parser
        
        parser = get_json_parser(f"{self.project_name}.{self.name}")
        return parser.parse_llm_response(response)
    
    def generate_with_context_enriched(self, clean_prompt: str, strategic_context: str = None, **kwargs) -> str:
        """
        Génère une réponse en enrichissant temporairement avec contexte stratégique 
        SANS polluer l'historique conversationnel.
        
        Args:
            clean_prompt: Prompt "propre" sans Project Charter (pour l'historique)
            strategic_context: Project Charter à injecter temporairement
            **kwargs: Arguments pour generate_with_context
            
        Returns:
            str: Réponse du LLM
        """
        # 1. Construire le prompt final enrichi (temporaire)
        if strategic_context:
            full_prompt = f"""{clean_prompt}

--- CONTEXTE STRATÉGIQUE DE RÉFÉRENCE (PROJECT CHARTER) ---
{strategic_context}
--- FIN DU CONTEXTE STRATÉGIQUE ---"""
        else:
            full_prompt = clean_prompt
        
        # 2. Obtenir l'historique existant
        messages = self.get_conversation_context()
        
        # 3. Ajouter le prompt système avec guidelines + outils
        guidelines_text = '\n'.join(['- ' + g for g in self.guidelines]) if self.guidelines else ""
        tools_description = self._format_tools_for_prompt()
        agent_system_prompt = f"""Tu es {self.name}, {self.role}.
Personnalité: {self.personality}

Guidelines comportementales:
{guidelines_text}

🛠️ OUTILS DISPONIBLES:
{tools_description}"""
        system_message = {
            "role": "system",
            "content": agent_system_prompt
        }
        messages.insert(0, system_message)
        
        # 4. Enrichir avec contexte RAG et ajouter le prompt final
        rag_context = self._get_smart_rag_context(full_prompt)
        if rag_context:
            enriched_prompt = f"{full_prompt}\n\n--- CONTEXTE DYNAMIQUE PERTINENT (RAG) ---\n{rag_context}\n--- FIN CONTEXTE DYNAMIQUE ---"
        else:
            enriched_prompt = full_prompt
            
        messages.append({"role": "user", "content": enriched_prompt})
        
        # 5. Gérer la compression si nécessaire (même logique que generate_with_context)
        from config import default_config
        compression_threshold = default_config['general']['conversation_compression_threshold']
        total_prompt_size = self._calculate_final_prompt_size(messages, None)
        
        if total_prompt_size > compression_threshold:
            # Même logique de compression que dans generate_with_context
            memory_size = default_config['general']['conversation_memory_size']
            short_term_memory = list(self.conversation_memory)[-memory_size:] if len(self.conversation_memory) > memory_size else list(self.conversation_memory)
            
            if len(self.conversation_history) > memory_size:
                history_to_compress = self.conversation_history[:-memory_size]
                
                if history_to_compress:
                    old_text = "\n\n".join([
                        f"[{msg.get('timestamp', '')}] {msg.get('role', '')}: {msg.get('content', '')}"
                        for msg in history_to_compress
                    ])
                    
                    try:
                        compressed_summary = self.lightweight_service.summarize_conversation(old_text)
                        
                        compressed_messages = []
                        # Système avec guidelines
                        compressed_messages.append(system_message)
                        
                        # Résumé compressé
                        if compressed_summary.strip():
                            summary_message = {
                                "role": "assistant", 
                                "content": f"[Résumé des échanges précédents : {compressed_summary}]"
                            }
                            compressed_messages.append(summary_message)
                        
                        # Mémoire court terme
                        for msg in short_term_memory:
                            compressed_messages.append({"role": msg["role"], "content": msg["content"]})
                        
                        # Nouveau prompt enrichi
                        compressed_messages.append({"role": "user", "content": enriched_prompt})
                        
                        messages = compressed_messages
                        
                        self.logger.info(f"⚡ Compression appliquée : {total_prompt_size} chars -> {self._calculate_final_prompt_size(messages, None)} chars (-{total_prompt_size - self._calculate_final_prompt_size(messages, None)})")
                    except Exception as e:
                        self.logger.warning(f"Échec de la compression, prompt non modifié: {str(e)}")
        
        # 6. Ajouter SEULEMENT le clean_prompt à l'historique (pas le full_prompt)
        self.add_message_to_memory("user", clean_prompt)
        
        # 7. Appel LLM
        llm = LLMFactory.create(model=self.llm_config['model'])
        
        try:
            response = llm.generate_with_messages(
                messages=messages, 
                agent_context=self.get_agent_context(),
                **{k: v for k, v in kwargs.items() if k not in ['prompt']}
            )
        except Exception as e:
            self.logger.error(f"Erreur génération LLM: {str(e)}")
            return f"Erreur lors de la génération: {str(e)}"
        
        if hasattr(response, 'text'):
            # Format objet avec attribut text
            response = response.text
            self.logger.debug("Extraction directe de la réponse")
        elif not isinstance(response, str):
            response = str(response)
            self.logger.warning(f"Type LLM de raisonnement {type(response)}, conversion réponse")
        
        # 8. Ajouter la réponse à l'historique
        self.add_message_to_memory("assistant", response)
        
        return response
    
    def _calculate_final_prompt_size(self, messages: List[Dict[str, str]], rag_context: Optional[str] = None) -> int:
        """
        Calcule la taille totale du prompt final qui sera envoyé au LLM.
        Simule la construction complète incluant tous les composants.
        """
        total_size = 0
        
        # Taille des messages de conversation
        for message in messages:
            total_size += len(str(message.get('content', '')))
            total_size += len(str(message.get('role', '')))
            total_size += 10  # Estimation overhead JSON/format
        
        # Taille du contexte RAG s'il existe
        if rag_context:
            total_size += len(rag_context)
            total_size += 50  # Overhead pour l'injection du contexte
        
        # Estimation de l'overhead du prompt système de l'agent (role, guidelines, etc.)
        total_size += len(self.role) if hasattr(self, 'role') else 0
        total_size += 500  # Estimation conservative pour le prompt système et instructions
        
        return total_size
    
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
        
        try:
            # Configuration depuis default_config.yaml
            max_context_length = auto_context_config.get('max_context_length', 50000)
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
                top_k=self.rag_engine.top_k,
                include_working_memory=True
            )
            
            if not results:
                result = None
                if cache_enabled:
                    self._rag_context_cache[cache_key] = result
                return result
            
            # Formater le contexte avec répartition équitable
            context_parts = []
            seen_sources = set()
            
            # Calcul automatique : répartir l'espace disponible entre les chunks
            chars_per_chunk = max_context_length // self.rag_engine.top_k if results else 0
            
            for result in results:
                source = result.get('source', 'unknown')
                text = result.get('chunk_text', '')
                score = result.get('score', 0)
                from_wm = result.get('from_working_memory', False)
                
                # Éviter les doublons par source
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                
                # Tronquer à la taille calculée automatiquement
                if len(text) > chars_per_chunk:
                    text_summary = text[:chars_per_chunk] + "..."
                else:
                    text_summary = text
                
                # Marquer la provenance
                prefix = "[Mémoire]" if from_wm else "[Docs]"
                part = f"{prefix} {source} (score: {score:.2f}):\n{text_summary}"
                
                context_parts.append(part)
            
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
    
    def _get_project_charter_from_file(self) -> Optional[str]:
        """
        Architecture unifiée: Récupère le Project Charter depuis le fichier uniquement.
        Tous les agents (y compris Supervisor) fonctionnent de la même façon.
        
        Returns:
            str: Contenu du Project Charter
            
        Raises:
            RuntimeError: Si le Project Charter est inaccessible
        """
        try:
            charter_path = Path("projects") / self.project_name / "docs" / "PROJECT_CHARTER.md"
            if charter_path.exists():
                charter = charter_path.read_text(encoding='utf-8')
                if charter and len(charter) > 50:  # Validation minimale
                    self.logger.debug(f"Project Charter inséré depuis le fichier: {charter_path}")
                    return charter
                else:
                    raise ValueError("Project Charter fichier vide ou trop court")
            else:
                raise FileNotFoundError(f"Project Charter non trouvé: {charter_path}")
                
        except Exception as e:
            self.logger.error(f"PROJET COMPROMIS: Impossible de lire le Project Charter: {str(e)}")
            raise RuntimeError(f"PROJET COMPROMIS: Project Charter inaccessible pour {self.project_name}: {str(e)}")
    
    def generate_json_with_context(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Génère une réponse JSON avec parsing robuste centralisé."""
        json_prompt = f"{prompt}\n\nRéponds uniquement avec un JSON valide."
        
        response = self.generate_with_context_enriched(
            clean_prompt=json_prompt,
            strategic_context=self._get_project_charter_from_file(),
            **kwargs
        )
        
        # Utiliser le parser JSON centralisé robuste
        from core.json_parser import get_json_parser
        
        parser = get_json_parser(f"{self.project_name}.{self.name}")
        result = parser.parse_llm_response(response)
        
        if not result:
            # ÉCHEC PARSING - Retourner un marqueur d'échec pour escalade
            error_msg = f"ÉCHEC PARSING JSON - Agent {self.name} - Réponse LLM: {response[:500]}..."
            self.logger.error(error_msg)
            return {
                "parsing_failed": True,
                "raw_response": response,
                "error": f"Parser JSON a échoué pour {self.name}"
            }
        
        return result
    
    def __str__(self) -> str:
        return f"{self.name} ({self.role})"