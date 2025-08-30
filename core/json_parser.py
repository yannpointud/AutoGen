"""
Parser JSON centralisé avec stratégies de récupération multiples.
Centralise toute la logique de parsing JSON pour éliminer la duplication de code.
"""

import json
import re
from typing import List, Dict, Any, Callable, Optional
from utils.logger import get_project_logger


class RobustJSONParser:
    """
    Parser JSON centralisé avec stratégies multiples et contextes adaptés.
    Remplace et unifie toute la logique de parsing JSON dispersée dans le système.
    """
    
    def __init__(self, logger_name: str = "JSONParser"):
        self.logger = get_project_logger("System", logger_name)
    
    def parse_universal(self, content: str, return_type: str = 'auto') -> Any:
        """
        Méthode universelle de parsing JSON avec TOUTES les stratégies robustes.
        
        Args:
            content: Contenu JSON à parser
            return_type: 'list', 'dict', ou 'auto' (détection automatique)
            
        Returns:
            Résultat parsé selon return_type, ou {} / [] si échec
        """
        # Toutes les stratégies robustes dans l'ordre de priorité
        strategies = [
            # JSON5 (gère multilignes, commentaires, trailing commas)
            self._strategy_markdown_json5,
            self._strategy_json5_direct_universal,
            # JSON standard avec markdown
            self._strategy_markdown_json,
            self._strategy_markdown_generic,
            self._strategy_json_direct,
            # Stratégies de réparation
            self._strategy_fix_incomplete_universal,
            self._strategy_progressive_parse_universal,
            # Fallbacks d'extraction
            self._strategy_extract_partial_universal,
            self._strategy_embedded_json,
            self._strategy_documentation_rescue_universal,
            self._strategy_regex_fallback_universal
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                result = strategy(content)
                if result:
                    # Auto-détection du type de retour si nécessaire
                    if return_type == 'auto':
                        if isinstance(result, list):
                            return_type = 'list'
                        elif isinstance(result, dict):
                            return_type = 'dict'
                    
                    # Conversion selon le type demandé
                    if return_type == 'list':
                        if isinstance(result, dict):
                            result = [result]
                        elif not isinstance(result, list):
                            result = []
                    elif return_type == 'dict':
                        if isinstance(result, list) and result:
                            result = result[0] if result else {}
                        elif not isinstance(result, dict):
                            result = {}
                    
                    if i > 1:
                        count = len(result) if isinstance(result, list) else 1
                        self.logger.info(f"JSON récupéré avec stratégie #{i}: {count} éléments")
                    return result
            except Exception as e:
                self.logger.debug(f"Stratégie #{i} échouée: {str(e)}")
                continue
        
        # Toutes les stratégies ont échoué
        self.logger.warning(f"ÉCHEC TOTAL parsing JSON: {content[:100]}...")
        return [] if return_type == 'list' else {}
    
    def parse_tool_array(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse une liste d'outils JSON (ex: réponses d'agents).
        Wrapper vers parse_universal pour compatibilité.
        """
        return self.parse_universal(content, return_type='list')
    
    def parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        Parse une réponse LLM simple (ex: auto-évaluation, configuration).
        Wrapper vers parse_universal pour compatibilité.
        """
        return self.parse_universal(content, return_type='dict')
    
    def parse_config_object(self, content: str) -> Dict[str, Any]:
        """
        Parse un objet de configuration simple.
        Wrapper vers parse_universal pour compatibilité.
        """
        return self.parse_universal(content, return_type='dict')
    
    def _execute_strategies(self, content: str, strategies: List[Callable], return_list: bool = True) -> Any:
        """
        Exécute les stratégies en cascade jusqu'au succès.
        
        Args:
            content: Contenu à parser
            strategies: Liste des stratégies à essayer
            return_list: Si True retourne liste, sinon dict
            
        Returns:
            Résultat du parsing selon return_list
        """
        for i, strategy in enumerate(strategies, 1):
            try:
                result = strategy(content)
                if result:
                    if i > 1:
                        count = len(result) if isinstance(result, list) else 1
                        self.logger.info(f"JSON récupéré avec stratégie #{i}: {count} éléments")
                    return result
            except Exception as e:
                self.logger.debug(f"Stratégie #{i} échouée: {str(e)}")
                continue
        
        # Toutes les stratégies ont échoué
        self.logger.warning(f"Échec de toutes les stratégies de parsing JSON: {content[:100]}...")
        return [] if return_list else {}
    
    # === STRATÉGIES POUR LISTES D'OUTILS ===
    
    def _strategy_json5_direct(self, content: str) -> List[Dict[str, Any]]:
        """Stratégie 1: Parsing JSON5 direct pour listes."""
        try:
            import json5
            parsed = json5.loads(content)
            
            # Normaliser sous forme de liste
            if isinstance(parsed, dict):
                return [parsed]
            elif isinstance(parsed, list):
                return parsed
            else:
                return []
        except ImportError:
            # Fallback si json5 pas disponible
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return [parsed]
            elif isinstance(parsed, list):
                return parsed
            else:
                return []
    
    def _strategy_fix_incomplete(self, content: str) -> List[Dict[str, Any]]:
        """Stratégie 2: Réparer JSON incomplet (accolades manquantes)."""
        # Compter les accolades ouvertes vs fermées
        open_braces = content.count('{')
        close_braces = content.count('}')
        
        if open_braces > close_braces:
            # Ajouter les accolades manquantes
            fixed_content = content + ('}' * (open_braces - close_braces))
            try:
                import json5
                parsed = json5.loads(fixed_content)
                if isinstance(parsed, dict):
                    return [parsed]
                elif isinstance(parsed, list):
                    return parsed
            except:
                pass
        
        return []
    
    def _strategy_progressive_parse(self, content: str) -> List[Dict[str, Any]]:
        """Stratégie 3: Parsing progressif pour contenu long."""
        objects = []
        pos = 0
        
        while pos < len(content):
            # Chercher le début d'un objet potentiel
            start_patterns = [
                content.find('{"tool":', pos),
                content.find('{ "tool":', pos),
                content.find('{\n  "tool":', pos),
                content.find('{\n    "tool":', pos)
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
            
            for i in range(start_pos, len(content)):
                char = content[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
            
            if brace_count == 0:
                obj_json = content[start_pos:end_pos]
                try:
                    import json5
                    obj = json5.loads(obj_json)
                    if isinstance(obj, dict) and "tool" in obj:
                        objects.append(obj)
                except:
                    pass
            
            pos = end_pos if end_pos > start_pos else start_pos + 1
        
        return objects
    
    def _strategy_extract_partial(self, content: str) -> List[Dict[str, Any]]:
        """Stratégie 4: Extraction partielle d'objets."""
        objects = []
        
        # Pattern plus complet qui capture aussi fulfills_deliverable
        tool_pattern = r'\{\s*"tool"\s*:\s*"[^"]+"\s*(?:,\s*"parameters"\s*:\s*\{[^}]*\})?(?:\s*,\s*"fulfills_deliverable"\s*:\s*\[[^\]]*\])?\s*\}'
        matches = re.finditer(tool_pattern, content)
        
        for match in matches:
            try:
                obj_json = match.group(0)
                obj = json.loads(obj_json)
                if isinstance(obj, dict) and "tool" in obj:
                    objects.append(obj)
            except:
                continue
        
        # Essayer aussi un pattern plus large pour capturer des objets incomplets
        broader_pattern = r'\{[^}]*"tool"\s*:\s*"[^"]+"\s*[^}]*\}'
        broader_matches = re.finditer(broader_pattern, content)
        
        for match in broader_matches:
            try:
                obj_json = match.group(0)
                # Essayer de nettoyer et parser l'objet
                obj_json = obj_json.replace('\n', '').replace('\r', '')
                obj = json.loads(obj_json)
                if isinstance(obj, dict) and "tool" in obj and obj not in objects:
                    objects.append(obj)
            except:
                continue
        
        return objects
    
    def _strategy_documentation_rescue(self, content: str) -> List[Dict[str, Any]]:
        """Stratégie 5: Récupération spécialisée pour documentation."""
        tools = []
        
        # Chercher des patterns create_document
        doc_pattern = r'"tool"\s*:\s*"create_document"[^}]*"filename"\s*:\s*"([^"]*)"'
        matches = re.finditer(doc_pattern, content)
        
        for match in matches:
            filename = match.group(1)
            # Créer un objet outil basique
            tool_data = {
                "tool": "create_document",
                "parameters": {"filename": filename}
            }
            
            # Extraire le contenu si possible
            content_match = re.search(r'"content"\s*:\s*"([^"]*)"', match.group(0))
            if content_match:
                tool_data["parameters"]["content"] = content_match.group(1)
            
            tools.append(tool_data)
        
        return tools
    
    def _strategy_regex_fallback(self, content: str) -> List[Dict[str, Any]]:
        """Stratégie 6: Fallback regex pour extraction basique."""
        tools = []
        
        # Patterns pour différents outils
        patterns = {
            "create_document": r'"tool"\s*:\s*"create_document"',
            "implement_code": r'"tool"\s*:\s*"implement_code"',
            "create_tests": r'"tool"\s*:\s*"create_tests"',
            "search_context": r'"tool"\s*:\s*"search_context"'
        }
        
        for tool_name, pattern in patterns.items():
            if re.search(pattern, content):
                # Extraire les paramètres basiques
                filename_match = re.search(r'"filename"\s*:\s*"([^"]*)"', content)
                tool_data = {
                    "tool": tool_name,
                    "parameters": {}
                }
                
                if filename_match:
                    tool_data["parameters"]["filename"] = filename_match.group(1)
                
                # Extraire d'autres paramètres communs
                for param_name in ["content", "query", "code", "description"]:
                    param_pattern = f'"{param_name}"\\s*:\\s*"([^"]*)"'
                    param_match = re.search(param_pattern, content)
                    if param_match:
                        tool_data["parameters"][param_name] = param_match.group(1)
                
                tools.append(tool_data)
        
        return tools
    
    # === STRATÉGIES UNIVERSELLES ROBUSTES ===
    
    def _strategy_markdown_json5(self, response: str) -> Dict[str, Any]:
        """Parse JSON5 depuis bloc markdown ```json ... ```"""
        if '```json' in response:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                self.logger.debug("JSON5 extrait depuis bloc markdown")
                try:
                    import json5
                    return json5.loads(json_content)
                except ImportError:
                    import json
                    return json.loads(json_content)
        return {}
    
    def _strategy_json5_direct_universal(self, content: str) -> Any:
        """Parse JSON5 direct universel (dict ou list)."""
        try:
            import json5
            parsed = json5.loads(content)
            return parsed
        except ImportError:
            import json
            parsed = json.loads(content)
            return parsed
        except:
            return {}
    
    def _strategy_fix_incomplete_universal(self, content: str) -> Any:
        """Répare JSON incomplet universel."""
        open_braces = content.count('{')
        close_braces = content.count('}')
        
        if open_braces > close_braces:
            fixed_content = content + ('}' * (open_braces - close_braces))
            try:
                import json5
                return json5.loads(fixed_content)
            except:
                import json
                return json.loads(fixed_content)
        return {}
    
    def _strategy_progressive_parse_universal(self, content: str) -> Any:
        """Parse progressif universel pour objets/listes."""
        # Essayer de détecter si c'est une liste ou un objet
        content_stripped = content.strip()
        if content_stripped.startswith('['):
            return self._strategy_progressive_parse(content)
        elif content_stripped.startswith('{'):
            # Parse objet unique avec équilibrage d'accolades
            brace_count = 0
            end_pos = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(content):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
            
            if brace_count == 0 and end_pos > 0:
                obj_json = content[:end_pos]
                try:
                    import json5
                    return json5.loads(obj_json)
                except:
                    import json
                    return json.loads(obj_json)
        return {}
    
    def _strategy_extract_partial_universal(self, content: str) -> Any:
        """Extraction partielle universelle."""
        # Essayer d'extraire un objet JSON basique
        json_pattern = r'\{[^{}]*\}'
        match = re.search(json_pattern, content)
        if match:
            try:
                import json5
                return json5.loads(match.group(0))
            except:
                try:
                    import json
                    return json.loads(match.group(0))
                except:
                    pass
        return {}
    
    def _strategy_documentation_rescue_universal(self, content: str) -> Any:
        """Récupération spécialisée universelle."""
        # Recherche de patterns de documentation basiques
        if '"tool"' in content and '"create_document"' in content:
            return self._strategy_documentation_rescue(content)
        return {}
    
    def _strategy_regex_fallback_universal(self, content: str) -> Any:
        """Fallback regex universel."""
        # Extraire des champs basiques avec regex
        decision_match = re.search(r'"decision"\s*:\s*"([^"]*)"', content)
        if decision_match:
            result = {"decision": decision_match.group(1)}
            
            # Chercher d'autres champs communs
            for field in ["success_rate", "confidence", "reason"]:
                field_match = re.search(f'"{field}"\s*:\s*([^,}}]+)', content)
                if field_match:
                    value = field_match.group(1).strip().strip('"')
                    try:
                        # Essayer de convertir en nombre
                        if '.' in value:
                            result[field] = float(value)
                        else:
                            result[field] = int(value)
                    except:
                        result[field] = value
            
            return result
        return {}
    
    # === STRATÉGIES POUR RÉPONSES LLM ===
    
    def _strategy_markdown_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON depuis bloc markdown ```json ... ```"""
        if '```json' in response:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                self.logger.debug("JSON extrait depuis bloc markdown")
                return json.loads(json_content)
        return {}
    
    def _strategy_markdown_generic(self, response: str) -> Dict[str, Any]:
        """Parse JSON depuis bloc markdown générique ``` ... ```"""
        if '```' in response and '```json' not in response:
            json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                if json_content.startswith('{') or json_content.startswith('['):
                    self.logger.debug("JSON extrait depuis bloc markdown générique")
                    return json.loads(json_content)
        return {}
    
    def _strategy_json_direct(self, response: str) -> Dict[str, Any]:
        """Parse JSON direct depuis réponse."""
        response_clean = response.strip()
        if response_clean.startswith('{') or response_clean.startswith('['):
            self.logger.debug("JSON parsé directement")
            return json.loads(response_clean)
        return {}
    
    def _strategy_embedded_json(self, response: str) -> Dict[str, Any]:
        """Chercher JSON intégré dans le texte."""
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, response)
        
        for match in json_matches:
            try:
                parsed = json.loads(match)
                self.logger.debug("JSON trouvé intégré dans le texte")
                return parsed
            except:
                continue
        return {}
    
    def _strategy_json5_direct_single(self, content: str) -> Dict[str, Any]:
        """Parse JSON5 direct pour objet unique."""
        try:
            import json5
            parsed = json5.loads(content)
            if isinstance(parsed, dict):
                return parsed
            else:
                return {}
        except ImportError:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
            else:
                return {}


# Instance globale pour utilisation simple
_default_parser = None

def get_json_parser(logger_name: str = "JSONParser") -> RobustJSONParser:
    """Récupère une instance du parser JSON (singleton par logger)."""
    global _default_parser
    if _default_parser is None:
        _default_parser = RobustJSONParser(logger_name)
    return _default_parser