"""
Agent Developer utilisant une architecture orientée outils.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json
import re

from agents.base_agent import BaseAgent, Tool, ToolResult
from core.llm_connector import LLMFactory
from config import default_config


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
            self._tool_implement_code
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
            self._tool_create_tests
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
            self._tool_create_project_file
        )



    def _tool_implement_code(self, parameters: Dict[str, Any]) -> ToolResult:
        """Implémente du code."""
        try:
            filename = parameters.get('filename', 'module.py')
            description = parameters.get('description', '')
            language = parameters.get('language')
            code = parameters.get('code', '')
            
            # Si pas de langage fourni, le détecter automatiquement
            if not language:
                language = self._detect_language_from_description(description, filename)
            language = language.lower()
            
            # Si pas de code fourni, le générer
            if not code and description:
                code = self._generate_code_from_description(description, language, filename)
            
            # Limiter la taille
            max_lines = self.tools_config.get('specific', {}).get('implement_code', {}).get('max_lines', 1000)
            code_lines = code.split('\n')
            if len(code_lines) > max_lines:
                code_lines = code_lines[:max_lines]
                code_lines.append(f"\n# Code tronqué à {max_lines} lignes")
                code = '\n'.join(code_lines)
            
            # Ajouter l'en-tête
            header = self._get_file_header(filename, language)
            full_code = header + code
            
            # Déterminer le chemin
            file_path = self.src_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder
            file_path.write_text(full_code, encoding='utf-8')
            
            # Indexer dans le RAG
            if self.rag_engine:
                self.rag_engine.index_document(
                    full_code,
                    {
                        'type': 'source_code',
                        'source': str(file_path.relative_to(Path("projects") / self.project_name)),
                        'agent_name': self.name,
                        'milestone': self.current_milestone_id,
                        'language': language,
                        'file_type': file_path.suffix,
                        'preserve': True
                    }
                )
            
            return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _tool_create_tests(self, parameters: Dict[str, Any]) -> ToolResult:
        """Crée des tests."""
        try:
            filename = parameters.get('filename', 'test_module.py')
            target_file = parameters.get('target_file', '')
            test_framework = parameters.get('test_framework', 'pytest')
            code = parameters.get('code', '')
            
            # S'assurer que le nom commence par test_
            if not filename.startswith('test_'):
                filename = 'test_' + filename
            
            # Si pas de code fourni, générer un template
            if not code and target_file:
                code = self._generate_test_template(target_file, test_framework)
            
            # Ajouter l'en-tête
            language = self._detect_language_from_filename(filename)
            header = self._get_file_header(filename, language)
            full_code = header + code
            
            # Par cette version robuste utilisant pathlib
            # Extraire uniquement le nom du fichier pour éviter les chemins imbriqués
            safe_filename = Path(filename).name
            # S'assurer que le nom commence par test_
            if not safe_filename.startswith('test_'):
                safe_filename = 'test_' + safe_filename

            # Sauvegarder dans le dossier tests
            file_path = self.tests_path / safe_filename
            file_path.write_text(full_code, encoding='utf-8')
            
            # Indexer dans le RAG
            if self.rag_engine:
                self.rag_engine.index_document(
                    full_code,
                    {
                        'type': 'source_code',
                        'source': str(file_path.relative_to(Path("projects") / self.project_name)),
                        'agent_name': self.name,
                        'milestone': self.current_milestone_id,
                        'language': language,
                        'file_type': file_path.suffix,
                        'is_test': True,
                        'preserve': True
                    }
                )
            
            return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _tool_create_project_file(self, parameters: Dict[str, Any]) -> ToolResult:
        """Crée un fichier de projet."""
        try:
            filename = parameters.get('filename', '')
            content = parameters.get('content', '')
            
            if not filename:
                return ToolResult('error', error="Nom de fichier requis")
            
            # Déterminer le chemin de base
            project_path = Path("projects") / self.project_name
            
            # Gérer les chemins relatifs
            if filename.startswith('/'):
                filename = filename[1:]
            
            file_path = project_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder
            file_path.write_text(content, encoding='utf-8')
            
            # Indexer seulement certains types de fichiers
            indexable_extensions = ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']
            if file_path.suffix in indexable_extensions and self.rag_engine:
                self.rag_engine.index_document(
                    content,
                    {
                        'type': 'project_file',
                        'source': str(file_path.relative_to(project_path)),
                        'agent_name': self.name,
                        'milestone': self.current_milestone_id,
                        'file_type': file_path.suffix
                    }
                )
            
            return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _generate_code_from_description(self, description: str, language: str, filename: str) -> str:
        """Génère du code à partir d'une description."""
        prompt = f"""TA MISSION : Générer un fichier de code {language} complet et fonctionnel pour la tâche suivante : {description}

    FICHIER CIBLE : {filename}

    EXIGENCES STRICTES :
    1.  **COMPLÉTUDE ABSOLUE** : Tu dois écrire l'intégralité du code demandé. N'omets AUCUNE fonction, classe ou méthode.
    2.  **AUCUN PLACEHOLDER** : Il est INTERDIT d'utiliser des commentaires placeholders comme `# ...`, `# TODO`, ou des ellipses (`...`) pour remplacer du code. Le code doit être COMPLET.
    3.  **PRÊT À L'EMPLOI** : Le fichier généré doit être directement utilisable et exécutable sans AUCUNE modification.
    4.  **CODE UNIQUEMENT** : Ta réponse ne doit contenir QUE le code source, sans aucune phrase d'introduction, explication ou conclusion.

    Commence à générer le code maintenant.
    """
        
        try:
            code = self.generate_with_context(
                prompt=prompt,
                temperature=0.3,
            )
            
            # Nettoyer le code
            code = self._clean_generated_code(code, language)
            return code
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de code pour {filename}: {str(e)}")
            return f"# Erreur lors de la génération: {str(e)}\n# TODO: Implémenter {description}"
    
    def _generate_test_template(self, target_file: str, framework: str) -> str:
        """Génère un template de test."""
        templates = {
            'pytest': f"""import pytest
from src.{target_file.replace('.py', '')} import *


class Test{target_file.replace('.py', '').title()}:
    \"\"\"Tests pour {target_file}\"\"\"
    
    def test_example(self):
        \"\"\"Test exemple\"\"\"
        assert True
        
    # TODO: Ajouter les tests
""",
            'jest': f"""const {{ }} = require('../{target_file.replace('.js', '')}');

describe('{target_file}', () => {{
    test('example test', () => {{
        expect(true).toBe(true);
    }});
    
    // TODO: Ajouter les tests
}});
""",
            'unittest': f"""import unittest
from src.{target_file.replace('.py', '')} import *


class Test{target_file.replace('.py', '').title()}(unittest.TestCase):
    \"\"\"Tests pour {target_file}\"\"\"
    
    def test_example(self):
        \"\"\"Test exemple\"\"\"
        self.assertTrue(True)
        
    # TODO: Ajouter les tests


if __name__ == '__main__':
    unittest.main()
"""
        }
        
        return templates.get(framework, "# TODO: Implémenter les tests")
    
    def _get_file_header(self, filename: str, language: str) -> str:
        """Génère l'en-tête d'un fichier."""
        headers = {
            'python': f'''"""
{filename}

Généré par: {self.name}
Date: {datetime.now().strftime('%Y-%m-%d')}
Projet: {self.project_name}
"""

''',
            'javascript': f'''/**
 * {filename}
 * 
 * Généré par: {self.name}
 * Date: {datetime.now().strftime('%Y-%m-%d')}
 * Projet: {self.project_name}
 */

''',
            'typescript': f'''/**
 * {filename}
 * 
 * Généré par: {self.name}
 * Date: {datetime.now().strftime('%Y-%m-%d')}
 * Projet: {self.project_name}
 */

'''
        }
        
        return headers.get(language, f"// {filename} - Généré le {datetime.now().strftime('%Y-%m-%d')}\n\n")
    
    def _detect_language_from_filename(self, filename: str) -> str:
        """Détecte le langage depuis l'extension du fichier."""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cs': 'csharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php'
        }
        
        ext = Path(filename).suffix
        return ext_to_lang.get(ext, 'python')
    
    def _detect_language_from_description(self, description: str, filename: str) -> str:
        """Détecte le langage à partir de la description et du nom de fichier."""
        # D'abord essayer depuis l'extension
        if filename:
            detected_from_ext = self._detect_language_from_filename(filename)
            # Si l'extension donne un résultat clair (pas le fallback), l'utiliser
            ext = Path(filename).suffix
            if ext in ['.js', '.jsx', '.ts', '.tsx', '.java', '.cs', '.go', '.rb', '.php']:
                return detected_from_ext
        
        # Si pas d'extension claire, demander au LLM
        if description:
            prompt = f"""Quel langage de programmation pour : "{description}"?
Fichier: {filename}
Réponds uniquement par: python/javascript/typescript/java/go/rust/php (un mot)"""
            
            try:
                response = self.generate_with_context(prompt, temperature=0.1)
                detected = response.strip().lower()
                valid_languages = ['python', 'javascript', 'typescript', 'java', 'go', 'rust', 'php', 'csharp', 'ruby']
                if detected in valid_languages:
                    return detected
            except Exception as e:
                self.logger.debug(f"Erreur détection langage: {e}")
        
        return 'python'  # Fallback
    
    def _clean_generated_code(self, code: str, language: str) -> str:
        """Nettoie le code généré par le LLM."""
        # Retirer les marqueurs de code
        if code.startswith(f"```{language}"):
            code = code[len(f"```{language}"):]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
    
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