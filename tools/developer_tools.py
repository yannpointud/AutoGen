"""
Outils spécifiques à l'agent Developer.
"""

from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from agents.base_agent import ToolResult


def _extract_filename_from_path(filename: str) -> str:
    """Extrait le nom du fichier si un chemin complet est fourni."""
    if '/' in filename:
        return Path(filename).name
    return filename


def tool_implement_code(agent, parameters: Dict[str, Any]) -> ToolResult:
    """Implémente du code."""
    try:
        filename = _extract_filename_from_path(parameters.get('filename', 'module.py'))
        description = parameters.get('description', '')
        language = parameters.get('language')
        code = parameters.get('code', '')
        
        # Si pas de langage fourni, le détecter automatiquement
        if not language:
            language = _detect_language_from_description(agent, description, filename)
        language = language.lower()
        
        # Si pas de code fourni, le générer
        if not code and description:
            code = _generate_code_from_description(agent, description, language, filename)
        
        # Limiter la taille
        max_lines = agent.tools_config.get('specific', {}).get('implement_code', {}).get('max_lines', 1000)
        code_lines = code.split('\n')
        if len(code_lines) > max_lines:
            code_lines = code_lines[:max_lines]
            code_lines.append(f"\n# Code tronqué à {max_lines} lignes")
            code = '\n'.join(code_lines)
        
        # Ajouter l'en-tête
        header = _get_file_header(agent, filename, language)
        full_code = header + code
        
        # Déterminer le chemin
        file_path = agent.src_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder
        file_path.write_text(full_code, encoding='utf-8')
        
        # Indexer dans le RAG
        if agent.rag_engine:
            agent.rag_engine.index_document(
                full_code,
                {
                    'type': 'source_code',
                    'source': str(file_path.relative_to(Path("projects") / agent.project_name)),
                    'agent_name': agent.name,
                    'milestone': agent.current_milestone_id,
                    'language': language,
                    'file_type': file_path.suffix,
                    'preserve': True
                }
            )
        
        return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_create_tests(agent, parameters: Dict[str, Any]) -> ToolResult:
    """Crée des tests."""
    try:
        filename = _extract_filename_from_path(parameters.get('filename', 'test_module.py'))
        target_file = parameters.get('target_file', '')
        test_framework = parameters.get('test_framework', 'pytest')
        code = parameters.get('code', '')
        
        # S'assurer que le nom commence par test_
        if not filename.startswith('test_'):
            filename = 'test_' + filename
        
        # Si pas de code fourni, générer un template
        if not code and target_file:
            code = _generate_test_template(target_file, test_framework)
        
        # Ajouter l'en-tête
        language = _detect_language_from_filename(filename)
        header = _get_file_header(agent, filename, language)
        full_code = header + code
        
        # Par cette version robuste utilisant pathlib
        # Extraire uniquement le nom du fichier pour éviter les chemins imbriqués
        safe_filename = Path(filename).name
        # S'assurer que le nom commence par test_
        if not safe_filename.startswith('test_'):
            safe_filename = 'test_' + safe_filename

        # Sauvegarder dans le dossier tests
        file_path = agent.tests_path / safe_filename
        file_path.write_text(full_code, encoding='utf-8')
        
        # Indexer dans le RAG
        if agent.rag_engine:
            agent.rag_engine.index_document(
                full_code,
                {
                    'type': 'source_code',
                    'source': str(file_path.relative_to(Path("projects") / agent.project_name)),
                    'agent_name': agent.name,
                    'milestone': agent.current_milestone_id,
                    'language': language,
                    'file_type': file_path.suffix,
                    'is_test': True,
                    'preserve': True
                }
            )
        
        return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_create_project_file(agent, parameters: Dict[str, Any]) -> ToolResult:
    """Crée un fichier de projet."""
    try:
        filename = parameters.get('filename', '')
        content = parameters.get('content', '')
        
        if not filename:
            return ToolResult('error', error="Nom de fichier requis")
        
        # Déterminer le chemin de base
        project_path = Path("projects") / agent.project_name
        
        # Gérer les chemins relatifs
        if filename.startswith('/'):
            filename = filename[1:]
        
        file_path = project_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder
        file_path.write_text(content, encoding='utf-8')
        
        # Indexer seulement certains types de fichiers
        indexable_extensions = ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']
        if file_path.suffix in indexable_extensions and agent.rag_engine:
            agent.rag_engine.index_document(
                content,
                {
                    'type': 'project_file',
                    'source': str(file_path.relative_to(project_path)),
                    'agent_name': agent.name,
                    'milestone': agent.current_milestone_id,
                    'file_type': file_path.suffix
                }
            )
        
        return ToolResult('success', result={'created': str(file_path)}, artifact=str(file_path))
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def _generate_code_from_description(agent, description: str, language: str, filename: str) -> str:
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
        code = agent.generate_with_context(
            prompt=prompt,
            temperature=0.3,
        )
        
        # Nettoyer le code
        code = _clean_generated_code(code, language)
        return code
        
    except Exception as e:
        agent.logger.error(f"Erreur lors de la génération de code pour {filename}: {str(e)}")
        return f"# Erreur lors de la génération: {str(e)}\n# TODO: Implémenter {description}"


def _generate_test_template(target_file: str, framework: str) -> str:
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


def _get_file_header(agent, filename: str, language: str) -> str:
    """Génère l'en-tête d'un fichier."""
    headers = {
        'python': f'''"""
{filename}

Généré par: {agent.name}
Date: {datetime.now().strftime('%Y-%m-%d')}
Projet: {agent.project_name}
"""

''',
        'javascript': f'''/**
 * {filename}
 * 
 * Généré par: {agent.name}
 * Date: {datetime.now().strftime('%Y-%m-%d')}
 * Projet: {agent.project_name}
 */

''',
        'typescript': f'''/**
 * {filename}
 * 
 * Généré par: {agent.name}
 * Date: {datetime.now().strftime('%Y-%m-%d')}
 * Projet: {agent.project_name}
 */

'''
    }
    
    return headers.get(language, f"// {filename} - Généré le {datetime.now().strftime('%Y-%m-%d')}\n\n")


def _detect_language_from_filename(filename: str) -> str:
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


def _detect_language_from_description(agent, description: str, filename: str) -> str:
    """Détecte le langage à partir de la description et du nom de fichier."""
    # D'abord essayer depuis l'extension
    if filename:
        detected_from_ext = _detect_language_from_filename(filename)
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
            response = agent.generate_with_context(prompt, temperature=0.1)
            detected = response.strip().lower()
            valid_languages = ['python', 'javascript', 'typescript', 'java', 'go', 'rust', 'php', 'csharp', 'ruby']
            if detected in valid_languages:
                return detected
        except Exception as e:
            agent.logger.debug(f"Erreur détection langage: {e}")
    
    return 'python'  # Fallback


def _clean_generated_code(code: str, language: str) -> str:
    """Nettoie le code généré par le LLM."""
    # Retirer les marqueurs de code
    if code.startswith(f"```{language}"):
        code = code[len(f"```{language}"):]
    elif code.startswith("```"):
        code = code[3:]
    
    if code.endswith("```"):
        code = code[:-3]
    
    return code.strip()