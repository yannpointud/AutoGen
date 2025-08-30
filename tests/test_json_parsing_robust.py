"""
Test unitaire pour le parsing JSON robuste - Priorité: CRITIQUE
Teste la méthode _parse_tool_calls de BaseAgent avec JSON malformé, incomplet
ou avec commentaires pour valider les stratégies de récupération.
"""

import unittest
from unittest.mock import patch, Mock
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from utils.logger import setup_logger


class MockAgent(BaseAgent):
    """Agent de test qui hérite de BaseAgent."""
    
    def __init__(self):
        # Mock du logger pour éviter les dépendances
        self.logger = Mock()
        self.agent_name = "TestAgent"
        self.name = "TestAgent"
        self.project_name = "TestProject"
        self.available_tools = {}
        self.llm_connector = Mock()
        self.state = {}
        self.current_milestone_id = None
    
    def think(self, task_context, **kwargs):
        return "Test think"
    
    def act(self, plan, task_context, **kwargs):
        return []
    
    def communicate(self, message, recipient=None):
        return "Test communication"


class TestJSONParsingRobust(unittest.TestCase):
    """Tests du parsing JSON robuste dans BaseAgent."""
    
    def setUp(self):
        """Setup avant chaque test."""
        self.agent = MockAgent()
    
    def test_valid_json_parsing(self):
        """Test parsing JSON parfaitement valide."""
        valid_json = '''```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "test.py",
            "content": "print('hello')"
        }
    },
    {
        "tool": "run_command",
        "parameters": {
            "command": "python test.py"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(valid_json)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["tool"], "create_file")
        self.assertEqual(result[0]["parameters"]["filename"], "test.py")
        self.assertEqual(result[1]["tool"], "run_command")
        self.assertEqual(result[1]["parameters"]["command"], "python test.py")
    
    def test_malformed_json_missing_brackets(self):
        """Test parsing JSON avec crochets manquants."""
        malformed_json = '''```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "test.py",
            "content": "print('hello')"
        }
    },
    {
        "tool": "run_command",
        "parameters": {
            "command": "python test.py"
        }
    }
// Missing closing bracket ]
```'''
        
        result = self.agent._parse_tool_calls(malformed_json)
        
        # Doit récupérer au moins un outil avec la stratégie de réparation
        self.assertGreaterEqual(len(result), 1)
        self.assertTrue(any(tool["tool"] == "create_file" for tool in result))
    
    def test_malformed_json_trailing_commas(self):
        """Test parsing JSON avec virgules en fin."""
        trailing_comma_json = '''```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "test.py",
            "content": "print('hello')",
        },
    },
]
```'''
        
        result = self.agent._parse_tool_calls(trailing_comma_json)
        
        # Doit gérer les virgules en fin et extraire l'outil
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0]["tool"], "create_file")
    
    def test_json_with_comments(self):
        """Test parsing JSON avec commentaires (invalid JSON mais common)."""
        json_with_comments = '''```json
[
    // Premier outil pour créer le fichier
    {
        "tool": "create_file",
        "parameters": {
            "filename": "test.py", // Le nom du fichier
            "content": "print('hello')" // Le contenu
        }
    },
    /* Deuxième outil pour exécuter */
    {
        "tool": "run_command",
        "parameters": {
            "command": "python test.py"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(json_with_comments)
        
        # json5 devrait gérer les commentaires
        self.assertGreaterEqual(len(result), 1)
        self.assertTrue(any(tool["tool"] == "create_file" for tool in result))
    
    def test_incomplete_json_partial_objects(self):
        """Test parsing JSON incomplet avec objets partiels."""
        incomplete_json = '''```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "test.py",
            "content": "print('hello world')"
        }
    },
    {
        "tool": "run_command",
        "parameters": {
            "command": "python te
```'''
        
        result = self.agent._parse_tool_calls(incomplete_json)
        
        # Doit extraire au moins le premier outil complet
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0]["tool"], "create_file")
        self.assertEqual(result[0]["parameters"]["filename"], "test.py")
    
    def test_nested_json_in_parameters(self):
        """Test parsing avec JSON imbriqué dans les paramètres."""
        nested_json = '''```json
[
    {
        "tool": "create_config",
        "parameters": {
            "filename": "config.json",
            "content": "{\\"database\\": {\\"host\\": \\"localhost\\", \\"port\\": 5432}}"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(nested_json)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool"], "create_config")
        self.assertIn("database", result[0]["parameters"]["content"])
    
    def test_mixed_quotes_json(self):
        """Test parsing JSON avec mélange guillemets simples/doubles."""
        mixed_quotes_json = '''```json
[
    {
        'tool': "create_file",
        "parameters": {
            'filename': "test.py",
            "content": 'print("hello world")'
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(mixed_quotes_json)
        
        # json5 devrait gérer les guillemets mixtes
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0]["tool"], "create_file")
    
    def test_large_code_content(self):
        """Test parsing JSON avec gros contenu de code."""
        large_code = "\\n".join([f"# Line {i}" for i in range(100)])
        large_json = f'''```json
[
    {{
        "tool": "implement_code",
        "parameters": {{
            "filename": "large_file.py",
            "language": "python",
            "code": "{large_code}",
            "description": "Un gros fichier de test"
        }}
    }}
]
```'''
        
        result = self.agent._parse_tool_calls(large_json)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool"], "implement_code")
        self.assertEqual(result[0]["parameters"]["filename"], "large_file.py")
        self.assertIn("Line 50", result[0]["parameters"]["code"])
    
    def test_multiple_code_blocks(self):
        """Test parsing avec plusieurs blocs de code JSON."""
        multiple_blocks = '''Voici les outils à utiliser:

```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "first.py",
            "content": "print('first')"
        }
    }
]
```

Et aussi:

```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "second.py",
            "content": "print('second')"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(multiple_blocks)
        
        # Doit extraire des outils des deux blocs
        self.assertEqual(len(result), 2)
        filenames = [tool["parameters"]["filename"] for tool in result]
        self.assertIn("first.py", filenames)
        self.assertIn("second.py", filenames)
    
    def test_json_without_code_blocks(self):
        """Test parsing JSON directement sans marqueurs ```."""
        direct_json = '''[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "direct.py",
            "content": "print('direct')"
        }
    }
]'''
        
        # Ce test vérifie si le parsing fonctionne même sans ``` 
        # (selon l'implémentation actuelle, il faut les ```)
        result = self.agent._parse_tool_calls(direct_json)
        
        # Peut ne pas fonctionner selon l'implémentation
        # Mais ne doit pas planter
        self.assertIsInstance(result, list)
    
    def test_malformed_parameters_object(self):
        """Test parsing avec objet parameters malformé."""
        bad_params_json = '''```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "test.py"
            // Missing comma and value
            "content"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(bad_params_json)
        
        # Peut récupérer partiellement ou échouer gracieusement
        self.assertIsInstance(result, list)
        # Ne doit pas planter même si récupération partielle
    
    def test_empty_and_null_responses(self):
        """Test parsing de réponses vides ou nulles."""
        test_cases = [
            "",
            None,
            "```json\n```",
            "```json\n\n```",
            "```json\n[]\n```"
        ]
        
        for test_input in test_cases:
            if test_input is None:
                continue
            result = self.agent._parse_tool_calls(test_input)
            self.assertIsInstance(result, list)
            # Résultat peut être vide mais doit être une liste
    
    def test_very_long_response_protection(self):
        """Test protection contre les réponses trop longues."""
        # Créer une réponse de plus de 50000 caractères
        long_content = "x" * 60000
        long_response = f'```json\n{{"tool": "test", "parameters": {{"content": "{long_content}"}}}}\n```'
        
        result = self.agent._parse_tool_calls(long_response)
        
        # Doit retourner une liste vide avec un warning
        self.assertEqual(result, [])
        # Vérifier que le logger a été appelé avec un warning
        self.agent.logger.warning.assert_called()
    
    def test_special_characters_in_content(self):
        """Test parsing avec caractères spéciaux dans le contenu."""
        special_chars_json = '''```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "special.py",
            "content": "print('Héllo wørld! 你好 🌍')"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(special_chars_json)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool"], "create_file")
        self.assertIn("Héllo", result[0]["parameters"]["content"])
        self.assertIn("🌍", result[0]["parameters"]["content"])
    
    def test_documentation_rescue_strategy(self):
        """Test stratégie de récupération pour documentation."""
        doc_content_json = '''```json
[
    {
        "tool": "create_project_file",
        "parameters": {
            "filename": "README.md",
            "content": "# Project Title\\n\\nThis is incomplete...
```'''
        
        result = self.agent._parse_tool_calls(doc_content_json)
        
        # Doit utiliser la stratégie de récupération documentation
        # Peut récupérer partiellement
        self.assertIsInstance(result, list)
    
    def test_progressive_parsing_strategy(self):
        """Test stratégie de parsing progressif."""
        # JSON avec structure complexe qui pourrait échouer en parsing global
        complex_json = '''```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "test1.py",
            "content": "print('test1')"
        }
    },
    // Complex structure that might break global parsing
    {
        "tool": "implement_code",
        "parameters": {
            "filename": "complex.py",
            "language": "python", 
            "code": "def complex_func():\\n    return 'complex'",
            "description": "Complex implementation"
        }
    },
    {
        "tool": "run_tests",
        "parameters": {
            "test_file": "test_complex.py"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(complex_json)
        
        # Doit extraire plusieurs outils même si structure complexe
        self.assertGreaterEqual(len(result), 2)
        tool_names = [tool["tool"] for tool in result]
        self.assertIn("create_file", tool_names)
    
    def test_all_strategies_fail(self):
        """Test comportement quand toutes les stratégies échouent."""
        completely_broken = '''This is not JSON at all!
        Random text without any structure.
        No tools, no parameters, nothing useful.
        ```
        Still not JSON!
        ```'''
        
        result = self.agent._parse_tool_calls(completely_broken)
        
        # Doit retourner liste vide (le logging peut varier selon l'implémentation)
        self.assertEqual(result, [])
        # Vérifier que c'est bien une liste et non None ou autre
        self.assertIsInstance(result, list)
    
    def test_fulfills_deliverable_preservation(self):
        """Test CRITIQUE: Vérifier que fulfills_deliverable est préservé dans toutes les stratégies."""
        json_with_fulfills = '''```json
[
    {
        "tool": "create_document",
        "parameters": {
            "filename": "architecture",
            "content": "# Architecture du système"
        },
        "fulfills_deliverable": ["Documentation", "Architecture"]
    },
    {
        "tool": "implement_code",
        "parameters": {
            "filename": "main.py",
            "language": "python",
            "code": "def main(): pass"
        },
        "fulfills_deliverable": ["Code principal"]
    }
]
```'''
        
        result = self.agent._parse_tool_calls(json_with_fulfills)
        
        self.assertEqual(len(result), 2)
        
        # Vérifier que fulfills_deliverable est préservé pour create_document
        doc_tool = next((tool for tool in result if tool["tool"] == "create_document"), None)
        self.assertIsNotNone(doc_tool)
        self.assertIn("fulfills_deliverable", doc_tool)
        self.assertEqual(doc_tool["fulfills_deliverable"], ["Documentation", "Architecture"])
        
        # Vérifier que fulfills_deliverable est préservé pour implement_code  
        code_tool = next((tool for tool in result if tool["tool"] == "implement_code"), None)
        self.assertIsNotNone(code_tool)
        self.assertIn("fulfills_deliverable", code_tool)
        self.assertEqual(code_tool["fulfills_deliverable"], ["Code principal"])
    
    def test_progressive_parsing_with_incomplete_json(self):
        """Test stratégie progressive avec JSON incomplet."""
        # JSON qui pourrait échouer avec la stratégie directe mais être récupéré par la progressive
        incomplete_json = '''```json
[
    {
        "tool": "create_document",
        "parameters": {
            "filename": "specs",
            "content": "Spécifications très longues..."
        }
    },
    {
        "tool": "create_project_file",
        "parameters": {
            "filename": "config.yaml",
            "content": "database:\n  host: localhost"
        }
    // Missing closing bracket to force progressive parsing
```'''
        
        result = self.agent._parse_tool_calls(incomplete_json)
        
        # Doit récupérer au moins un outil
        self.assertGreaterEqual(len(result), 1)
        
        # Vérifier la structure basique des outils récupérés
        for tool in result:
            self.assertIn("tool", tool)
            self.assertIn("parameters", tool)
    
    def test_documentation_rescue_with_fulfills_deliverable(self):
        """Test stratégie documentation rescue avec fulfills_deliverable."""
        malformed_doc_json = '''```json
[
    {
        "tool": "create_document",
        "parameters": {
            "filename": "README",
            "content": "# Projet important\n\nCette documentation est..."
        },
        "fulfills_deliverable": ["Documentation utilisateur", "README.md"]
    // JSON cassé ici pour forcer la stratégie rescue
```'''
        
        result = self.agent._parse_tool_calls(malformed_doc_json)
        
        # Si la stratégie rescue fonctionne, elle devrait préserver fulfills_deliverable
        if len(result) > 0:
            doc_tool = result[0]
            self.assertEqual(doc_tool["tool"], "create_document")
            # Peut avoir fulfills_deliverable selon la capacité de rescue
            if "fulfills_deliverable" in doc_tool:
                self.assertIsInstance(doc_tool["fulfills_deliverable"], list)
    
    def test_regex_fallback_with_fulfills_deliverable(self):
        """Test stratégie regex fallback avec fulfills_deliverable."""
        broken_json_with_context = '''Completement cassé mais contient:
        
        "tool": "create_document"
        "parameters": {"filename": "test", "content": "contenu"}
        "fulfills_deliverable": ["Test documentation"]
        
        Et aussi:
        "tool": "implement_code"  
        "fulfills_deliverable": ["Code de test"]
        "parameters": {"filename": "test.py", "code": "print('test')"}
        '''
        
        result = self.agent._parse_tool_calls(broken_json_with_context)
        
        # La stratégie fallback devrait récupérer quelque chose
        if len(result) > 0:
            # Vérifier qu'au moins certains outils ont fulfills_deliverable
            tools_with_fulfills = [tool for tool in result if "fulfills_deliverable" in tool]
            # Au moins un outil devrait avoir récupéré son fulfills_deliverable
            self.assertGreater(len(tools_with_fulfills), 0, "La stratégie fallback devrait récupérer fulfills_deliverable")


if __name__ == '__main__':
    unittest.main()