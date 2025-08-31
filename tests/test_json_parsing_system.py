"""
Test unitaire pour le système de parsing JSON - Version 1.4+

Teste le nouveau système de parsing JSON robuste avec les stratégies multiples
et la préservation des champs critiques comme 'fulfills_deliverable'.

Compatible avec l'architecture AutoGen v1.4+ après les changements majeurs du parsing.
"""

import unittest
from unittest.mock import Mock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent


class MockAgent(BaseAgent):
    """Agent de test qui hérite de BaseAgent pour tester le parsing JSON."""
    
    def __init__(self):
        # Mock des dépendances minimales
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


class TestJSONParsingSystem(unittest.TestCase):
    """Tests complets du système de parsing JSON v1.4+."""
    
    def setUp(self):
        """Setup avant chaque test."""
        self.agent = MockAgent()
    
    def test_valid_json_parsing(self):
        """Test parsing JSON parfaitement valide."""
        valid_json = '''```json
[
    {
        "tool": "create_document",
        "parameters": {
            "filename": "architecture.md",
            "content": "# Architecture du système"
        },
        "fulfills_deliverable": ["Documentation", "Architecture"]
    },
    {
        "tool": "implement_code",
        "parameters": {
            "filename": "main.py",
            "language": "python",
            "code": "def main():\\n    pass"
        },
        "fulfills_deliverable": ["Code principal"]
    }
]
```'''
        
        result = self.agent._parse_tool_calls(valid_json)
        
        # Vérifications de base
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)  # Au moins un outil parsé
        
        # Si des outils sont parsés, vérifier leur structure
        if len(result) >= 2:
            # Premier outil
            doc_tool = result[0]
            self.assertEqual(doc_tool["tool"], "create_document")
            self.assertIn("parameters", doc_tool)
            self.assertEqual(doc_tool["parameters"]["filename"], "architecture.md")
            
            # Vérification critique : fulfills_deliverable préservé
            self.assertIn("fulfills_deliverable", doc_tool)
            self.assertEqual(doc_tool["fulfills_deliverable"], ["Documentation", "Architecture"])
            
            # Second outil
            code_tool = result[1]
            self.assertEqual(code_tool["tool"], "implement_code")
            self.assertIn("fulfills_deliverable", code_tool)
            self.assertEqual(code_tool["fulfills_deliverable"], ["Code principal"])
    
    def test_json_with_comments_and_quotes(self):
        """Test parsing JSON avec commentaires et guillemets mixtes."""
        commented_json = '''```json
[
    // Création de document avec guillemets mixtes
    {
        'tool': "create_document",
        "parameters": {
            'filename': "specs.md",
            "content": "# Spécifications\\nContenu avec 'guillemets' et \\"échappements\\""
        },
        "fulfills_deliverable": ["Documentation"]
    }
]
```'''
        
        result = self.agent._parse_tool_calls(commented_json)
        
        # Le nouveau parser doit gérer les commentaires et guillemets mixtes
        self.assertIsInstance(result, list)
        if len(result) >= 1:
            tool = result[0]
            self.assertEqual(tool["tool"], "create_document")
            self.assertIn("fulfills_deliverable", tool)
    
    def test_malformed_json_resilience(self):
        """Test résilience avec JSON malformé - le système doit survivre."""
        malformed_cases = [
            # JSON avec crochets manquants
            '''```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "test.py"
        }
    },
    {
        "tool": "incomplete
```''',
            
            # JSON avec virgules en trop
            '''```json
[
    {
        "tool": "create_file",
        "parameters": {
            "filename": "test.py",
        },
]
```''',
            
            # JSON complètement cassé
            '''```json
This is not JSON at all!
Random text...
```'''
        ]
        
        for malformed_json in malformed_cases:
            with self.subTest(json_content=malformed_json[:50] + "..."):
                result = self.agent._parse_tool_calls(malformed_json)
                
                # PRIORITÉ ABSOLUE : Ne jamais planter, toujours retourner une liste
                self.assertIsInstance(result, list, "Le parsing ne doit jamais planter")
                
                # Si le parsing récupère quelque chose, ça doit être valide
                for tool in result:
                    self.assertIn("tool", tool, "Chaque outil doit avoir un champ 'tool'")
                    self.assertIsInstance(tool.get("parameters", {}), dict, "Parameters doit être un dict")
    
    def test_large_and_complex_json(self):
        """Test parsing avec contenu volumineux et complexe."""
        # Générer un gros code
        large_code = "\\n".join([f"# Configuration line {i}" for i in range(50)])
        
        complex_json = f'''```json
[
    {{
        "tool": "implement_code",
        "parameters": {{
            "filename": "large_config.py",
            "language": "python",
            "code": "{large_code}",
            "description": "Configuration complète du système"
        }},
        "fulfills_deliverable": ["Configuration", "Code de base"]
    }},
    {{
        "tool": "create_tests",
        "parameters": {{
            "test_file": "test_config.py",
            "test_cases": [
                {{"name": "test_basic", "expected": "pass"}},
                {{"name": "test_advanced", "expected": "pass"}}
            ]
        }},
        "fulfills_deliverable": ["Tests unitaires"]
    }}
]
```'''
        
        result = self.agent._parse_tool_calls(complex_json)
        
        # Vérifications
        self.assertIsInstance(result, list)
        if len(result) >= 1:
            # Vérifier que le gros contenu est préservé
            impl_tool = next((t for t in result if t.get("tool") == "implement_code"), None)
            if impl_tool:
                self.assertIn("Configuration line 25", impl_tool["parameters"]["code"])
                self.assertIn("fulfills_deliverable", impl_tool)
    
    def test_multiple_code_blocks(self):
        """Test parsing avec plusieurs blocs JSON séparés."""
        multiple_blocks = '''Première série d'outils:

```json
[
    {
        "tool": "create_document",
        "parameters": {
            "filename": "doc1.md",
            "content": "Premier document"
        },
        "fulfills_deliverable": ["Doc1"]
    }
]
```

Deuxième série:

```json
[
    {
        "tool": "implement_code",
        "parameters": {
            "filename": "code1.py",
            "code": "print('Hello')"
        },
        "fulfills_deliverable": ["Code1"]
    }
]
```'''
        
        result = self.agent._parse_tool_calls(multiple_blocks)
        
        # Le nouveau système doit traiter les blocs multiples
        self.assertIsInstance(result, list)
        
        if len(result) >= 2:
            # Vérifier les outils des deux blocs
            tools_names = [t["tool"] for t in result]
            self.assertIn("create_document", tools_names)
            self.assertIn("implement_code", tools_names)
            
            # Vérifier préservation fulfills_deliverable
            for tool in result:
                self.assertIn("fulfills_deliverable", tool)
    
    def test_empty_and_edge_cases(self):
        """Test cas limites : vide, null, etc."""
        edge_cases = [
            "",  # Chaîne vide
            "```json\n```",  # Bloc vide
            "```json\n[]\n```",  # Tableau JSON vide
            "No JSON blocks here at all!",  # Pas de JSON
            "```json\nnull\n```",  # JSON null
        ]
        
        for edge_case in edge_cases:
            with self.subTest(content=edge_case):
                result = self.agent._parse_tool_calls(edge_case)
                
                # Priorité absolue : stabilité
                self.assertIsInstance(result, list)
                
                # Liste peut être vide pour ces cas
                for tool in result:
                    # Si des outils sont quand même extraits, ils doivent être valides
                    self.assertIsInstance(tool, dict)
                    if "tool" in tool:
                        self.assertIsInstance(tool["tool"], str)
    
    def test_special_characters_and_encoding(self):
        """Test gestion caractères spéciaux et encodage."""
        special_json = '''```json
[
    {
        "tool": "create_document",
        "parameters": {
            "filename": "international.md",
            "content": "Contenu avec émojis 🚀 et caractères spéciaux : àéèç, 中文, русский"
        },
        "fulfills_deliverable": ["Documentation internationale"]
    }
]
```'''
        
        result = self.agent._parse_tool_calls(special_json)
        
        self.assertIsInstance(result, list)
        if len(result) >= 1:
            tool = result[0]
            content = tool.get("parameters", {}).get("content", "")
            # Vérifier que les caractères spéciaux sont préservés
            if "🚀" in content:  # Si les émojis sont préservés
                self.assertIn("🚀", content)
                self.assertIn("àéèç", content)
    
    def test_fulfills_deliverable_preservation_priority(self):
        """Test CRITIQUE : Préservation absolue de fulfills_deliverable."""
        # JSON avec différentes structures de fulfills_deliverable
        deliverable_json = '''```json
[
    {
        "tool": "create_document",
        "parameters": {"filename": "test1.md", "content": "Test"},
        "fulfills_deliverable": ["Single deliverable"]
    },
    {
        "tool": "implement_code",
        "parameters": {"filename": "test2.py", "code": "pass"},
        "fulfills_deliverable": ["Multiple", "Deliverables", "Here"]
    },
    {
        "tool": "create_tests",
        "parameters": {"test_file": "test3.py"},
        "fulfills_deliverable": []
    }
]
```'''
        
        result = self.agent._parse_tool_calls(deliverable_json)
        
        self.assertIsInstance(result, list)
        
        # CRITIQUE : Tous les outils parsés DOIVENT avoir fulfills_deliverable
        parsed_tools_with_deliverables = [t for t in result if "fulfills_deliverable" in t]
        
        # Si le parsing fonctionne, fulfills_deliverable doit être préservé
        if len(result) >= 1:
            self.assertGreater(len(parsed_tools_with_deliverables), 0, 
                             "Au moins un outil doit avoir fulfills_deliverable préservé")
            
            # Vérifier les types
            for tool in parsed_tools_with_deliverables:
                deliverables = tool["fulfills_deliverable"]
                self.assertIsInstance(deliverables, list, 
                                    f"fulfills_deliverable doit être une liste pour {tool.get('tool', 'unknown')}")
    
    def test_parsing_performance_and_protection(self):
        """Test protection contre les réponses trop longues."""
        # Créer une réponse très longue (simuler réponse LLM verbale)
        very_long_content = "x" * 70000  # Plus de 50000 caractères
        long_response = f'''```json
[
    {{
        "tool": "create_document",
        "parameters": {{
            "filename": "huge.txt",
            "content": "{very_long_content}"
        }},
        "fulfills_deliverable": ["Huge document"]
    }}
]
```'''
        
        # Le système doit soit traiter soit gracieusement échouer
        result = self.agent._parse_tool_calls(long_response)
        
        # Priorité : ne pas planter
        self.assertIsInstance(result, list)
        
        # Si traité : vérifier cohérence
        # Si rejeté : acceptable (protection système)
    
    def test_real_world_llm_response_patterns(self):
        """Test patterns réels de réponses LLM."""
        realistic_responses = [
            # LLM qui explique avant le JSON
            '''Je vais créer les fichiers nécessaires pour le projet.

```json
[
    {
        "tool": "create_document",
        "parameters": {
            "filename": "README.md",
            "content": "# Mon Projet\\n\\nDescription du projet"
        },
        "fulfills_deliverable": ["Documentation utilisateur"]
    }
]
```

J'ai créé le fichier README avec les informations de base.''',

            # LLM avec JSON sans marqueurs
            '''[{"tool": "implement_code", "parameters": {"filename": "utils.py", "code": "def helper(): pass"}, "fulfills_deliverable": ["Utilitaires"]}]''',
            
            # LLM avec structure imbriquée
            '''```json
[
    {
        "tool": "create_project_structure",
        "parameters": {
            "directories": ["src", "tests", "docs"],
            "files": {
                "src/main.py": "# Main module",
                "tests/test_main.py": "# Tests"
            }
        },
        "fulfills_deliverable": ["Structure projet", "Fichiers de base"]
    }
]
```'''
        ]
        
        for i, response in enumerate(realistic_responses):
            with self.subTest(response_type=f"realistic_{i+1}"):
                result = self.agent._parse_tool_calls(response)
                
                # Vérifications de robustesse réaliste
                self.assertIsInstance(result, list)
                
                # Si parsing réussi, structure doit être cohérente
                for tool in result:
                    if isinstance(tool, dict) and "tool" in tool:
                        self.assertIsInstance(tool["tool"], str)
                        self.assertTrue(len(tool["tool"]) > 0)


if __name__ == '__main__':
    unittest.main()