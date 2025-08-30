"""
Test unitaire simplifié pour le parsing JSON robuste
Teste les stratégies principales sans référence à fulfills_deliverable
"""

import unittest
from unittest.mock import Mock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent


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


class TestJSONParsingSimplified(unittest.TestCase):
    """Tests simplifiés du parsing JSON robuste."""
    
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
            "title": "Test Document",
            "content": "Contenu du document"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(valid_json)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool"], "create_document")
        self.assertIn("parameters", result[0])
        self.assertEqual(result[0]["parameters"]["title"], "Test Document")
    
    def test_malformed_json_brackets(self):
        """Test parsing JSON avec brackets malformés."""
        malformed_json = '''```json
[
    {
        "tool": "create_document",
        "parameters": {
            "title": "Test",
            "content": "Content"
        }
    }
    // Missing closing bracket
```'''
        
        result = self.agent._parse_tool_calls(malformed_json)
        
        # Doit récupérer au moins un outil même avec JSON malformé
        self.assertGreaterEqual(len(result), 1)
        tool = result[0]
        self.assertEqual(tool["tool"], "create_document")
        self.assertIn("parameters", tool)
    
    def test_json_with_comments(self):
        """Test parsing JSON avec commentaires."""
        json_with_comments = '''```json
[
    {
        "tool": "implement_code", // Outil de création de code
        "parameters": {
            "filename": "test.py",
            "content": "print('Hello')"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(json_with_comments)
        
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["tool"], "implement_code")
    
    def test_incomplete_json_partial_objects(self):
        """Test parsing JSON incomplet avec objets partiels."""
        incomplete_json = '''```json
[
    {
        "tool": "create_document",
        "parameters": {
            "title": "Incomplete"
            // Missing content and proper closing
    },
    {
        "tool": "implement_code",
        "parameters": {
            "filename": "test.py"
        }
    }
```'''
        
        result = self.agent._parse_tool_calls(incomplete_json)
        
        # Doit récupérer au moins quelques objets même incomplets
        self.assertGreater(len(result), 0)
        
        # Vérifier qu'au moins un outil a été récupéré correctement
        tool_names = [tool.get("tool") for tool in result]
        self.assertTrue(any(name in ["create_document", "implement_code"] for name in tool_names))
    
    def test_nested_json_in_parameters(self):
        """Test parsing avec JSON imbriqué dans les paramètres."""
        nested_json = '''```json
[
    {
        "tool": "create_project_file",
        "parameters": {
            "filename": "config.json",
            "content": "{\\"database\\": {\\"host\\": \\"localhost\\", \\"port\\": 5432}}"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(nested_json)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool"], "create_project_file")
        self.assertIn("content", result[0]["parameters"])
    
    def test_multiple_tools(self):
        """Test parsing avec plusieurs outils."""
        multiple_tools = '''```json
[
    {
        "tool": "create_document",
        "parameters": {
            "title": "Doc1",
            "content": "Content1"
        }
    },
    {
        "tool": "implement_code",
        "parameters": {
            "filename": "code1.py",
            "content": "print('code1')"
        }
    },
    {
        "tool": "create_project_file", 
        "parameters": {
            "filename": "config.yaml",
            "content": "debug: true"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(multiple_tools)
        
        self.assertEqual(len(result), 3)
        tools = [tool["tool"] for tool in result]
        self.assertIn("create_document", tools)
        self.assertIn("implement_code", tools)
        self.assertIn("create_project_file", tools)
    
    def test_empty_and_null_responses(self):
        """Test parsing de réponses vides ou nulles."""
        test_inputs = ["", "```json\n\n```", "```json\nnull\n```", "```json\n[]\n```"]
        
        for test_input in test_inputs:
            with self.subTest(input=test_input):
                result = self.agent._parse_tool_calls(test_input)
                self.assertIsInstance(result, list)  # Doit retourner une liste même vide
    
    def test_special_characters_in_content(self):
        """Test parsing avec caractères spéciaux dans le contenu."""
        special_chars_json = '''```json
[
    {
        "tool": "create_document",
        "parameters": {
            "title": "Caractères spéciaux: àéèêç",
            "content": "Contenu avec \\"guillemets\\" et \\nnouveaux lignes\\net \\ttabs"
        }
    }
]
```'''
        
        result = self.agent._parse_tool_calls(special_chars_json)
        
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["tool"], "create_document")
        self.assertIn("Caractères spéciaux", result[0]["parameters"]["title"])


if __name__ == '__main__':
    # Lancer les tests
    unittest.main(verbosity=2)