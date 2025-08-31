"""
Test unitaire pour le système d'auto-évaluation des agents - Priorité: CRITIQUE

Teste la logique d'auto-évaluation des agents, leur rapport au superviseur, 
et les décisions du superviseur qui en découlent.

Couvre:
- LightweightLLMService.self_evaluate_mission()
- BaseAgent._generate_structured_report()
- tool_report_to_supervisor()
- Supervisor._verify_milestone_completion()
- Supervisor._apply_verification_decision()
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, ToolResult
from agents.supervisor import Supervisor, _UnifiedMilestoneManager
from core.lightweight_llm_service import LightweightLLMService, get_lightweight_llm_service
from tools.base_tools import tool_report_to_supervisor


class MockAgent(BaseAgent):
    """Agent de test qui hérite de BaseAgent."""
    
    def __init__(self, name="TestAgent", project_name="TestProject"):
        # Mock des dépendances minimales
        self.logger = Mock()
        self.name = name
        self.project_name = project_name
        self.supervisor = None
        self.rag_engine = None
        self.state = {'current_task_id': 'test_task_1'}
        self.current_milestone_id = 'milestone_1'
        
        # Mock du lightweight service 
        self.lightweight_service = Mock()
        
        # Tools et configuration
        self.tools = {}
        self.tool_definitions = {}
        
        # Conversation memory mock
        self.conversation_memory = []
        
    def communicate(self, message, recipient=None):
        return "Test communication response"


class TestAgentSelfEvaluationSystem(unittest.TestCase):
    """Tests complets du système d'auto-évaluation des agents."""
    
    def setUp(self):
        """Setup avant chaque test."""
        # Créer un répertoire temporaire pour les tests
        self.test_dir = tempfile.mkdtemp()
        self.project_name = "TestSelfEvaluationProject"
        self.project_path = Path(self.test_dir) / "projects" / self.project_name
        
        # Changer le répertoire de travail temporairement
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Créer la structure de base
        self.project_path.mkdir(parents=True, exist_ok=True)
        (self.project_path / "docs").mkdir(exist_ok=True)
        
        # Créer un mock agent
        self.mock_agent = MockAgent(project_name=self.project_name)
        
        # Mock du superviseur
        self.mock_supervisor = Mock()
        self.mock_agent.supervisor = self.mock_supervisor
        
    def tearDown(self):
        """Nettoyage après chaque test."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    # ========== TESTS LIGHTWEIGHTLLMSERVICE SELF-EVALUATION ==========
    
    @patch('core.llm_connector.LLMFactory.create')
    def test_self_evaluate_mission_compliant(self, mock_llm_factory):
        """Test évaluation compliant - succès complet."""
        # Mock du LLM qui retourne une évaluation positive
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"assessment": "compliant", "reason": "Tous les objectifs atteints", "confidence": 0.95}'
        mock_llm_factory.return_value = mock_llm
        
        # Mock de la config pour activer l'évaluation
        with patch('core.lightweight_llm_service.default_config', {
            'rag': {
                'auto_context_injection': {
                    'keyword_extraction': {'enabled': True, 'model': 'test-model'}
                }
            }
        }):
            service = LightweightLLMService(self.project_name)
            
            result = service.self_evaluate_mission(
                objective="Créer un fichier de configuration",
                artifacts=["config.yaml", "settings.json"],
                issues=[]
            )
            
            # Vérifications
            self.assertEqual(result['assessment'], 'compliant')
            self.assertEqual(result['reason'], 'Tous les objectifs atteints')
            self.assertEqual(result['confidence'], 0.95)
            mock_llm.generate.assert_called_once()
    
    @patch('core.llm_connector.LLMFactory.create')
    def test_self_evaluate_mission_partial(self, mock_llm_factory):
        """Test évaluation partielle - succès partiel avec problèmes."""
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"assessment": "partial", "reason": "Certains livrables manquants", "confidence": 0.7}'
        mock_llm_factory.return_value = mock_llm
        
        with patch('core.lightweight_llm_service.default_config', {
            'rag': {
                'auto_context_injection': {
                    'keyword_extraction': {'enabled': True, 'model': 'test-model'}
                }
            }
        }):
            service = LightweightLLMService(self.project_name)
            
            result = service.self_evaluate_mission(
                objective="Implémenter 3 fonctions principales",
                artifacts=["function1.py"],  # Seulement 1 sur 3
                issues=["Erreur lors de l'implémentation function2", "function3 non terminée"]
            )
            
            # Vérifications
            self.assertEqual(result['assessment'], 'partial')
            self.assertIn('manquants', result['reason'])
            self.assertEqual(result['confidence'], 0.7)
    
    @patch('core.llm_connector.LLMFactory.create')
    def test_self_evaluate_mission_failed(self, mock_llm_factory):
        """Test évaluation échec - échec critique."""
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"assessment": "failed", "reason": "Aucun livrable créé, erreurs majeures", "confidence": 0.9}'
        mock_llm_factory.return_value = mock_llm
        
        with patch('core.lightweight_llm_service.default_config', {
            'rag': {
                'auto_context_injection': {
                    'keyword_extraction': {'enabled': True, 'model': 'test-model'}
                }
            }
        }):
            service = LightweightLLMService(self.project_name)
            
            result = service.self_evaluate_mission(
                objective="Créer une base de données complète",
                artifacts=[],  # Aucun artefact
                issues=["Connexion DB échouée", "Permissions insuffisantes", "Configuration invalide"]
            )
            
            # Vérifications
            self.assertEqual(result['assessment'], 'failed')
            self.assertIn('Aucun livrable', result['reason'])
            self.assertEqual(result['confidence'], 0.9)
    
    def test_self_evaluate_mission_disabled(self):
        """Test évaluation avec service désactivé."""
        with patch('core.lightweight_llm_service.default_config', {
            'rag': {
                'auto_context_injection': {
                    'keyword_extraction': {'enabled': False}
                }
            }
        }):
            service = LightweightLLMService(self.project_name)
            
            result = service.self_evaluate_mission(
                objective="Test objectif",
                artifacts=["test.py"],
                issues=[]
            )
            
            # Vérifications du fallback
            self.assertEqual(result['assessment'], 'partial')
            self.assertIn('désactivée', result['reason'])
            self.assertEqual(result['confidence'], 0.5)
    
    @patch('core.llm_connector.LLMFactory.create')
    def test_self_evaluate_mission_json_parse_error(self, mock_llm_factory):
        """Test gestion d'erreur de parsing JSON."""
        mock_llm = Mock()
        mock_llm.generate.return_value = 'Invalid JSON response from LLM'
        mock_llm_factory.return_value = mock_llm
        
        # Mock du parser JSON pour simuler un échec
        with patch('core.lightweight_llm_service.default_config', {
            'rag': {
                'auto_context_injection': {
                    'keyword_extraction': {'enabled': True, 'model': 'test-model'}
                }
            }
        }):
            service = LightweightLLMService(self.project_name)
            
            result = service.self_evaluate_mission(
                objective="Test objectif",
                artifacts=["test.py"],
                issues=[]
            )
            
            # Vérifications du fallback en cas d'erreur
            self.assertEqual(result['assessment'], 'failed')
            self.assertIn('Erreur interne', result['reason'])
            self.assertEqual(result['confidence'], 0.1)
    
    # ========== TESTS STRUCTURED REPORT GENERATION ==========
    
    def test_generate_structured_report_compliant(self):
        """Test génération rapport structuré - cas compliant."""
        # Mock du lightweight service pour retourner une évaluation positive
        mock_evaluation = {
            'assessment': 'compliant',
            'reason': 'Mission accomplie avec succès',
            'confidence': 0.9
        }
        self.mock_agent.lightweight_service.self_evaluate_mission.return_value = mock_evaluation
        
        # Données d'entrée
        plan = {
            'analysis': 'Créer un module de calcul mathématique',
            'milestone_id': 'milestone_1'
        }
        result = {
            'artifacts': ['calculator.py', 'tests.py'],
            'tools_executed': [
                {'status': 'success', 'result': {'status': 'success'}},
                {'status': 'success', 'result': {'status': 'success'}}
            ]
        }
        
        # Exécuter la méthode
        structured_report = self.mock_agent._generate_structured_report(plan, result)
        
        # Vérifications
        self.assertIsInstance(structured_report, dict)
        self.assertEqual(structured_report['self_assessment'], 'compliant')
        self.assertEqual(structured_report['artifacts_created'], ['calculator.py', 'tests.py'])
        self.assertEqual(structured_report['confidence_level'], 0.9)
        self.assertEqual(structured_report['agent_name'], 'TestAgent')
        self.assertIn('completion_timestamp', structured_report)
        
        # Vérifier que le service léger a été appelé
        self.mock_agent.lightweight_service.self_evaluate_mission.assert_called_once()
    
    def test_generate_structured_report_partial(self):
        """Test génération rapport structuré - cas partiel."""
        mock_evaluation = {
            'assessment': 'partial',
            'reason': 'Quelques problèmes rencontrés mais majorité du travail accompli',
            'confidence': 0.6
        }
        self.mock_agent.lightweight_service.self_evaluate_mission.return_value = mock_evaluation
        
        plan = {'analysis': 'Implémenter API REST complète', 'milestone_id': 'milestone_2'}
        result = {
            'artifacts': ['api.py'],
            'tools_executed': [
                {'status': 'success', 'result': {'status': 'success'}},
                {'status': 'error', 'result': {'error': 'Timeout lors de la création de tests'}}
            ]
        }
        
        structured_report = self.mock_agent._generate_structured_report(plan, result)
        
        # Vérifications
        self.assertEqual(structured_report['self_assessment'], 'partial')
        self.assertIn('Timeout lors de la création de tests', structured_report['issues_encountered'])
        self.assertEqual(structured_report['confidence_level'], 0.6)
    
    def test_generate_structured_report_error_handling(self):
        """Test gestion d'erreur lors de la génération du rapport."""
        # Mock qui lève une exception
        self.mock_agent.lightweight_service.self_evaluate_mission.side_effect = Exception("Service indisponible")
        
        plan = {'analysis': 'Test analysis', 'milestone_id': 'milestone_error'}
        result = {'artifacts': [], 'tools_executed': []}
        
        structured_report = self.mock_agent._generate_structured_report(plan, result)
        
        # Vérifications du fallback
        self.assertEqual(structured_report['self_assessment'], 'failed')
        self.assertIn('Erreur critique', structured_report['assessment_reason'])
        self.assertEqual(structured_report['confidence_level'], 0.1)
        self.assertEqual(structured_report['agent_name'], 'TestAgent')
    
    # ========== TESTS TOOL REPORT TO SUPERVISOR ==========
    
    def test_tool_report_to_supervisor_automatic_format(self):
        """Test envoi de rapport automatique au superviseur."""
        # Préparer le contenu du rapport structuré
        structured_content = {
            'self_assessment': 'compliant',
            'artifacts_created': ['main.py', 'config.yaml'],
            'confidence_level': 0.85,
            'agent_name': 'TestAgent',
            'issues_encountered': []
        }
        
        parameters = {
            'report_type': 'completion',
            'content': structured_content
        }
        
        # Exécuter l'outil
        result = tool_report_to_supervisor(self.mock_agent, parameters)
        
        # Vérifications
        self.assertEqual(result.status, 'success')
        self.assertTrue(result.result['report_sent'])
        
        # Vérifier que le superviseur a reçu le rapport
        self.mock_supervisor.receive_report.assert_called_once()
        
        # Vérifier la structure du rapport envoyé
        call_args = self.mock_supervisor.receive_report.call_args
        self.assertEqual(call_args[0][0], 'TestAgent')  # agent_name
        
        sent_report = call_args[0][1]  # report
        self.assertEqual(sent_report['type'], 'completion')
        self.assertEqual(sent_report['agent'], 'TestAgent')
        self.assertEqual(sent_report['content']['type'], 'automatic')
        self.assertEqual(sent_report['content']['self_assessment'], 'compliant')
    
    def test_tool_report_to_supervisor_manual_format(self):
        """Test envoi de rapport manuel au superviseur."""
        parameters = {
            'report_type': 'progress',
            'content': 'Travail en cours - 50% des fonctions implémentées avec succès'
        }
        
        result = tool_report_to_supervisor(self.mock_agent, parameters)
        
        # Vérifications
        self.assertEqual(result.status, 'success')
        
        # Vérifier la normalisation du contenu manuel
        call_args = self.mock_supervisor.receive_report.call_args
        sent_report = call_args[0][1]
        
        self.assertEqual(sent_report['content']['type'], 'manual')
        self.assertEqual(sent_report['content']['message'], 'Travail en cours - 50% des fonctions implémentées avec succès')
        # L'analyse sémantique devrait détecter le succès
        self.assertEqual(sent_report['content']['self_assessment'], 'compliant')
    
    def test_tool_report_to_supervisor_no_supervisor(self):
        """Test rapport sans superviseur assigné."""
        self.mock_agent.supervisor = None
        
        parameters = {
            'report_type': 'issue',
            'content': 'Problème détecté'
        }
        
        result = tool_report_to_supervisor(self.mock_agent, parameters)
        
        # Vérifications de l'erreur
        self.assertEqual(result.status, 'error')
        self.assertIn('Pas de superviseur', result.error)
    
    # ========== TESTS SUPERVISOR DECISION-MAKING ==========
    
    def test_supervisor_verify_milestone_completion_fast_approval(self):
        """Test vérification rapide de jalon - tous agents conformes."""
        # Créer un superviseur de test
        with patch('agents.supervisor.default_config', {
            'agents': {'supervisor': {'role': 'test', 'personality': 'test', 'model': 'test-model'}},
            'llm': {'default_model': 'test-model'}
        }):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value="# Test Charter"):
                    supervisor = Supervisor(
                        project_name=self.project_name,
                        project_prompt="Test project",
                        rag_engine=None
                    )
        
        # Données de test - jalon avec résultats positifs
        milestone = {
            'id': 1,
            'name': 'Test Milestone',
            'description': 'Test milestone',
            'deliverables': ['doc.md', 'code.py']
        }
        
        milestone_result = {
            'milestone_id': 1,
            'tasks_completed': [
                {
                    'agent': 'analyst',
                    'result': {
                        'structured_report': {
                            'self_assessment': 'compliant',
                            'confidence_level': 0.9,
                            'artifacts_created': ['doc.md'],
                            'agent_name': 'analyst'
                        }
                    }
                },
                {
                    'agent': 'developer', 
                    'result': {
                        'structured_report': {
                            'self_assessment': 'compliant',
                            'confidence_level': 0.85,
                            'artifacts_created': ['code.py'],
                            'agent_name': 'developer'
                        }
                    }
                }
            ]
        }
        
        # Exécuter la vérification
        verification = supervisor._verify_milestone_completion(milestone, milestone_result)
        
        # Vérifications - doit être approuvé rapidement
        self.assertEqual(verification['decision'], 'approve')
        self.assertIn('conformité complète', verification['reason'])
        self.assertEqual(verification['confidence'], 0.9)
        self.assertEqual(verification['evaluation_type'], 'fast')
    
    def test_supervisor_verify_milestone_completion_deep_evaluation_needed(self):
        """Test vérification approfondie nécessaire - résultats mixtes."""
        with patch('agents.supervisor.default_config', {
            'agents': {'supervisor': {'role': 'test', 'personality': 'test', 'model': 'test-model'}},
            'llm': {'default_model': 'test-model'}
        }):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value="# Test Charter"):
                    supervisor = Supervisor(
                        project_name=self.project_name,
                        project_prompt="Test project", 
                        rag_engine=None
                    )
        
        milestone = {
            'id': 1,
            'name': 'Complex Milestone',
            'description': 'Complex milestone needing evaluation',
            'deliverables': ['spec.md', 'impl.py', 'tests.py']
        }
        
        milestone_result = {
            'milestone_id': 1,
            'tasks_completed': [
                {
                    'agent': 'analyst',
                    'result': {
                        'structured_report': {
                            'self_assessment': 'compliant',
                            'confidence_level': 0.8,
                            'artifacts_created': ['spec.md'],
                            'agent_name': 'analyst'
                        }
                    }
                },
                {
                    'agent': 'developer',
                    'result': {
                        'structured_report': {
                            'self_assessment': 'failed',  # Échec critique
                            'confidence_level': 0.9,
                            'artifacts_created': [],  # Pas d'artefacts
                            'issues_encountered': ['Compilation échouée', 'Tests non exécutables'],
                            'agent_name': 'developer'
                        }
                    }
                }
            ]
        }
        
        # Mock de l'évaluation approfondie qui va être appelée
        with patch.object(supervisor, '_deep_milestone_evaluation') as mock_deep_eval:
            mock_deep_eval.return_value = {
                'decision': 'request_rework',
                'reason': 'Échec critique du développeur compromise le jalon',
                'confidence': 0.85,
                'evaluation_type': 'deep_ai'
            }
            
            verification = supervisor._verify_milestone_completion(milestone, milestone_result)
            
            # Vérifications
            mock_deep_eval.assert_called_once()
            self.assertEqual(verification['decision'], 'request_rework')
            self.assertEqual(verification['evaluation_type'], 'deep_ai')
            self.assertIn('critique du développeur', verification['reason'])
    
    def test_supervisor_deep_milestone_evaluation(self):
        """Test évaluation approfondie avec LLM."""
        with patch('agents.supervisor.default_config', {
            'agents': {'supervisor': {'role': 'test', 'personality': 'test', 'model': 'test-model'}},
            'llm': {'default_model': 'test-model'}
        }):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value="# Test Project Charter\nObjectif: Créer système complet"):
                    supervisor = Supervisor(
                        project_name=self.project_name,
                        project_prompt="Test project",
                        rag_engine=None
                    )
        
        # Mock de la génération JSON
        mock_evaluation_response = {
            'decision': 'request_rework',
            'success_rate': 60,
            'reason': 'Taux de réussite 60% insuffisant (seuil: 90%)',
            'confidence': 0.8,
            'agents_analysis': {
                'analyst': 'success - documentation complète',
                'developer': 'failure - aucun code fonctionnel produit'
            }
        }
        
        milestone = {
            'id': 1,
            'name': 'Implementation Milestone',
            'description': 'Implement core functionality',
            'deliverables': ['docs/', 'src/', 'tests/']
        }
        
        milestone_result = {
            'milestone_id': 1,
            'agents_involved': ['analyst', 'developer'],
            'tasks_completed': []
        }
        
        structured_reports = [
            {
                'agent_name': 'analyst',
                'self_assessment': 'compliant',
                'confidence_level': 0.9,
                'artifacts_created': ['architecture.md', 'specs.md'],
                'issues_encountered': []
            },
            {
                'agent_name': 'developer',
                'self_assessment': 'failed',
                'confidence_level': 0.95,
                'artifacts_created': [],
                'issues_encountered': ['Build system configuration failed', 'No executable code produced']
            }
        ]
        
        # Mock de _get_project_charter_from_file aussi
        with patch.object(supervisor, '_get_project_charter_from_file', return_value="# Test Project Charter\nObjectif: Créer système complet"):
            with patch.object(supervisor, 'generate_json_with_context', return_value=mock_evaluation_response) as mock_generate:
                verification = supervisor._deep_milestone_evaluation(milestone, milestone_result, structured_reports)
                
                # Vérifications
                self.assertEqual(verification['decision'], 'request_rework')
                self.assertIn('60%', verification['reason'])
                self.assertEqual(verification['confidence'], 0.8)
                self.assertEqual(verification['evaluation_type'], 'deep_ai')
                
                # Vérifier que la méthode a été appelée
                mock_generate.assert_called_once()
    
    def test_supervisor_apply_verification_decision_approve(self):
        """Test application de décision d'approbation."""
        with patch('agents.supervisor.default_config', {
            'agents': {'supervisor': {'role': 'test', 'personality': 'test', 'model': 'test-model', 'max_global_corrections': 5}},
            'llm': {'default_model': 'test-model'}
        }):
            supervisor = Supervisor(
                project_name=self.project_name,
                project_prompt="Test project",
                rag_engine=None
            )
        
        # Simuler un jalon en cours
        test_milestone = {
            'id': 1,
            'name': 'Test Milestone',
            'status': 'in_progress'
        }
        supervisor._milestone_manager.milestones = [test_milestone]
        supervisor._milestone_manager.current_index = 0
        
        verification = {
            'decision': 'approve',
            'reason': 'Tous les critères respectés',
            'confidence': 0.9,
            'evaluation_type': 'fast'
        }
        
        with patch.object(supervisor, '_create_journal_entry') as mock_journal:
            supervisor._apply_verification_decision(verification, test_milestone)
            
            # Vérifications
            self.assertEqual(test_milestone['verification_status'], 'approve')
            self.assertEqual(supervisor._milestone_manager.current_index, 1)  # Avancé
            self.assertEqual(supervisor.project_state['milestones_completed'], 1)
            mock_journal.assert_called_once()
    
    def test_supervisor_apply_verification_decision_rework_with_human_approval(self):
        """Test demande de rework avec approbation humaine."""
        with patch('agents.supervisor.default_config', {
            'agents': {'supervisor': {'role': 'test', 'personality': 'test', 'model': 'test-model', 'max_global_corrections': 5}},
            'llm': {'default_model': 'test-model'}
        }):
            supervisor = Supervisor(
                project_name=self.project_name,
                project_prompt="Test project",
                rag_engine=None
            )
        
        test_milestone = {
            'id': 1,
            'name': 'Test Milestone',
            'status': 'in_progress',
            'agents_required': ['analyst']
        }
        supervisor._milestone_manager.milestones = [test_milestone]
        supervisor._milestone_manager.current_index = 0
        supervisor.project_state['total_corrections'] = 0
        
        verification = {
            'decision': 'request_rework',
            'reason': 'Qualité insuffisante détectée',
            'confidence': 0.8
        }
        
        # Mock de la validation humaine qui approuve
        with patch.object(supervisor, '_request_human_validation', return_value={"action": "approve_recommendation", "instruction": ""}):
            with patch.object(supervisor, '_milestone_manager') as mock_manager:
                mock_manager.insert_correction_after_current.return_value = {'id': 2, 'name': 'Correction'}
                mock_manager.current_index = 0
                
                supervisor._apply_verification_decision(verification, test_milestone)
                
                # Vérifications
                self.assertEqual(supervisor.project_state['total_corrections'], 1)
                mock_manager.insert_correction_after_current.assert_called_once()


if __name__ == '__main__':
    unittest.main()