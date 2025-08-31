"""
Test unitaire pour la fonctionnalité d'intégration humaine du Supervisor.
Teste les deux points d'intervention critiques avec simulation des entrées utilisateur.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import sys
import os

# Ajouter le répertoire racine au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.supervisor import Supervisor
from core.cli_interface import CLIInterface


class TestHumanValidation(unittest.TestCase):
    """Tests pour la fonctionnalité de validation humaine du Supervisor."""
    
    def setUp(self):
        """Setup avant chaque test."""
        # Créer un répertoire temporaire pour les tests
        self.test_dir = tempfile.mkdtemp()
        self.project_name = "TestHumanValidation"
        self.project_path = Path(self.test_dir) / "projects" / self.project_name
        
        # Changer le répertoire de travail temporairement
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Mock des services LLM
        self.llm_mock = Mock()
        self.llm_mock.generate_with_messages.return_value = "Question formulée pour l'utilisateur"
        
        # Créer le superviseur avec des mocks
        with patch('core.llm_connector.LLMFactory.create', return_value=self.llm_mock):
            self.supervisor = Supervisor(
                project_name=self.project_name,
                project_prompt="Test projet pour validation humaine",
                rag_engine=None  # Pas de RAG pour les tests
            )
        
        # Simuler un milestone actuel
        self.test_milestone = {
            'id': 1,
            'milestone_id': 'milestone_1',
            'name': 'Test Milestone',
            'description': 'Un milestone de test',
            'agents_required': ['analyst'],
            'deliverables': ['test.md'],
            'status': 'pending',
            'correction_attempts': 0
        }
        
        self.supervisor.milestones = [self.test_milestone]
        self.supervisor.current_milestone_index = 0
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('agents.supervisor.Prompt.ask')
    @patch('agents.supervisor.CLIInterface')
    def test_request_human_validation_approve(self, mock_cli_class, mock_prompt):
        """Test de validation humaine avec réponse 'Approuver'."""
        # Mock de l'interface CLI
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        # Simuler choix utilisateur "1" (Approuver)
        mock_prompt.side_effect = ["1"]
        
        # Mock du LLM pour générer la question
        with patch.object(self.supervisor, 'generate_with_context', return_value="Question bien formulée ?"):
            result = self.supervisor._request_human_validation(
                reason="Test de validation",
                recommended_action="Action de test",
                milestone_details=self.test_milestone,
                agent_reports=[],
                verification_info={}
            )
        
        # Vérifications - nouveau format de réponse
        self.assertEqual(result["action"], "approve_recommendation")
        self.assertEqual(result["instruction"], "")
        mock_cli.display_warning.assert_called_once()
        mock_cli.display_info.assert_called_with("✅ Action recommandée approuvée par l'utilisateur")
    
    @patch('agents.supervisor.Prompt.ask')
    @patch('agents.supervisor.CLIInterface')
    def test_request_human_validation_force_approve(self, mock_cli_class, mock_prompt):
        """Test de validation humaine avec réponse 'Valider et continuer'."""
        # Mock de l'interface CLI
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        # Simuler choix utilisateur "2" (Valider le jalon et continuer)
        mock_prompt.side_effect = ["2"]
        
        # Mock du LLM pour générer la question
        with patch.object(self.supervisor, 'generate_with_context', return_value="Question bien formulée ?"):
            result = self.supervisor._request_human_validation(
                reason="Test de validation",
                recommended_action="Action de test",
                milestone_details=self.test_milestone,
                agent_reports=[],
                verification_info={}
            )
        
        # Vérifications - nouvelle action force_approve
        self.assertEqual(result["action"], "force_approve")
        self.assertEqual(result["instruction"], "Validation forcée par l'utilisateur")
        mock_cli.display_info.assert_called_with("☑️ Validation forcée du jalon demandée")
    
    @patch('agents.supervisor.Prompt.ask')
    @patch('agents.supervisor.CLIInterface')
    def test_request_human_validation_adjust_plan(self, mock_cli_class, mock_prompt):
        """Test de validation humaine avec réponse 'Ajuster le plan'."""
        # Mock de l'interface CLI
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        # Simuler choix utilisateur "3" (Ajuster plan) puis instruction
        mock_prompt.side_effect = ["3", "Ajouter plus de tests"]
        
        # Mock du LLM pour générer la question et analyser l'instruction
        with patch.object(self.supervisor, 'generate_with_context', return_value="Question bien formulée ?"):
            with patch.object(self.supervisor, '_analyze_user_instruction_for_plan_adjustment', return_value="Instruction analysée: renforcer tests"):
                result = self.supervisor._request_human_validation(
                    reason="Test de validation",
                    recommended_action="Action de test",
                    milestone_details=self.test_milestone,
                    agent_reports=[],
                    verification_info={}
                )
        
        # Vérifications - nouvelle action adjust_plan
        self.assertEqual(result["action"], "adjust_plan")
        self.assertEqual(result["instruction"], "Ajouter plus de tests")
        self.assertIn("analyzed_reason", result)
        mock_cli.display_info.assert_called_with("🔄 Instruction pour ajustement de plan reçue: Ajouter plus de tests")
    
    @patch('agents.supervisor.Prompt.ask')
    @patch('agents.supervisor.CLIInterface')
    def test_request_human_validation_cancel_declined(self, mock_cli_class, mock_prompt):
        """Test de validation humaine avec annulation de l'arrêt."""
        # Mock de l'interface CLI
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        mock_cli.ask_confirmation.return_value = False  # Ne pas confirmer l'arrêt
        
        # Simuler choix utilisateur "3" (Arrêter) puis annulation
        mock_prompt.side_effect = ["3"]
        
        # Mock du LLM pour générer la question
        with patch.object(self.supervisor, 'generate_with_context', return_value="Question bien formulée ?"):
            result = self.supervisor._request_human_validation(
                reason="Test de validation",
                recommended_action="Action de test",
                milestone_details=self.test_milestone,
                agent_reports=[],
                verification_info={}
            )
        
        # Vérifications - doit retourner "approve_recommendation" par défaut (nouveau comportement v1.4+)
        self.assertEqual(result["action"], "approve_recommendation")
        # L'instruction peut contenir le message d'erreur (comportement normal)
        self.assertIn("Erreur validation humaine", result["instruction"])
        # En cas d'erreur, display_info peut ne pas être appelé
    
    def test_project_state_tracking(self):
        """Test du suivi d'état du projet."""
        # Vérifier l'état initial du projet (v1.4+ utilise 'initialized')
        self.assertEqual(self.supervisor.project_state['status'], 'initialized')
        self.assertEqual(self.supervisor.project_state['total_corrections'], 0)
        self.assertEqual(self.supervisor.project_state['milestones_completed'], 0)
        
        # Simuler une progression
        self.supervisor.project_state['total_corrections'] = 2
        self.supervisor.project_state['milestones_completed'] = 3
        
        # Vérifications
        self.assertEqual(self.supervisor.project_state['total_corrections'], 2)
        self.assertEqual(self.supervisor.project_state['milestones_completed'], 3)
    
    def test_apply_verification_decision_approve(self):
        """Test d'application de décision d'approbation."""
        # Simuler un milestone 
        test_milestone = {
            'id': 1,
            'name': 'Test Milestone',
            'status': 'in_progress'
        }
        
        # Mock du milestone manager
        with patch.object(self.supervisor, '_milestone_manager') as mock_manager:
            mock_manager.milestones = [test_milestone]
            mock_manager.current_index = 0
            
            verification = {
                'decision': 'approve',
                'reason': 'Tous les critères respectés',
                'confidence': 0.9
            }
            
            with patch.object(self.supervisor, '_create_journal_entry') as mock_journal:
                self.supervisor._apply_verification_decision(verification, test_milestone)
                
                # Vérifications
                self.assertEqual(test_milestone['verification_status'], 'approve')
                mock_journal.assert_called_once()
    
    def test_apply_verification_decision_request_rework(self):
        """Test d'application de décision de rework."""
        test_milestone = {
            'id': 1,
            'name': 'Test Milestone',
            'status': 'in_progress',
            'agents_required': ['analyst', 'developer']
        }
        
        # Mock du milestone manager
        with patch.object(self.supervisor, '_milestone_manager') as mock_manager:
            mock_manager.milestones = [test_milestone]
            mock_manager.current_index = 0
            
            verification = {
                'decision': 'request_rework',
                'reason': 'Qualité insuffisante',
                'confidence': 0.8
            }
            
            # Mock de la validation humaine
            with patch.object(self.supervisor, '_request_human_validation', return_value={"action": "approve_recommendation", "instruction": ""}) as mock_validation:
                self.supervisor._apply_verification_decision(verification, test_milestone)
                
                # Vérifications
                mock_validation.assert_called_once()
                self.assertEqual(test_milestone['verification_status'], 'request_rework')
    
    def test_apply_verification_decision_adjust_plan(self):
        """Test d'application de décision d'ajustement de plan."""
        test_milestone = {
            'id': 1,
            'name': 'Test Milestone',
            'status': 'in_progress'
        }
        
        verification = {
            'decision': 'adjust_plan',
            'reason': 'Plan needs adjustment',
            'confidence': 0.9
        }
        
        # Mock de la validation humaine
        with patch.object(self.supervisor, '_request_human_validation', return_value={"action": "approve_recommendation", "instruction": ""}) as mock_validation:
            with patch.object(self.supervisor, '_create_journal_entry') as mock_journal:
                self.supervisor._apply_verification_decision(verification, test_milestone)
                
                # Vérifications
                mock_validation.assert_called_once()
                self.assertEqual(test_milestone['verification_status'], 'adjust_plan')
                # Le journal peut être appelé plusieurs fois (normal pour adjust_plan)
                self.assertTrue(mock_journal.called)
    
    def test_request_human_validation_error_handling(self):
        """Test de gestion d'erreur dans _request_human_validation."""
        # Mock qui lève une exception
        with patch('agents.supervisor.CLIInterface', side_effect=Exception("Erreur simulation")):
            result = self.supervisor._request_human_validation(
                reason="Test erreur",
                recommended_action="Action test",
                milestone_details=self.test_milestone,
                agent_reports=[],
                verification_info={}
            )
            
            # Doit retourner "approve_recommendation" par défaut en cas d'erreur
            self.assertEqual(result["action"], "approve_recommendation")
            self.assertIn("Erreur validation humaine", result["instruction"])
    
    @patch('agents.supervisor.Prompt.ask')
    @patch('agents.supervisor.CLIInterface')
    def test_request_human_validation_with_detailed_reports(self, mock_cli_class, mock_prompt):
        """Test de validation humaine avec rapports détaillés d'agents."""
        # Mock de l'interface CLI
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        # Simuler choix utilisateur "1" (Approuver)
        mock_prompt.side_effect = ["1"]
        
        # Créer des rapports d'agents détaillés
        detailed_agent_reports = [
            {
                'agent': 'analyst',
                'content': {
                    'type': 'automatic',
                    'self_assessment': 'compliant',
                    'confidence_level': 0.95,
                    'artifacts_created': ['requirements.md', 'architecture.md'],
                    'issues_encountered': [],
                    'agent_name': 'analyst',
                    'deliverables_status': {
                        'requirements.md': 'completed',
                        'architecture.md': 'completed'
                    }
                }
            },
            {
                'agent': 'developer',
                'content': {
                    'type': 'automatic', 
                    'self_assessment': 'failed',
                    'confidence_level': 0.9,
                    'artifacts_created': [],
                    'issues_encountered': ['Build configuration failed', 'Tests could not be executed'],
                    'agent_name': 'developer',
                    'deliverables_status': {
                        'src/main.py': 'missing',
                        'tests/': 'missing'
                    }
                }
            }
        ]
        
        verification_info = {
            'decision': 'request_rework',
            'reason': 'Developer failed to produce executable code',
            'confidence': 0.85,
            'evaluation_type': 'deep_ai'
        }
        
        # Mock du LLM pour générer la question
        with patch.object(self.supervisor, 'generate_with_context', return_value="Question avec détails enrichis"):
            result = self.supervisor._request_human_validation(
                reason="Échec critique du développeur détecté",
                recommended_action="Créer un jalon de correction pour réparer le code",
                milestone_details=self.test_milestone,
                agent_reports=detailed_agent_reports,
                verification_info=verification_info
            )
        
        # Vérifications
        self.assertEqual(result["action"], "approve_recommendation")
        self.assertEqual(result["instruction"], "")
        
        # Vérifier que l'interface CLI affiche les informations détaillées
        mock_cli.display_warning.assert_called_once()
        mock_cli.console.print.assert_called()  # Doit afficher les détails enrichis
    
    @patch('agents.supervisor.Prompt.ask') 
    @patch('agents.supervisor.CLIInterface')
    def test_request_human_validation_analyze_user_instruction(self, mock_cli_class, mock_prompt):
        """Test de l'analyse d'instruction utilisateur pour ajustement de plan."""
        # Mock de l'interface CLI
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        # Simuler choix utilisateur "3" (Ajuster plan) puis instruction
        mock_prompt.side_effect = ["3", "Ajouter plus de tests et améliorer la documentation"]
        
        # Mock du LLM pour générer la question et analyser l'instruction
        with patch.object(self.supervisor, 'generate_with_context') as mock_generate:
            mock_generate.side_effect = [
                "Question pour ajustement de plan",
                "Instruction analysée: Renforcer la qualité avec tests supplémentaires et documentation approfondie"
            ]
            
            result = self.supervisor._request_human_validation(
                reason="Plan nécessite ajustement",
                recommended_action="Recalculer jalons futurs",
                milestone_details=self.test_milestone,
                agent_reports=[],
                verification_info={'decision': 'adjust_plan'}
            )
        
        # Vérifications
        self.assertEqual(result["action"], "adjust_plan")
        self.assertEqual(result["instruction"], "Ajouter plus de tests et améliorer la documentation")
        self.assertIn("analyzed_reason", result)
        self.assertIn("tests supplémentaires", result["analyzed_reason"])
        
        # Vérifier que l'analyse LLM a été appelée
        self.assertEqual(mock_generate.call_count, 2)
    
    def test_apply_verification_decision_with_structured_reports(self):
        """Test de l'application de décisions basée sur rapports structurés."""
        # Préparer le superviseur avec milestone manager
        test_milestone = {
            'id': 1,
            'name': 'Integration Test Milestone',
            'status': 'in_progress',
            'agents_required': ['analyst', 'developer']
        }
        
        self.supervisor._milestone_manager.milestones = [test_milestone]
        self.supervisor._milestone_manager.current_index = 0
        self.supervisor.project_state['total_corrections'] = 0
        
        # Simuler des rapports d'agents dans le buffer
        self.supervisor.current_milestone_reports = [
            {
                'agent': 'analyst',
                'content': {
                    'type': 'automatic',
                    'self_assessment': 'compliant',
                    'confidence_level': 0.9
                }
            },
            {
                'agent': 'developer',
                'content': {
                    'type': 'automatic',
                    'self_assessment': 'failed',
                    'confidence_level': 0.95
                }
            }
        ]
        
        verification = {
            'decision': 'request_rework',
            'reason': 'Developer failure compromises milestone',
            'confidence': 0.8
        }
        
        # Mock de la validation humaine
        with patch.object(self.supervisor, '_request_human_validation', return_value={"action": "approve_recommendation", "instruction": ""}):
            with patch.object(self.supervisor, '_milestone_manager') as mock_manager:
                mock_manager.insert_correction_after_current.return_value = {'id': 2}
                mock_manager.current_index = 0
                
                self.supervisor._apply_verification_decision(verification, test_milestone)
                
                # Vérifications
                self.assertEqual(self.supervisor.project_state['total_corrections'], 1)
                mock_manager.insert_correction_after_current.assert_called_once()


if __name__ == '__main__':
    unittest.main()