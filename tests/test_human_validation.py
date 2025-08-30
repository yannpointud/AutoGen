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
        
        # Vérifications
        self.assertEqual(result["action"], "approve")
        self.assertEqual(result["instruction"], "")
        mock_cli.display_warning.assert_called_once()
        mock_cli.display_info.assert_called_with("✅ Action approuvée par l'utilisateur")
    
    @patch('agents.supervisor.Prompt.ask')
    @patch('agents.supervisor.CLIInterface')
    def test_request_human_validation_alternative(self, mock_cli_class, mock_prompt):
        """Test de validation humaine avec réponse 'Alternative'."""
        # Mock de l'interface CLI
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        # Simuler choix utilisateur "2" (Alternative) puis instruction
        mock_prompt.side_effect = ["2", "Instruction alternative de test"]
        
        # Mock du LLM pour générer la question
        with patch.object(self.supervisor, 'generate_with_context', return_value="Question bien formulée ?"):
            result = self.supervisor._request_human_validation(
                reason="Test de validation",
                recommended_action="Action de test",
                milestone_details=self.test_milestone,
                agent_reports=[],
                verification_info={}
            )
        
        # Vérifications
        self.assertEqual(result["action"], "alternative")
        self.assertEqual(result["instruction"], "Instruction alternative de test")
        mock_cli.display_info.assert_called_with("📝 Instruction alternative reçue: Instruction alternative de test")
    
    @patch('agents.supervisor.Prompt.ask')
    @patch('agents.supervisor.CLIInterface')
    def test_request_human_validation_cancel(self, mock_cli_class, mock_prompt):
        """Test de validation humaine avec réponse 'Arrêter'."""
        # Mock de l'interface CLI
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        mock_cli.ask_confirmation.return_value = True  # Confirmer l'arrêt
        
        # Simuler choix utilisateur "3" (Arrêter)
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
        
        # Vérifications
        self.assertEqual(result["action"], "cancel")
        self.assertEqual(result["instruction"], "Arrêt demandé par l'utilisateur")
        mock_cli.ask_confirmation.assert_called_once()
        mock_cli.display_warning.assert_called_with("🛑 Arrêt du projet demandé par l'utilisateur")
    
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
        
        # Vérifications - doit retourner "approve" par défaut
        self.assertEqual(result["action"], "approve")
        self.assertEqual(result["instruction"], "")
        mock_cli.display_info.assert_called_with("Annulation de l'arrêt - Approuver par défaut")
    
    def test_stop_orchestration_gracefully(self):
        """Test de l'arrêt gracieux de l'orchestration."""
        # Mock de la création d'entrée de journal
        with patch.object(self.supervisor, '_create_journal_entry') as mock_journal:
            self.supervisor._stop_orchestration_gracefully("Test d'arrêt utilisateur")
        
        # Vérifications
        self.assertTrue(self.supervisor._orchestration_halted)
        self.assertEqual(self.supervisor.project_state['status'], 'stopped_by_user')
        self.assertIsNotNone(self.supervisor.project_state.get('stopped_at'))
        self.assertEqual(self.supervisor.project_state['stop_reason'], "Test d'arrêt utilisateur")
        mock_journal.assert_called_once()
    
    def test_point_intervention_1_max_corrections_approve(self):
        """Test du point d'intervention #1 : limite de corrections avec approbation."""
        # Simuler un milestone avec corrections maximales atteintes
        self.test_milestone['correction_attempts'] = 1  # max_corrections = 1 dans config
        
        # Mock de la validation humaine retournant "approve"
        with patch.object(self.supervisor, '_request_human_validation', return_value={"action": "approve", "instruction": ""}) as mock_validation:
            with patch.object(self.supervisor, '_mark_milestone_partially_completed') as mock_partial:
                
                # Simuler la décision de vérification
                verification = {
                    'decision': 'request_rework',
                    'reason': 'Encore des erreurs',
                    'confidence': 0.8
                }
                
                self.supervisor._apply_verification_decision(verification, self.test_milestone)
                
                # Vérifications
                mock_validation.assert_called_once()
                mock_partial.assert_called_once()
    
    def test_point_intervention_1_max_corrections_alternative(self):
        """Test du point d'intervention #1 : limite de corrections avec alternative."""
        # Simuler un milestone avec corrections maximales atteintes
        self.test_milestone['correction_attempts'] = 1
        
        # Mock du résultat de l'outil add_milestone
        mock_tool_result = Mock()
        mock_tool_result.status = 'success'
        
        # Mock de la validation humaine retournant "alternative"
        with patch.object(self.supervisor, '_request_human_validation', return_value={"action": "alternative", "instruction": "Faire ceci à la place"}) as mock_validation:
            with patch.object(self.supervisor, 'tools', {'add_milestone': Mock(return_value=mock_tool_result)}) as mock_tools:
                
                # Simuler la décision de vérification
                verification = {
                    'decision': 'request_rework',
                    'reason': 'Encore des erreurs',
                    'confidence': 0.8
                }
                
                self.supervisor._apply_verification_decision(verification, self.test_milestone)
                
                # Vérifications
                mock_validation.assert_called_once()
                mock_tools['add_milestone'].assert_called_once()
                # Le compteur de corrections doit être remis à zéro
                self.assertEqual(self.test_milestone['correction_attempts'], 0)
    
    def test_point_intervention_1_max_corrections_cancel(self):
        """Test du point d'intervention #1 : limite de corrections avec annulation."""
        # Simuler un milestone avec corrections maximales atteintes
        self.test_milestone['correction_attempts'] = 1
        
        # Mock de la validation humaine retournant "cancel"
        with patch.object(self.supervisor, '_request_human_validation', return_value={"action": "cancel", "instruction": "Arrêt demandé"}) as mock_validation:
            with patch.object(self.supervisor, '_stop_orchestration_gracefully') as mock_stop:
                
                # Simuler la décision de vérification
                verification = {
                    'decision': 'request_rework',
                    'reason': 'Encore des erreurs',
                    'confidence': 0.8
                }
                
                self.supervisor._apply_verification_decision(verification, self.test_milestone)
                
                # Vérifications
                mock_validation.assert_called_once()
                mock_stop.assert_called_once()
    
    def test_point_intervention_2_adjust_plan_approve(self):
        """Test du point d'intervention #2 : ajustement du plan avec approbation."""
        # Mock de la validation humaine retournant "approve"
        with patch.object(self.supervisor, '_request_human_validation', return_value={"action": "approve", "instruction": ""}) as mock_validation:
            with patch.object(self.supervisor, 'adjust_plan') as mock_adjust:
                with patch.object(self.supervisor, '_create_journal_entry') as mock_journal:
                    
                    # Simuler la décision de vérification
                    verification = {
                        'decision': 'adjust_plan',
                        'reason': 'Le plan doit être modifié',
                        'confidence': 0.9
                    }
                    
                    self.supervisor._apply_verification_decision(verification, self.test_milestone)
                    
                    # Vérifications
                    mock_validation.assert_called_once()
                    mock_adjust.assert_called_once()
                    mock_journal.assert_called_once()
                    # Le milestone doit être marqué comme complété
                    self.assertEqual(self.test_milestone['status'], 'completed')
                    self.assertEqual(self.supervisor.current_milestone_index, 1)
    
    def test_point_intervention_2_adjust_plan_alternative(self):
        """Test du point d'intervention #2 : ajustement du plan avec alternative."""
        # Mock du résultat de l'outil add_milestone
        mock_tool_result = Mock()
        mock_tool_result.status = 'success'
        
        # Mock de la validation humaine retournant "alternative"
        with patch.object(self.supervisor, '_request_human_validation', return_value={"action": "alternative", "instruction": "Faire autre chose"}) as mock_validation:
            with patch.object(self.supervisor, 'tools', {'add_milestone': Mock(return_value=mock_tool_result)}) as mock_tools:
                
                # Simuler la décision de vérification
                verification = {
                    'decision': 'adjust_plan',
                    'reason': 'Le plan doit être modifié',
                    'confidence': 0.9
                }
                
                self.supervisor._apply_verification_decision(verification, self.test_milestone)
                
                # Vérifications
                mock_validation.assert_called_once()
                mock_tools['add_milestone'].assert_called_once()
                # Le milestone doit être complété et l'index avancé
                self.assertEqual(self.test_milestone['status'], 'completed')
                self.assertEqual(self.supervisor.current_milestone_index, 1)
    
    def test_point_intervention_2_adjust_plan_cancel(self):
        """Test du point d'intervention #2 : ajustement du plan avec annulation."""
        # Mock de la validation humaine retournant "cancel"
        with patch.object(self.supervisor, '_request_human_validation', return_value={"action": "cancel", "instruction": "Arrêt demandé"}) as mock_validation:
            with patch.object(self.supervisor, '_stop_orchestration_gracefully') as mock_stop:
                
                # Simuler la décision de vérification
                verification = {
                    'decision': 'adjust_plan',
                    'reason': 'Le plan doit être modifié',
                    'confidence': 0.9
                }
                
                self.supervisor._apply_verification_decision(verification, self.test_milestone)
                
                # Vérifications
                mock_validation.assert_called_once()
                mock_stop.assert_called_once()
    
    def test_orchestration_with_halt_flag(self):
        """Test que l'orchestration s'arrête quand le flag halt est activé."""
        # Activer le flag d'arrêt
        self.supervisor._orchestration_halted = True
        
        # Mock de quelques méthodes pour éviter les erreurs
        with patch.object(self.supervisor, 'create_agents', return_value={}):
            with patch.object(self.supervisor, '_execute_milestone', return_value={}):
                with patch.object(self.supervisor, '_verify_milestone_completion', return_value={'decision': 'approve'}):
                    
                    result = self.supervisor.orchestrate()
                    
                    # La boucle ne devrait pas s'exécuter
                    self.assertEqual(len(result.get('milestones_results', [])), 0)
    
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
            
            # Doit retourner "approve" par défaut en cas d'erreur
            self.assertEqual(result["action"], "approve")
            self.assertIn("Erreur validation humaine", result["instruction"])


if __name__ == '__main__':
    unittest.main()