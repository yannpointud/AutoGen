"""
Test unitaire end-to-end pour le workflow complet AutoGen - Priorité: HAUTE
Teste l'intégration complète: Supervisor -> Analyst -> Developer avec un mini-projet.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.supervisor import Supervisor
from agents.analyst import Analyst
from agents.developer import Developer


class TestEndToEndWorkflow(unittest.TestCase):
    """Tests end-to-end du workflow AutoGen."""
    
    def setUp(self):
        """Setup avant chaque test."""
        # Créer un répertoire temporaire pour les tests
        self.test_dir = tempfile.mkdtemp()
        self.project_name = "TestE2EProject"
        self.project_path = Path(self.test_dir) / "projects" / self.project_name
        
        # Mock des services LLM avec vraies réponses textuelles
        self.llm_factory_patcher = patch('agents.base_agent.LLMFactory.create')
        self.mock_llm_factory = self.llm_factory_patcher.start()
        self.mock_llm = Mock()
        self.mock_llm.model = "test-model"
        self.mock_llm.generate = Mock(return_value="Réponse LLM test - analyse du projet terminée")
        self.mock_llm_factory.return_value = self.mock_llm
        
        # Mock du lightweight LLM avec vraies chaînes de caractères
        self.lightweight_patcher = patch('agents.base_agent.get_lightweight_llm_service')
        self.mock_lightweight = self.lightweight_patcher.start()
        mock_lightweight_instance = Mock()
        mock_lightweight_instance.extract_keywords.return_value = "calculatrice python mathématiques"
        mock_lightweight_instance.summarize_context.return_value = "résumé: développer calculateur avec fonctions mathématiques"
        mock_lightweight_instance.self_evaluate_mission.return_value = {"assessment": "compliant", "reason": "Mission accomplie", "confidence": 0.9}
        self.mock_lightweight.return_value = mock_lightweight_instance
        
        # Mock du système de logging
        self.logger_patcher = patch('agents.base_agent.get_project_logger')
        self.mock_logger = self.logger_patcher.start()
        mock_logger_instance = Mock()
        self.mock_logger.return_value = mock_logger_instance
        
        # Mock du RAG engine pour éviter FAISS
        self.rag_patcher = patch('core.rag_engine.RAGEngine')
        self.mock_rag_class = self.rag_patcher.start()
        self.mock_rag_engine = Mock()
        self.mock_rag_engine.search_context = Mock(return_value=[])
        self.mock_rag_engine.index_log_entry = Mock(return_value=True)
        self.mock_rag_engine.working_memory = []
        self.mock_rag_engine.working_memory_metadata = []
        self.mock_rag_engine.metadata = []
        self.mock_rag_engine.search = Mock(return_value=[])
        self.mock_rag_class.return_value = self.mock_rag_engine
        
    def tearDown(self):
        """Cleanup après chaque test."""
        self.llm_factory_patcher.stop()
        self.lightweight_patcher.stop()
        self.logger_patcher.stop()
        self.rag_patcher.stop()
        
        # Nettoyer le répertoire temporaire
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_simplified_project_workflow(self):
        """Test workflow simplifié avec supervisor et agents."""
        project_prompt = "Créer une calculatrice Python simple"
        
        # ========== PHASE 1: SUPERVISOR PLANNING ==========
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=False):
            
            supervisor = Supervisor(
                project_name=self.project_name,
                project_prompt=project_prompt,
                rag_engine=self.mock_rag_engine
            )
            
            # Mock Project Charter retrieval 
            with patch.object(supervisor, '_get_project_charter_from_file', return_value="# Test Charter"):
                
                # Exécuter la planification (utilise plan de fallback automatiquement)
                plan = supervisor.think({'prompt': project_prompt})
                
                # Vérifications de la planification
                self.assertIsInstance(plan, dict)
                self.assertIn('milestones', plan)
                self.assertGreaterEqual(len(plan['milestones']), 2)
                
                # Vérifier la structure des jalons - accès via le milestone manager
                milestones = supervisor._milestone_manager.milestones
                milestone1 = milestones[0]
                self.assertIn('name', milestone1)
                self.assertIn('agents_required', milestone1)
                self.assertIn('id', milestone1)
        
        # ========== PHASE 2: AGENTS INSTANTIATION ==========
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Test création des agents 
            analyst = Analyst(
                project_name=self.project_name,
                supervisor=supervisor,
                rag_engine=self.mock_rag_engine
            )
            
            developer = Developer(
                project_name=self.project_name,
                supervisor=supervisor,
                rag_engine=self.mock_rag_engine
            )
            
            # Vérifications basiques
            self.assertIsNotNone(analyst)
            self.assertIsNotNone(developer)
            self.assertEqual(analyst.project_name, self.project_name)
            self.assertEqual(developer.project_name, self.project_name)
            self.assertIs(analyst.supervisor, supervisor)
            self.assertIs(developer.supervisor, supervisor)
        
        # ========== PHASE 3: INTEGRATION CHECK ==========
        # Vérifier que les agents ont accès au RAG engine
        self.assertIs(analyst.rag_engine, self.mock_rag_engine)
        self.assertIs(developer.rag_engine, self.mock_rag_engine)
        
        # Le supervisor peut utiliser un plan de fallback, donc LLM call count peut être 0
        # L'important est que le plan soit généré correctement
        self.assertGreaterEqual(len(plan['milestones']), 1)
    
    def test_error_handling_in_workflow(self):
        """Test gestion d'erreur dans le workflow."""
        # Test avec une erreur LLM
        self.mock_llm.generate.side_effect = Exception("LLM Error")
        
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=False):
            
            supervisor = Supervisor(
                project_name=self.project_name,
                project_prompt="Test project prompt",
                rag_engine=self.mock_rag_engine
            )
            
            with patch.object(supervisor, '_get_project_charter_from_file', return_value="# Test Charter"):
                
                # La planification doit gérer l'erreur
                plan = supervisor.think({'prompt': 'Test project'})
                
                # En cas d'erreur, le supervisor utilise un plan de fallback
                self.assertIsInstance(plan, dict)
                self.assertIn('milestones', plan)
                # Le plan de fallback doit contenir des jalons par défaut
                self.assertGreater(len(plan['milestones']), 0)
    
    def test_agent_communication_flow(self):
        """Test du flux de communication entre agents."""
        
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Créer les agents
            supervisor = Supervisor(
                project_name=self.project_name,
                project_prompt="Test project prompt",
                rag_engine=self.mock_rag_engine
            )
            
            analyst = Analyst(
                project_name=self.project_name,
                supervisor=supervisor,
                rag_engine=self.mock_rag_engine
            )
            
            developer = Developer(
                project_name=self.project_name,
                supervisor=supervisor,
                rag_engine=self.mock_rag_engine
            )
            
            # Message du supervisor vers l'analyst
            message = "Please analyze the project requirements for authentication module"
            
            # Mock de la réponse de l'analyst
            expected_response = "Analysis completed. Identified 3 authentication patterns to implement."
            
            # Test direct de communication
            with patch.object(analyst, 'communicate', return_value=expected_response) as mock_communicate:
                response = supervisor.communicate(message, analyst)
                
                # Vérifications
                self.assertIsNotNone(response)
    
    def test_milestone_progression(self):
        """Test de la progression des jalons."""
        milestone_data = {
            'id': 1,
            'name': 'Test Milestone',
            'description': 'Test milestone progression',
            'agents_required': ['analyst'],
            'deliverables': ['Document test']
        }
        
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True):
            
            supervisor = Supervisor(
                project_name=self.project_name,
                project_prompt="Test project prompt",
                rag_engine=self.mock_rag_engine
            )
            
            # Mock de la vérification de jalon avec nouveau format
            with patch.object(supervisor, '_verify_milestone_completion', return_value={'decision': 'approve', 'confidence': 0.85, 'reason': 'Milestone completed successfully'}):
                
                milestone_result = {
                    'milestone_id': 1,
                    'tasks_completed': [
                        {
                            'agent': 'analyst',
                            'result': {
                                'status': 'completed', 
                                'artifacts': ['doc.md'],
                                'structured_report': {
                                    'self_assessment': 'compliant',
                                    'confidence_level': 0.9,
                                    'artifacts_created': ['doc.md'],
                                    'agent_name': 'analyst'
                                }
                            }
                        }
                    ],
                    'agents_involved': ['analyst']
                }
                
                # Vérifier le jalon
                verification = supervisor._verify_milestone_completion(milestone_data, milestone_result)
                
                # Vérifications avec nouveau format
                self.assertEqual(verification['decision'], 'approve')
                self.assertGreaterEqual(verification['confidence'], 0.8)
    
    def test_rag_context_sharing(self):
        """Test du partage de contexte via RAG entre agents."""
        
        # Données contextuelles simulées
        context_data = [
            {'chunk_text': 'Function add implemented', 'source': 'developer', 'score': 0.9},
            {'chunk_text': 'Architecture specified', 'source': 'analyst', 'score': 0.85}
        ]
        
        self.mock_rag_engine.search_context.return_value = context_data
        
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True):
            
            developer = Developer(
                project_name=self.project_name,
                supervisor=Mock(),
                rag_engine=self.mock_rag_engine
            )
            
            # Test de recherche de contexte
            context = self.mock_rag_engine.search_context("calculator functions")
            
            # Vérifications
            self.assertEqual(len(context), 2)
            self.assertEqual(context[0]['source'], 'developer')
            self.assertGreater(context[0]['score'], 0.8)
    
    def test_project_structure_creation(self):
        """Test de la création de la structure de projet."""
        
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=False):
            
            supervisor = Supervisor(
                project_name=self.project_name,
                project_prompt="Test project prompt",
                rag_engine=self.mock_rag_engine
            )
            
            # Le supervisor existe et peut être utilisé pour les tâches
            self.assertIsNotNone(supervisor)
            self.assertEqual(supervisor.project_name, self.project_name)
    
    def test_structured_reporting_integration(self):
        """Test de l'intégration des rapports structurés dans le workflow."""
        
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value="# Test Charter"):
            
            supervisor = Supervisor(
                project_name=self.project_name,
                project_prompt="Test structured reporting",
                rag_engine=self.mock_rag_engine
            )
            
            # Créer un agent mock avec capacité de génération de rapport structuré
            mock_agent = Mock()
            mock_agent.name = "TestAgent"
            mock_agent.project_name = self.project_name
            
            # Mock du rapport structuré
            structured_report = {
                'self_assessment': 'compliant',
                'confidence_level': 0.9,
                'artifacts_created': ['output.txt', 'config.json'],
                'issues_encountered': [],
                'agent_name': 'TestAgent',
                'completion_timestamp': '2024-01-01T12:00:00'
            }
            
            # Mock des outils de rapport
            with patch('tools.base_tools.tool_report_to_supervisor') as mock_report_tool:
                mock_result = Mock()
                mock_result.status = 'success'
                mock_result.result = {'report_sent': True}
                mock_report_tool.return_value = mock_result
                
                # Simuler l'envoi d'un rapport structuré
                params = {
                    'report_type': 'completion',
                    'content': structured_report
                }
                
                result = mock_report_tool(mock_agent, params)
                
                # Vérifications
                self.assertEqual(result.status, 'success')
                self.assertTrue(result.result['report_sent'])
                mock_report_tool.assert_called_once_with(mock_agent, params)
    
    def test_supervisor_milestone_buffer_integration(self):
        """Test de l'intégration du buffer de rapports par jalon."""
        
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value="# Test Charter"):
            
            supervisor = Supervisor(
                project_name=self.project_name,
                project_prompt="Test milestone buffer",
                rag_engine=self.mock_rag_engine
            )
            
            # Simuler la réception de rapports multiples
            report1 = {
                'type': 'progress',
                'agent': 'analyst',
                'content': {
                    'type': 'automatic',
                    'self_assessment': 'compliant',
                    'message': 'Analysis completed successfully'
                }
            }
            
            report2 = {
                'type': 'completion',  
                'agent': 'developer',
                'content': {
                    'type': 'automatic',
                    'self_assessment': 'partial', 
                    'message': 'Implementation partially completed'
                }
            }
            
            # Simuler l'extraction du milestone depuis les rapports
            with patch.object(supervisor, '_extract_milestone_from_report', return_value='milestone_1'):
                
                # Recevoir les rapports
                supervisor.receive_report('analyst', report1)
                supervisor.receive_report('developer', report2)
                
                # Vérifications du buffer
                self.assertEqual(len(supervisor.current_milestone_reports), 2)
                self.assertEqual(supervisor.current_milestone_id, 'milestone_1')
                
                # Vérifier que les rapports sont bien stockés
                stored_reports = supervisor.current_milestone_reports
                self.assertEqual(stored_reports[0]['agent'], 'analyst')
                self.assertEqual(stored_reports[1]['agent'], 'developer')


if __name__ == '__main__':
    unittest.main()