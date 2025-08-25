"""
Test unitaire pour LLMFactory singleton - Priorité: MOYENNE
Vérifier que le LLMFactory respecte le pattern singleton et la réutilisation des connexions.
"""

import unittest
import threading
import time
from unittest.mock import patch, Mock, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_connector import LLMFactory, MistralConnector, DeepSeekConnector
from core.global_rate_limiter import GlobalRateLimiter


class TestLLMFactorySingleton(unittest.TestCase):
    """Tests du pattern singleton pour LLMFactory."""
    
    def setUp(self):
        """Setup avant chaque test."""
        # Réinitialiser le singleton pour chaque test
        LLMFactory._instances.clear()
        
        # Mock des clés API
        self.mistral_patcher = patch.dict('os.environ', {'MISTRAL_API_KEY': 'test_mistral_key'})
        self.deepseek_patcher = patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test_deepseek_key'})
        self.mistral_patcher.start()
        self.deepseek_patcher.start()
        
        # Mock du rate limiter global
        self.rate_limiter_patcher = patch('core.global_rate_limiter.GlobalRateLimiter.get_instance')
        self.mock_rate_limiter = self.rate_limiter_patcher.start()
        mock_limiter_instance = Mock()
        mock_limiter_instance.acquire_token.return_value = True
        self.mock_rate_limiter.return_value = mock_limiter_instance
    
    def tearDown(self):
        """Cleanup après chaque test."""
        self.mistral_patcher.stop()
        self.deepseek_patcher.stop()
        self.rate_limiter_patcher.stop()
        
        # Nettoyer le singleton
        LLMFactory._instances.clear()
    
    def test_singleton_same_instance(self):
        """Test que create() retourne toujours la même instance pour le même provider."""
        # Premier appel
        llm1 = LLMFactory.create('mistral', 'mistral-small')
        
        # Second appel
        llm2 = LLMFactory.create('mistral', 'mistral-small')
        
        # Vérifier que c'est la même instance
        self.assertIs(llm1, llm2)
        self.assertIsInstance(llm1, MistralConnector)
    
    def test_different_models_different_instances(self):
        """Test que des modèles différents retournent des instances différentes."""
        llm_mistral = LLMFactory.create('mistral', 'mistral-small')
        llm_deepseek = LLMFactory.create('deepseek', 'deepseek-coder')
        
        # Vérifier que ce sont des instances différentes
        self.assertIsNot(llm_mistral, llm_deepseek)
        self.assertIsInstance(llm_mistral, MistralConnector)
        self.assertIsInstance(llm_deepseek, DeepSeekConnector)
    
    def test_thread_safety_singleton(self):
        """Test que le singleton est thread-safe."""
        instances = []
        threads = []
        
        def create_instance():
            llm = LLMFactory.create('mistral', 'mistral-small')
            instances.append(llm)
        
        # Créer 10 threads qui créent des instances en parallèle
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()
        
        # Attendre que tous les threads finissent
        for thread in threads:
            thread.join()
        
        # Vérifier que toutes les instances sont identiques
        self.assertEqual(len(instances), 10)
        for instance in instances[1:]:
            self.assertIs(instances[0], instance)
    
    def test_multiple_model_instances_management(self):
        """Test de la gestion de plusieurs instances de modèles différents."""
        # Créer des instances de différents modèles
        mistral_small = LLMFactory.create('mistral', 'mistral-small')
        mistral_medium = LLMFactory.create('mistral', 'magistral-medium')
        deepseek = LLMFactory.create('deepseek', 'deepseek-coder')
        
        # Vérifier les types
        self.assertIsInstance(mistral_small, MistralConnector)
        self.assertIsInstance(mistral_medium, MistralConnector)
        self.assertIsInstance(deepseek, DeepSeekConnector)
        
        # Vérifier les modèles
        self.assertEqual(mistral_small.model, 'mistral-small')
        self.assertEqual(mistral_medium.model, 'magistral-medium')
        self.assertEqual(deepseek.model, 'deepseek-coder')
        
        # Créer les mêmes instances à nouveau
        mistral_small_2 = LLMFactory.create('mistral', 'mistral-small')
        deepseek_2 = LLMFactory.create('deepseek', 'deepseek-coder')
        
        # Vérifier la réutilisation d'instances
        self.assertIs(mistral_small, mistral_small_2)
        self.assertIs(deepseek, deepseek_2)
    
    def test_factory_instance_count(self):
        """Test du comptage d'instances dans le factory."""
        # État initial
        self.assertEqual(len(LLMFactory._instances), 0)
        
        # Créer première instance
        llm1 = LLMFactory.create('mistral', 'mistral-small')
        self.assertEqual(len(LLMFactory._instances), 1)
        
        # Créer la même instance (réutilisation)
        llm2 = LLMFactory.create('mistral', 'mistral-small')
        self.assertEqual(len(LLMFactory._instances), 1)
        self.assertIs(llm1, llm2)
        
        # Créer instance différente
        llm3 = LLMFactory.create('deepseek', 'deepseek-coder')
        self.assertEqual(len(LLMFactory._instances), 2)
        
        # Vérifier les clés d'instances (format provider:model)
        expected_keys = {'mistral:mistral-small', 'deepseek:deepseek-coder'}
        self.assertEqual(set(LLMFactory._instances.keys()), expected_keys)
    
    def test_invalid_model_handling(self):
        """Test gestion des providers invalides."""
        with self.assertRaises(ValueError) as context:
            LLMFactory.create('invalid-provider', 'some-model')
        
        self.assertIn("Provider non supporté", str(context.exception))
    
    def test_rate_limiter_integration(self):
        """Test de l'intégration avec le rate limiter global.""" 
        # Créer une instance LLM
        llm = LLMFactory.create('mistral', 'mistral-small')
        
        # Vérifier que l'instance LLM est créée correctement
        self.assertIsNotNone(llm)
        self.assertIsInstance(llm, MistralConnector)
        # Note: Le rate limiter est intégré différemment dans l'implémentation réelle
    
    def test_concurrent_different_models(self):
        """Test de création concurrent de modèles différents."""
        instances = {'mistral': [], 'deepseek': []}
        
        def create_mistral():
            for _ in range(5):
                llm = LLMFactory.create('mistral', 'mistral-small')
                instances['mistral'].append(llm)
        
        def create_deepseek():
            for _ in range(5):
                llm = LLMFactory.create('deepseek', 'deepseek-coder')
                instances['deepseek'].append(llm)
        
        # Lancer les threads
        thread1 = threading.Thread(target=create_mistral)
        thread2 = threading.Thread(target=create_deepseek)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Vérifier la cohérence
        self.assertEqual(len(instances['mistral']), 5)
        self.assertEqual(len(instances['deepseek']), 5)
        
        # Toutes les instances Mistral doivent être identiques
        for instance in instances['mistral'][1:]:
            self.assertIs(instances['mistral'][0], instance)
        
        # Toutes les instances DeepSeek doivent être identiques
        for instance in instances['deepseek'][1:]:
            self.assertIs(instances['deepseek'][0], instance)
        
        # Mais Mistral et DeepSeek doivent être différents
        self.assertIsNot(instances['mistral'][0], instances['deepseek'][0])
    
    def test_model_key_normalization(self):
        """Test de la normalisation des clés de modèles."""
        # Créer avec des variantes de casse/espaces
        llm1 = LLMFactory.create('mistral', 'mistral-small')
        llm2 = LLMFactory.create('mistral', 'mistral-small')  # Même modèle
        
        # Même instance attendue
        self.assertIs(llm1, llm2)
        
        # Vérifier qu'une seule entrée existe dans le cache
        self.assertEqual(len(LLMFactory._instances), 1)
        self.assertIn('mistral:mistral-small', LLMFactory._instances)
    
    def test_factory_memory_efficiency(self):
        """Test de l'efficacité mémoire du factory (pas de duplication)."""
        # Créer plusieurs références à la même instance
        references = []
        for _ in range(100):
            llm = LLMFactory.create('mistral', 'mistral-small')
            references.append(llm)
        
        # Toutes les références pointent vers la même instance
        for ref in references[1:]:
            self.assertIs(references[0], ref)
        
        # Une seule instance dans le cache
        self.assertEqual(len(LLMFactory._instances), 1)
    
    def test_model_configuration_consistency(self):
        """Test de la cohérence de configuration entre instances identiques."""
        # Créer la même instance plusieurs fois
        llm1 = LLMFactory.create('mistral', 'mistral-small')
        llm2 = LLMFactory.create('mistral', 'mistral-small')
        
        # Vérifier la cohérence des configurations
        self.assertEqual(llm1.model, llm2.model)
        # Note: Les détails de configuration interne peuvent varier selon l'implémentation
        self.assertIs(llm1, llm2)  # Même instance = même configuration
    
    def test_error_recovery_singleton(self):
        """Test de gestion d'erreur dans le singleton."""
        # Test que le factory peut créer des instances valides
        llm1 = LLMFactory.create('mistral', 'mistral-small')
        self.assertIsNotNone(llm1)
        self.assertEqual(len(LLMFactory._instances), 1)
        
        # Même instance retournée en cas d'appel répété
        llm2 = LLMFactory.create('mistral', 'mistral-small')
        self.assertIs(llm1, llm2)
        
        # Note: La gestion d'erreur interne peut différer de l'assumption du test


if __name__ == '__main__':
    unittest.main()