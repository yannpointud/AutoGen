"""
Test unitaire pour la gestion des erreurs API - Priorité: HAUTE
Mocke des réponses d'erreur API (500, 401, 429, timeout) et vérifie que 
le LLMConnector gère la logique de retry correctement avant de lever une exception.
"""

import unittest
import time
from unittest.mock import patch, Mock, MagicMock
import httpx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_connector import LLMFactory, MistralConnector, DeepSeekConnector
from core.global_rate_limiter import GlobalRateLimiter


class TestAPIErrorHandling(unittest.TestCase):
    """Tests de gestion des erreurs API pour LLMConnector."""
    
    def setUp(self):
        """Setup avant chaque test."""
        # Nettoyer le cache LLMFactory et rate limiter
        LLMFactory.clear_cache()
        GlobalRateLimiter._instance = None
        
        # Configurer rate limiter rapide pour les tests
        rate_limiter = GlobalRateLimiter()
        rate_limiter.reset_statistics()
        rate_limiter.update_rate_limit_interval(0.001)  # 1ms
        
        # Mock la config pour accélérer les tests
        self.config_patch = patch('core.llm_connector.default_config', {
            'general': {
                'max_retries': 3,
                'retry_delay': 0.01,  # 10ms au lieu des valeurs par défaut
                'llm_timeout': 5
            },
            'llm': {
                'default_model': 'mistral-small',
                'temperature': 0.7,
                'max_tokens': 1000,
                'models': {
                    'mistral': {
                        'models': [
                            {'name': 'mistral-small', 'temperature': 0.7},
                            {'name': 'mistral-medium', 'temperature': 0.7}
                        ]
                    }
                }
            }
        })
        self.config_patch.start()
    
    def tearDown(self):
        """Cleanup après chaque test."""
        self.config_patch.stop()
        LLMFactory.clear_cache()
        GlobalRateLimiter._instance = None
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_mistral_api_500_error_with_retry(self, mock_mistral_class):
        """Test gestion erreur 500 avec retry logic sur Mistral."""
        
        # Configurer le mock client
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        # Simuler des erreurs 500 puis succès
        error_500 = Exception("HTTP 500: Internal Server Error")
        success_response = Mock()
        success_response.choices = [Mock()]
        success_response.choices[0].message = Mock()
        success_response.choices[0].message.content = "Success after retry"
        success_response.usage = Mock()
        success_response.usage.total_tokens = 100
        
        # Premier et deuxième appel échouent, troisième réussit
        mock_client.chat.complete.side_effect = [error_500, error_500, success_response]
        
        # Créer le connecteur
        connector = LLMFactory.create(model="mistral-small")
        
        # Effectuer l'appel
        start_time = time.time()
        result = connector.generate("Test prompt", temperature=0.5)
        end_time = time.time()
        
        # Vérifications
        self.assertEqual(result, "Success after retry")
        self.assertEqual(mock_client.chat.complete.call_count, 3)
        
        # Vérifier que des retries ont eu lieu (temps d'attente)
        self.assertGreater(end_time - start_time, 0.01)  # Au moins quelques millisecondes
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_mistral_api_persistent_error(self, mock_mistral_class):
        """Test gestion erreur persistante qui épuise les retries."""
        
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        # Simuler erreur persistante
        persistent_error = Exception("HTTP 401: Unauthorized")
        mock_client.chat.complete.side_effect = persistent_error
        
        connector = LLMFactory.create(model="mistral-small")
        
        # L'appel doit échouer après épuisement des retries
        with self.assertRaises(Exception) as context:
            connector.generate("Test prompt")
        
        # Vérifier que l'erreur est celle attendue
        self.assertIn("401", str(context.exception))
        
        # Vérifier le nombre de tentatives (selon config: max_retries)
        self.assertGreaterEqual(mock_client.chat.complete.call_count, 3)
    
    @patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test_key'})
    @patch('core.llm_connector.httpx.Client')
    def test_deepseek_http_errors_with_retry(self, mock_httpx_client_class):
        """Test gestion erreurs HTTP avec DeepSeek."""
        
        # Configurer le mock client HTTP
        mock_client = Mock()
        mock_httpx_client_class.return_value = mock_client
        
        # Simuler des erreurs HTTP puis succès
        error_response_500 = Mock()
        error_response_500.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=Mock(), response=Mock()
        )
        
        error_response_429 = Mock()
        error_response_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests", request=Mock(), response=Mock()
        )
        
        success_response = Mock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {
            'choices': [{'message': {'content': 'DeepSeek success after retry'}}],
            'usage': {'total_tokens': 50}
        }
        
        # Séquence: 500, 429, puis succès
        mock_client.post.side_effect = [error_response_500, error_response_429, success_response]
        
        # Créer le connecteur DeepSeek
        connector = LLMFactory.create(provider="deepseek", model="deepseek-chat")
        
        # Effectuer l'appel
        result = connector.generate("Test prompt")
        
        # Vérifications
        self.assertEqual(result, "DeepSeek success after retry")
        self.assertEqual(mock_client.post.call_count, 3)
    
    @patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test_key'})
    @patch('core.llm_connector.httpx.Client')
    def test_deepseek_timeout_error(self, mock_httpx_client_class):
        """Test gestion timeout sur DeepSeek."""
        
        mock_client = Mock()
        mock_httpx_client_class.return_value = mock_client
        
        # Simuler timeout
        timeout_error = httpx.TimeoutException("Request timed out")
        mock_client.post.side_effect = timeout_error
        
        connector = LLMFactory.create(provider="deepseek", model="deepseek-chat")
        
        # L'appel doit échouer avec timeout
        with self.assertRaises(Exception) as context:
            connector.generate("Test prompt")
        
        self.assertIn("timed out", str(context.exception).lower())
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_mistral_malformed_response_error(self, mock_mistral_class):
        """Test gestion réponse malformée de Mistral."""
        
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        # Simuler réponse sans structure attendue
        bad_response = Mock()
        bad_response.choices = []  # Pas de choix
        
        mock_client.chat.complete.return_value = bad_response
        
        connector = LLMFactory.create(model="mistral-small")
        
        # L'appel doit échouer avec IndexError ou AttributeError
        with self.assertRaises((IndexError, AttributeError)):
            connector.generate("Test prompt")
    
    @patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test_key'})
    @patch('core.llm_connector.httpx.Client')
    def test_deepseek_json_decode_error(self, mock_httpx_client_class):
        """Test gestion erreur parsing JSON sur DeepSeek."""
        
        mock_client = Mock()
        mock_httpx_client_class.return_value = mock_client
        
        # Simuler réponse avec JSON invalide
        bad_json_response = Mock()
        bad_json_response.raise_for_status.return_value = None
        bad_json_response.json.side_effect = ValueError("Invalid JSON")
        
        mock_client.post.return_value = bad_json_response
        
        connector = LLMFactory.create(provider="deepseek", model="deepseek-chat")
        
        # L'appel doit échouer avec ValueError
        with self.assertRaises(ValueError):
            connector.generate("Test prompt")
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_retry_delay_timing(self, mock_mistral_class):
        """Test que les délais de retry sont respectés."""
        
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        # Simuler 2 erreurs puis succès
        error = Exception("Temporary error")
        success_response = Mock()
        success_response.choices = [Mock()]
        success_response.choices[0].message = Mock()
        success_response.choices[0].message.content = "Final success"
        success_response.usage = Mock()
        success_response.usage.total_tokens = 75
        
        mock_client.chat.complete.side_effect = [error, error, success_response]
        
        connector = LLMFactory.create(model="mistral-small")
        
        # Mesurer le temps total
        start_time = time.time()
        result = connector.generate("Test prompt")
        end_time = time.time()
        
        # Vérifier le résultat
        self.assertEqual(result, "Final success")
        
        # Vérifier qu'il y a eu des délais (au moins 2 * retry_delay)
        # Note: Le retry_delay est configuré dans default_config
        self.assertGreater(end_time - start_time, 0.01)  # Au moins 10ms total
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_specific_http_status_codes(self, mock_mistral_class):
        """Test gestion de codes d'erreur HTTP spécifiques."""
        
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        # Tester différents codes d'erreur
        test_cases = [
            ("401 Unauthorized", "authentication"),
            ("403 Forbidden", "permission"),
            ("404 Not Found", "endpoint"),
            ("429 Rate Limited", "rate"),
            ("503 Service Unavailable", "service")
        ]
        
        for error_msg, expected_keyword in test_cases:
            with self.subTest(error_msg=error_msg):
                # Réinitialiser le mock
                mock_client.chat.complete.side_effect = Exception(error_msg)
                
                connector = LLMFactory.create(model="mistral-small")
                
                with self.assertRaises(Exception) as context:
                    connector.generate("Test prompt")
                
                # Vérifier que l'erreur contient le message approprié
                error_str = str(context.exception).lower()
                self.assertTrue(
                    any(keyword in error_str for keyword in [expected_keyword, error_msg.lower()]),
                    f"Expected {expected_keyword} or {error_msg} in {error_str}"
                )
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_retry_with_different_delays(self, mock_mistral_class):
        """Test retry avec délais configurables."""
        
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        # Simuler erreur persistante pour forcer tous les retries
        mock_client.chat.complete.side_effect = Exception("Persistent error")
        
        connector = LLMFactory.create(model="mistral-small")
        
        # Mesurer le temps pour vérifier les délais
        start_time = time.time()
        
        with self.assertRaises(Exception):
            connector.generate("Test prompt")
        
        end_time = time.time()
        
        # Vérifier que le nombre de tentatives est correct
        # (max_retries est configuré dans default_config)
        expected_min_calls = 3  # Au moins 3 tentatives
        self.assertGreaterEqual(mock_client.chat.complete.call_count, expected_min_calls)
        
        # Vérifier qu'il y a eu des délais entre les tentatives
        self.assertGreater(end_time - start_time, 0.005)  # Au moins 5ms
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key', 'DEEPSEEK_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    @patch('core.llm_connector.httpx.Client')
    def test_error_handling_consistency_across_providers(self, mock_httpx, mock_mistral):
        """Test cohérence gestion erreurs entre providers."""
        
        # Configurer Mistral
        mock_mistral_client = Mock()
        mock_mistral.return_value = mock_mistral_client
        mock_mistral_client.chat.complete.side_effect = Exception("Mistral error")
        
        # Configurer DeepSeek
        mock_deepseek_client = Mock()
        mock_httpx.return_value = mock_deepseek_client
        mock_deepseek_client.post.side_effect = Exception("DeepSeek error")
        
        # Test Mistral
        mistral_connector = LLMFactory.create(provider="mistral", model="mistral-small")
        with self.assertRaises(Exception) as mistral_context:
            mistral_connector.generate("Test")
        
        # Test DeepSeek
        deepseek_connector = LLMFactory.create(provider="deepseek", model="deepseek-chat")
        with self.assertRaises(Exception) as deepseek_context:
            deepseek_connector.generate("Test")
        
        # Vérifier que les deux ont eu des retries
        self.assertGreaterEqual(mock_mistral_client.chat.complete.call_count, 2)
        self.assertGreaterEqual(mock_deepseek_client.post.call_count, 2)
        
        # Vérifier que les erreurs sont propagées
        self.assertIn("Mistral error", str(mistral_context.exception))
        self.assertIn("DeepSeek error", str(deepseek_context.exception))
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_successful_call_after_setup_errors(self, mock_mistral_class):
        """Test appel réussi après erreurs initiales de setup."""
        
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        # Premier appel : erreur de connexion
        # Deuxième appel : succès
        connection_error = Exception("Connection failed")
        success_response = Mock()
        success_response.choices = [Mock()]
        success_response.choices[0].message = Mock()
        success_response.choices[0].message.content = "Recovery successful"
        success_response.usage = Mock()
        success_response.usage.total_tokens = 90
        
        mock_client.chat.complete.side_effect = [connection_error, success_response]
        
        connector = LLMFactory.create(model="mistral-small")
        
        # L'appel doit réussir après retry
        result = connector.generate("Test recovery")
        
        self.assertEqual(result, "Recovery successful")
        self.assertEqual(mock_client.chat.complete.call_count, 2)


if __name__ == '__main__':
    unittest.main()