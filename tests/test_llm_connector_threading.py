"""
Test unitaire pour la sécurité multithreading du LLMConnector - Priorité: CRITIQUE
Valide que les verrous (_llm_execution_lock) et le singleton (LLMFactory) 
empêchent les race conditions lors d'appels concurrents.
"""

import unittest
import threading
import time
from unittest.mock import patch, MagicMock, Mock
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_connector import LLMFactory, MistralConnector, _llm_execution_lock
from core.global_rate_limiter import GlobalRateLimiter


class TestLLMConnectorThreading(unittest.TestCase):
    """Tests de sécurité multithreading pour LLMConnector."""
    
    def setUp(self):
        """Setup avant chaque test."""
        # Nettoyer le cache LLMFactory
        LLMFactory.clear_cache()
        
        # Reset le rate limiter
        GlobalRateLimiter._instance = None
        rate_limiter = GlobalRateLimiter()
        rate_limiter.reset_statistics()
        rate_limiter.update_rate_limit_interval(0.01)  # Très court pour tests
        
        # Mock pour éviter les vraies requêtes API
        self.mock_mistral_client = Mock()
        self.mock_response = Mock()
        self.mock_response.choices = [Mock()]
        self.mock_response.choices[0].message = Mock()
        self.mock_response.choices[0].message.content = "Test response"
        self.mock_mistral_client.chat.complete.return_value = self.mock_response
        
    def tearDown(self):
        """Cleanup après chaque test."""
        LLMFactory.clear_cache()
        GlobalRateLimiter._instance = None
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_singleton_thread_safety_llm_factory(self, mock_mistral_class):
        """Test que LLMFactory.create() est thread-safe pour le singleton."""
        mock_mistral_class.return_value = self.mock_mistral_client
        
        instances = []
        errors = []
        
        def create_connector():
            try:
                connector = LLMFactory.create(model="mistral-small")
                instances.append(connector)
            except Exception as e:
                errors.append(e)
        
        # Lancer plusieurs threads simultanément
        num_threads = 20
        threads = []
        
        for _ in range(num_threads):
            thread = threading.Thread(target=create_connector)
            threads.append(thread)
        
        # Démarrer tous les threads en même temps
        for thread in threads:
            thread.start()
        
        # Attendre la fin de tous les threads
        for thread in threads:
            thread.join()
        
        # Vérifier qu'aucune erreur n'est survenue
        self.assertEqual(len(errors), 0, f"Erreurs lors de la création: {errors}")
        
        # Vérifier que tous les connecteurs sont identiques (même instance)
        self.assertEqual(len(instances), num_threads)
        base_instance = instances[0]
        for instance in instances:
            self.assertIs(instance, base_instance)
            self.assertEqual(id(instance), id(base_instance))
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_different_models_different_instances(self, mock_mistral_class):
        """Test que différents modèles donnent différentes instances."""
        mock_mistral_class.return_value = self.mock_mistral_client
        
        instances = {}
        
        def create_connector(model_name):
            try:
                connector = LLMFactory.create(model=model_name)
                instances[model_name] = connector
            except Exception as e:
                instances[model_name] = e
        
        models = ["mistral-small", "mistral-medium", "codestral"]
        threads = []
        
        for model in models:
            thread = threading.Thread(target=create_connector, args=(model,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Vérifier que chaque modèle a sa propre instance
        self.assertEqual(len(instances), 3)
        instance_ids = [id(instance) for instance in instances.values()]
        self.assertEqual(len(set(instance_ids)), 3)  # Tous différents
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_llm_execution_lock_prevents_race_conditions(self, mock_mistral_class):
        """Test que _llm_execution_lock empêche les race conditions."""
        mock_mistral_class.return_value = self.mock_mistral_client
        
        # Configurer le mock pour simuler une latence
        call_times = []
        call_order = []
        
        def slow_api_call(*args, **kwargs):
            thread_id = threading.current_thread().ident
            start_time = time.time()
            call_order.append(f"start_{thread_id}")
            
            # Simuler une latence d'API
            time.sleep(0.05)
            
            end_time = time.time()
            call_times.append((thread_id, start_time, end_time))
            call_order.append(f"end_{thread_id}")
            
            return self.mock_response
        
        self.mock_mistral_client.chat.complete.side_effect = slow_api_call
        
        # Créer un connecteur
        connector = LLMFactory.create(model="mistral-small")
        
        results = []
        
        def make_llm_call(call_id):
            try:
                result = connector.generate("Test prompt", temperature=0.7)
                results.append((call_id, result, threading.current_thread().ident))
            except Exception as e:
                results.append((call_id, str(e), threading.current_thread().ident))
        
        # Lancer plusieurs appels concurrents
        num_calls = 5
        threads = []
        
        for i in range(num_calls):
            thread = threading.Thread(target=make_llm_call, args=(i,))
            threads.append(thread)
        
        start_time = time.time()
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Vérifier les résultats
        self.assertEqual(len(results), num_calls)
        self.assertEqual(len(call_times), num_calls)
        
        # Vérifier que les appels sont sérialisés (pas de chevauchement)
        call_times.sort(key=lambda x: x[1])  # Trier par start_time
        
        for i in range(1, len(call_times)):
            prev_end = call_times[i-1][2]
            curr_start = call_times[i][1]
            # Le prochain appel doit commencer après que le précédent se termine
            self.assertGreaterEqual(curr_start, prev_end - 0.01)  # Tolérance 10ms
        
        # Le temps total doit être au moins la somme des latences individuelles
        total_time = end_time - start_time
        expected_min_time = num_calls * 0.05  # 5 appels * 50ms chacun
        self.assertGreaterEqual(total_time, expected_min_time - 0.1)  # Tolérance
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_concurrent_factory_creation_same_model(self, mock_mistral_class):
        """Test la création factory concurrent pour le même modèle."""
        mock_mistral_class.return_value = self.mock_mistral_client
        
        creation_times = []
        instances = []
        
        def create_and_time():
            start_time = time.time()
            connector = LLMFactory.create(model="mistral-small")
            end_time = time.time()
            
            creation_times.append((start_time, end_time, threading.current_thread().ident))
            instances.append(connector)
        
        # Créer plusieurs instances simultanément
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_and_time) for _ in range(num_threads)]
            
            # Attendre la fin de tous
            for future in as_completed(futures):
                future.result()
        
        # Vérifier que toutes les instances sont identiques
        base_instance = instances[0]
        for instance in instances:
            self.assertIs(instance, base_instance)
        
        # Vérifier que le client Mistral n'a été créé qu'une fois
        self.assertEqual(mock_mistral_class.call_count, 1)
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_thread_isolation_different_calls(self, mock_mistral_class):
        """Test l'isolation des threads pour différents appels."""
        mock_mistral_class.return_value = self.mock_mistral_client
        
        # Configurer des réponses différentes par thread
        responses = {}
        
        def thread_specific_response(*args, **kwargs):
            thread_id = threading.current_thread().ident
            response_text = f"Response from thread {thread_id}"
            responses[thread_id] = response_text
            
            mock_resp = Mock()
            mock_resp.choices = [Mock()]
            mock_resp.choices[0].message = Mock()
            mock_resp.choices[0].message.content = response_text
            return mock_resp
        
        self.mock_mistral_client.chat.complete.side_effect = thread_specific_response
        
        # Créer plusieurs connecteurs et faire des appels
        results = {}
        
        def make_call(call_id):
            connector = LLMFactory.create(model="mistral-small")
            result = connector.generate(f"Prompt {call_id}")
            thread_id = threading.current_thread().ident
            results[call_id] = (result, thread_id)
        
        num_calls = 8
        with ThreadPoolExecutor(max_workers=num_calls) as executor:
            futures = {executor.submit(make_call, i): i for i in range(num_calls)}
            
            for future in as_completed(futures):
                future.result()
        
        # Vérifier que chaque thread a eu sa propre réponse
        self.assertEqual(len(results), num_calls)
        
        for call_id, (result, thread_id) in results.items():
            expected_response = f"Response from thread {thread_id}"
            self.assertEqual(result, expected_response)
    
    def test_lock_acquisition_timeout(self):
        """Test le comportement en cas de blocage prolongé du verrou."""
        # Simuler un verrou occupé
        lock_acquired = threading.Event()
        lock_released = threading.Event()
        
        def hold_lock():
            with _llm_execution_lock:
                lock_acquired.set()
                lock_released.wait(timeout=1.0)  # Maintenir le verrou 1 seconde max
        
        def try_acquire_lock():
            lock_acquired.wait(timeout=0.5)  # Attendre que l'autre thread ait le verrou
            start_time = time.time()
            with _llm_execution_lock:
                end_time = time.time()
                return end_time - start_time
        
        # Thread qui garde le verrou
        holder_thread = threading.Thread(target=hold_lock)
        holder_thread.start()
        
        # Thread qui essaie d'acquérir le verrou
        waiter_thread = threading.Thread(target=try_acquire_lock)
        waiter_thread.start()
        
        # Attendre un peu puis libérer le verrou
        time.sleep(0.2)
        lock_released.set()
        
        # Nettoyer
        holder_thread.join(timeout=2)
        waiter_thread.join(timeout=2)
        
        # Test réussi si aucun deadlock
        self.assertFalse(holder_thread.is_alive())
        self.assertFalse(waiter_thread.is_alive())
    
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    @patch('core.llm_connector.Mistral')
    def test_factory_clear_cache_thread_safety(self, mock_mistral_class):
        """Test que clear_cache est thread-safe."""
        mock_mistral_class.return_value = self.mock_mistral_client
        
        # Créer des instances
        for i in range(5):
            LLMFactory.create(model=f"model-{i}")
        
        results = []
        
        def clear_and_create():
            try:
                LLMFactory.clear_cache()
                connector = LLMFactory.create(model="test-model")
                results.append(("success", connector))
            except Exception as e:
                results.append(("error", str(e)))
        
        # Lancer plusieurs clear_cache concurrents
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=clear_and_create)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Vérifier qu'aucune erreur n'est survenue
        errors = [result for result in results if result[0] == "error"]
        self.assertEqual(len(errors), 0, f"Erreurs durant clear_cache: {errors}")
        
        # Vérifier que des connecteurs ont été créés
        successes = [result for result in results if result[0] == "success"]
        self.assertGreater(len(successes), 0)


if __name__ == '__main__':
    unittest.main()