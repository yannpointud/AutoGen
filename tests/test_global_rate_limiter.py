"""
Test unitaire pour le GlobalRateLimiter - Priorité: CRITIQUE
Valide le rate limiting global pour éviter les erreurs 429 API.
"""

import unittest
import time
import threading
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.global_rate_limiter import GlobalRateLimiter


class TestGlobalRateLimiter(unittest.TestCase):
    """Tests pour GlobalRateLimiter."""
    
    def setUp(self):
        """Setup avant chaque test."""
        # Créer une nouvelle instance pour chaque test
        GlobalRateLimiter._instance = None
        self.rate_limiter = GlobalRateLimiter()
        self.rate_limiter.reset_statistics()
        # Définir un intervalle court pour les tests
        self.rate_limiter.update_rate_limit_interval(0.1)  # 100ms
    
    def tearDown(self):
        """Cleanup après chaque test."""
        if hasattr(self, 'rate_limiter'):
            self.rate_limiter.reset_statistics()
        GlobalRateLimiter._instance = None
    
    def test_singleton_behavior(self):
        """Test que GlobalRateLimiter est bien un singleton."""
        limiter1 = GlobalRateLimiter()
        limiter2 = GlobalRateLimiter()
        
        self.assertIs(limiter1, limiter2)
        self.assertEqual(id(limiter1), id(limiter2))
    
    def test_rate_limiting_enforcement_single_thread(self):
        """Test l'application du rate limiting dans un seul thread."""
        start_time = time.time()
        
        # Premier appel - ne devrait pas attendre
        self.rate_limiter.enforce_rate_limit("TestConnector1")
        first_call_time = time.time()
        
        # Deuxième appel rapide - devrait attendre
        self.rate_limiter.enforce_rate_limit("TestConnector2") 
        second_call_time = time.time()
        
        # Vérifier que le deuxième appel a attendu
        total_time = second_call_time - start_time
        self.assertGreaterEqual(total_time, 0.1)  # Au moins l'intervalle configuré
        
        # Vérifier les statistiques
        stats = self.rate_limiter.get_statistics()
        self.assertEqual(stats['total_requests'], 2)
        self.assertEqual(stats['blocked_requests'], 1)
        self.assertGreater(stats['total_wait_time'], 0)
    
    def test_concurrent_rate_limiting(self):
        """Test rate limiting avec plusieurs threads simultanés."""
        num_threads = 5
        results = []
        
        def make_request(thread_id):
            start_time = time.time()
            self.rate_limiter.enforce_rate_limit(f"Thread{thread_id}")
            end_time = time.time()
            return {
                'thread_id': thread_id,
                'duration': end_time - start_time,
                'timestamp': end_time
            }
        
        # Lancer les requêtes concurrentes
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_thread = {
                executor.submit(make_request, i): i for i in range(num_threads)
            }
            
            for future in as_completed(future_to_thread):
                results.append(future.result())
        
        # Trier par timestamp pour vérifier l'ordre
        results.sort(key=lambda x: x['timestamp'])
        
        # Vérifier que les appels sont espacés d'au moins l'intervalle
        for i in range(1, len(results)):
            time_diff = results[i]['timestamp'] - results[i-1]['timestamp']
            self.assertGreaterEqual(
                time_diff, 0.08,  # Tolérance de 20ms pour le threading
                f"Calls {i-1} and {i} too close: {time_diff:.3f}s"
            )
        
        # Vérifier les statistiques finales
        stats = self.rate_limiter.get_statistics()
        self.assertEqual(stats['total_requests'], num_threads)
        self.assertGreaterEqual(stats['blocked_requests'], num_threads - 1)
    
    def test_statistics_accuracy(self):
        """Test la précision des statistiques collectées."""
        # Effectuer quelques appels contrôlés
        start_time = time.time()
        
        self.rate_limiter.enforce_rate_limit("Call1")  # Premier appel - pas d'attente
        self.rate_limiter.enforce_rate_limit("Call2")  # Deuxième - attente
        self.rate_limiter.enforce_rate_limit("Call3")  # Troisième - attente
        
        stats = self.rate_limiter.get_statistics()
        
        # Vérifier les compteurs
        self.assertEqual(stats['total_requests'], 3)
        self.assertEqual(stats['blocked_requests'], 2)
        
        # Vérifier les temps
        self.assertGreater(stats['total_wait_time'], 0)
        self.assertGreater(stats['avg_wait_time'], 0)
        
        # Vérifier les ratios
        expected_block_rate = 2/3
        self.assertAlmostEqual(stats['block_rate'], expected_block_rate, places=2)
    
    def test_rate_limit_interval_update(self):
        """Test la mise à jour dynamique de l'intervalle."""
        # Tester avec l'intervalle initial
        self.rate_limiter.enforce_rate_limit("Call1")
        start_time = time.time()
        self.rate_limiter.enforce_rate_limit("Call2")
        end_time = time.time()
        
        initial_wait = end_time - start_time
        self.assertGreaterEqual(initial_wait, 0.08)  # Tolérance
        
        # Changer l'intervalle
        self.rate_limiter.reset_statistics()
        self.rate_limiter.update_rate_limit_interval(0.05)  # 50ms
        
        # Tester avec le nouvel intervalle
        self.rate_limiter.enforce_rate_limit("Call3")
        start_time = time.time()
        self.rate_limiter.enforce_rate_limit("Call4")
        end_time = time.time()
        
        new_wait = end_time - start_time
        self.assertGreaterEqual(new_wait, 0.03)  # Tolérance pour 50ms
        self.assertLess(new_wait, initial_wait)  # Doit être plus court
    
    def test_statistics_reset(self):
        """Test la remise à zéro des statistiques."""
        # Générer des statistiques
        self.rate_limiter.enforce_rate_limit("Call1")
        self.rate_limiter.enforce_rate_limit("Call2")
        
        # Vérifier que les stats ne sont pas vides
        stats_before = self.rate_limiter.get_statistics()
        self.assertGreater(stats_before['total_requests'], 0)
        
        # Reset
        self.rate_limiter.reset_statistics()
        
        # Vérifier la remise à zéro
        stats_after = self.rate_limiter.get_statistics()
        self.assertEqual(stats_after['total_requests'], 0)
        self.assertEqual(stats_after['blocked_requests'], 0)
        self.assertEqual(stats_after['total_wait_time'], 0)
    
    @patch('core.global_rate_limiter.default_config')
    def test_config_fallback(self, mock_config):
        """Test le fallback de configuration en cas d'erreur."""
        # Simuler une config défaillante
        mock_config.get.side_effect = Exception("Config error")
        
        # Recréer l'instance
        GlobalRateLimiter._instance = None
        limiter = GlobalRateLimiter()
        
        # Vérifier que l'intervalle par défaut est utilisé
        stats = limiter.get_statistics()
        self.assertEqual(limiter._request_interval, 1.5)  # Valeur par défaut
    
    def test_thread_safety_singleton(self):
        """Test que la création singleton est thread-safe."""
        instances = []
        
        def create_instance():
            instances.append(GlobalRateLimiter())
        
        # Créer plusieurs instances concurrentes
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()
        
        # Attendre la fin
        for thread in threads:
            thread.join()
        
        # Vérifier que toutes les instances sont identiques
        for instance in instances:
            self.assertIs(instance, instances[0])
    
    def test_no_wait_for_spaced_calls(self):
        """Test qu'aucune attente n'est nécessaire pour des appels bien espacés."""
        # Premier appel
        start_time = time.time()
        self.rate_limiter.enforce_rate_limit("Call1")
        first_call_time = time.time()
        
        # Attendre plus que l'intervalle
        time.sleep(0.15)
        
        # Deuxième appel
        self.rate_limiter.enforce_rate_limit("Call2")
        second_call_time = time.time()
        
        # Le deuxième appel ne devrait pas avoir attendu
        call_duration = second_call_time - (first_call_time + 0.15)
        self.assertLess(call_duration, 0.05)  # Moins de 50ms
        
        # Vérifier les statistiques
        stats = self.rate_limiter.get_statistics()
        self.assertEqual(stats['blocked_requests'], 0)  # Aucun appel bloqué


if __name__ == '__main__':
    unittest.main()