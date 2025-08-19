"""
Gestionnaire de rate limiting global pour tous les appels API.
Coordination centralisée pour éviter les dépassements de quotas.
"""

import time
import threading
from typing import Optional
import logging
from datetime import datetime

from config import default_config


class GlobalRateLimiter:
    """
    Gestionnaire de rate limiting global pour tous les appels API.
    Thread-safe et partagé entre tous les connecteurs LLM.
    """
    
    # Instance singleton
    _instance: Optional['GlobalRateLimiter'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Pattern singleton thread-safe."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialise le rate limiter global."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.logger = logging.getLogger("GlobalRateLimiter")
        
        # Configuration du rate limiting
        self._global_lock = threading.Lock()
        self._last_request_time = time.time()  # Initialiser avec timestamp actuel
        
        # Interval global entre les requêtes (configurable)
        # Valeur conservative pour éviter les 429
        self._request_interval = self._get_rate_limit_interval()
        
        # Statistiques
        self._total_requests = 0
        self._total_wait_time = 0.0
        self._blocked_requests = 0
        
        self.logger.info(f"GlobalRateLimiter initialisé avec interval {self._request_interval}s")
    
    def _get_rate_limit_interval(self) -> float:
        """Récupère l'intervalle de rate limiting depuis la config."""
        try:
            # Essayer de récupérer depuis la config
            return float(default_config.get('general', {}).get('api_rate_limit_interval'))
        except:
            # Valeur par défaut sécurisée
            return 1.5
    
    def enforce_rate_limit(self, connector_name: str = "Unknown") -> None:
        """
        Applique le rate limiting global pour tous les appels API.
        Corrigé pour éviter les race conditions.
        
        Args:
            connector_name: Nom du connecteur pour logging (optionnel)
        """
        with self._global_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            self._total_requests += 1
            
            if time_since_last < self._request_interval:
                # Calcul du temps d'attente nécessaire
                sleep_duration = self._request_interval - time_since_last
                
                self._blocked_requests += 1
                self._total_wait_time += sleep_duration
                
                self.logger.debug(
                    f"Global API rate limit: {connector_name} waiting {sleep_duration:.2f}s "
                    f"(last request {time_since_last:.2f}s ago)"
                )
                
                # Mettre à jour AVANT le sleep pour éviter que d'autres threads
                # calculent le même temps d'attente
                self._last_request_time = current_time + sleep_duration
                
                # Libérer le verrou pendant le sleep pour permettre aux autres d'attendre
                self._global_lock.release()
                try:
                    time.sleep(sleep_duration)
                finally:
                    self._global_lock.acquire()
            else:
                # Pas de sleep, utiliser le timestamp actuel
                self._last_request_time = current_time
    
    def get_statistics(self) -> dict:
        """Retourne les statistiques de rate limiting."""
        with self._global_lock:
            total_time = time.time() - self._last_request_time if self._total_requests > 0 else 0
            
            return {
                'total_requests': self._total_requests,
                'blocked_requests': self._blocked_requests,
                'total_wait_time': self._total_wait_time,
                'avg_wait_time': self._total_wait_time / max(self._blocked_requests, 1),
                'block_rate': self._blocked_requests / max(self._total_requests, 1),
                'current_interval': self._request_interval,
                'uptime': total_time
            }
    
    def reset_statistics(self) -> None:
        """Reset les statistiques (utile pour les tests)."""
        with self._global_lock:
            self._total_requests = 0
            self._total_wait_time = 0.0
            self._blocked_requests = 0
            self._last_request_time = 0.0
    
    def update_rate_limit_interval(self, new_interval: float) -> None:
        """Met à jour l'intervalle de rate limiting."""
        with self._global_lock:
            old_interval = self._request_interval
            self._request_interval = new_interval
            self.logger.info(f"Rate limit interval updated: {old_interval}s -> {new_interval}s")
    
    @classmethod
    def get_instance(cls) -> 'GlobalRateLimiter':
        """Récupère l'instance singleton."""
        return cls()


# Instance globale accessible
global_rate_limiter = GlobalRateLimiter()