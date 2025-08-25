"""
Connecteur unifié pour les différents LLMs.
Supporte Mistral (par défaut) et DeepSeek.
"""

import os
import time
import json
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import httpx

import numpy as np
from mistralai import Mistral

from dotenv import load_dotenv

load_dotenv()

from utils.logger import setup_logger, log_llm_interaction, log_llm_complete_exchange
from config import default_config
from core.global_rate_limiter import global_rate_limiter

# Créer un verrou global unique pour tous les appels LLM
_llm_execution_lock = threading.Lock()


class LLMConnector(ABC):
    """
    Classe abstraite pour les connecteurs LLM.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Génère une réponse à partir d'un prompt.
        
        Args:
            prompt: Prompt utilisateur
            system_prompt: Prompt système (optionnel)
            **kwargs: Paramètres supplémentaires pour l'API (temperature, max_tokens, etc.)
            
        Returns:
            str: Réponse générée
        """
        pass

    @abstractmethod
    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        agent_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Génère avec historique de messages."""
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Génère une réponse JSON structurée.
        
        Args:
            prompt: Prompt utilisateur
            system_prompt: Prompt système
            schema: Schéma JSON attendu (pour validation)
            **kwargs: Paramètres supplémentaires pour l'API (temperature, max_tokens, etc.)
            
        Returns:
            Dict: Réponse JSON parsée
        """
        pass


class MistralConnector(LLMConnector):
    """
    Connecteur pour l'API Mistral.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialise le connecteur Mistral.
        
        Args:
            api_key: Clé API Mistral (utilise l'env var si non fournie)
            model: Modèle à utiliser (utilise le défaut si non fourni)
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Clé API Mistral non trouvée. Définissez MISTRAL_API_KEY dans .env")
        
        self.client = Mistral(api_key=self.api_key)
        self.model = model or default_config['llm']['default_model']
        self.logger = setup_logger("MistralConnector")

        # Configuration des modèles Mistral
        self.model_configs = {
            model_info['name']: model_info 
            for model_info in default_config['llm']['models']['mistral']['models']
        }

    def _enforce_rate_limit(self):
        """Applique le rate limiting global pour tous les appels API."""
        global_rate_limiter.enforce_rate_limit(f"MistralConnector({self.model})")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Génère une réponse avec Mistral.
        Gère la priorité des paramètres : appel > config > défaut.
        """
        
        with _llm_execution_lock:
        
            start_time = time.time()
            
            # Fusion des paramètres de configuration
            # Commence avec les paramètres LLM globaux de la config
            params = default_config['llm'].copy()
            # Retire les clés de configuration qui ne sont pas des paramètres d'API
            params.pop('models', None)
            params.pop('model_preferences', None)
            params.pop('default_model', None)

            # 2. On fusionne avec les paramètres spécifiques au modèle, qui ont la priorité.
            model_specific_params = self.model_configs.get(self.model, {}).copy()
            model_specific_params.pop('name', None) # Retirer la clé 'name'
            params.update(model_specific_params)

            # 3. On fusionne avec les paramètres de l'appel (kwargs), qui ont la priorité la plus élevée.
            params.update(kwargs)




            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            try:
                # Retry logic
                max_retries = default_config['general']['max_retries']
                retry_delay = default_config['general']['retry_delay']
                
                for attempt in range(max_retries):
                    try:
                        # Appliquer le rate limiting global avant la requête
                        self._enforce_rate_limit()
                        
                        # On passe les paramètres fusionnés directement à l'API
                        response = self.client.chat.complete(
                            model=self.model,
                            messages=messages,
                            **params
                        )
                        
                        result = response.choices[0].message.content
                        duration = time.time() - start_time
                        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
                        
                        # LOG COMPLET ET EXHAUSTIF
                        log_llm_complete_exchange(
                            agent_name=agent_context.get('agent_name', 'Unknown') if agent_context else 'Unknown',
                            model=self.model,
                            messages=messages,
                            response=result,
                            parameters=params,
                            tokens_used=tokens_used,
                            duration=duration,
                            context=agent_context or {}
                        )
                        
                        # Log l'interaction (ancien système pour compatibilité)
                        log_llm_interaction(
                            self.logger,
                            prompt=prompt,
                            response=result,
                            model=self.model,
                            tokens_used=tokens_used,
                            duration=duration
                        )
                        

                        return result
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Tentative {attempt + 1} échouée: {str(e)}")
                            time.sleep(retry_delay)
                        else:
                            raise
                            
            except Exception as e:
                self.logger.error(f"Erreur lors de la génération Mistral: {str(e)}")
                raise

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        agent_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Génération avec historique complet."""

        with _llm_execution_lock:

            start_time = time.time()
            
            params = self.model_configs.get(self.model, {}).copy()
            params.pop('name', None)
            params.update(kwargs)
            
            try:
                # Retry logic
                max_retries = default_config['general']['max_retries']
                retry_delay = default_config['general']['retry_delay']
                
                for attempt in range(max_retries):
                    try:
                        # Appliquer le rate limiting global avant la requête
                        self._enforce_rate_limit()
                        
                        response = self.client.chat.complete(
                            model=self.model,
                            messages=messages,
                            **params
                        )
                        
                        result = response.choices[0].message.content
                        duration = time.time() - start_time
                        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
                        
                        # LOG COMPLET ET EXHAUSTIF
                        log_llm_complete_exchange(
                            agent_name=agent_context.get('agent_name', 'Unknown') if agent_context else 'Unknown',
                            model=self.model,
                            messages=messages,
                            response=result,
                            parameters=params,
                            tokens_used=tokens_used,
                            duration=duration,
                            context=agent_context or {}
                        )
                        
                        return result
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Tentative {attempt + 1} échouée: {str(e)}")
                            time.sleep(retry_delay)
                        else:
                            raise
                            
            except Exception as e:
                self.logger.error(f"Erreur lors de la génération Mistral avec messages: {str(e)}")
                raise
    

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Génère une réponse JSON avec Mistral.
        """
        # Enrichir le prompt pour demander du JSON
        json_prompt = f"{prompt}\n\nRéponds uniquement avec un JSON valide, sans texte avant ou après."
        
        if schema:
            json_prompt += f"\n\nLe JSON doit suivre ce schéma:\n{json.dumps(schema, indent=2)}"
        
        # Ajouter une instruction système pour le JSON
        if system_prompt:
            system_prompt += "\nTu dois toujours répondre avec du JSON valide uniquement."
        else:
            system_prompt = "Tu es un assistant qui répond toujours avec du JSON valide uniquement, sans texte supplémentaire."
        
        # Définir une température basse par défaut pour le JSON, sauf si surchargée.
        kwargs.setdefault('temperature', 0.3)
        
        response = self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # Nettoyer la réponse et parser le JSON
        try:
            # Retirer les éventuels marqueurs de code
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            
            # Valider contre le schéma si fourni
            if schema:
                self._validate_json_schema(result, schema)
            
            #time.sleep(1)
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Erreur de parsing JSON: {str(e)}")
            self.logger.debug(f"Réponse brute: {response}")
            # Essayer de réparer le JSON
            return self._try_repair_json(response)
    
    def _validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """
        Validation basique du schéma JSON.
        """
        # Implémentation simple - vérifier les clés requises
        if 'required' in schema:
            for key in schema['required']:
                if key not in data:
                    raise ValueError(f"Clé requise manquante: {key}")
    
    def _try_repair_json(self, response: str) -> Dict[str, Any]:
        """
        Essaie de réparer un JSON invalide.
        """
        # Stratégies de réparation simples
        attempts = [
            response,
            response.replace("'", '"'),  # Remplacer quotes simples
            '{"error": "Invalid JSON", "raw": ' + json.dumps(response) + '}',
        ]
        
        for attempt in attempts:
            try:
                return json.loads(attempt)
            except:
                continue
        
        # Si tout échoue, retourner une erreur structurée
        return {"error": "Could not parse JSON", "raw_response": response}




class MistralEmbedConnector:
    """Connecteur pour l'API Mistral Embed."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Clé API Mistral non trouvée. Définissez MISTRAL_API_KEY dans .env")
        
        self.client = Mistral(api_key=self.api_key)
        self.logger = setup_logger("MistralEmbedConnector")
        self.embedding_dimension = 1024  # Dimension fixe de Mistral Embed

    def _enforce_rate_limit(self):
        """Applique le rate limiting global pour tous les appels API."""
        global_rate_limiter.enforce_rate_limit("MistralEmbedConnector")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Génère des embeddings pour une liste de textes."""
        try:
            # Retry logic
            max_retries = default_config['general']['max_retries']
            retry_delay = default_config['general']['retry_delay']
            
            for attempt in range(max_retries):
                try:
                    # Appliquer le rate limiting global avant la requête
                    self._enforce_rate_limit()
                                
                    response = self.client.embeddings.create(
                        model="mistral-embed",
                        inputs=texts
                    )
                    
                    # Convertir en numpy array
                    embeddings = np.array([embedding.embedding for embedding in response.data])
                    
                    # Normaliser pour cosinus similarity
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    normalized_embeddings = embeddings / norms
                    
                    return normalized_embeddings
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Tentative {attempt + 1} échouée pour embeddings: {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        raise
                        
        except Exception as e:
            self.logger.error(f"Erreur Mistral Embed: {str(e)}")
            raise




class DeepSeekConnector(LLMConnector):
    """
    Connecteur pour l'API DeepSeek.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        """
        Initialise le connecteur DeepSeek.
        
        Args:
            api_key: Clé API DeepSeek
            model: Modèle à utiliser
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Clé API DeepSeek non trouvée. Définissez DEEPSEEK_API_KEY dans .env")
        
        self.base_url = "https://api.deepseek.com/v1"
        self.model = model
        self.logger = setup_logger("DeepSeekConnector")
        
        # Client HTTP
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=float(default_config['general']['llm_timeout'])
        )

    def _enforce_rate_limit(self):
        """Applique le rate limiting global pour tous les appels API."""
        global_rate_limiter.enforce_rate_limit(f"DeepSeekConnector({self.model})")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Génère une réponse avec DeepSeek.
        """
        
        with _llm_execution_lock:
            start_time = time.time()
            
            # Logique de fusion des paramètres
            params = {
                'temperature': 0.7,
            }
            params.update(kwargs)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                **params
            }
            
            try:
                # Retry logic
                max_retries = default_config['general']['max_retries']
                retry_delay = default_config['general']['retry_delay']
                
                for attempt in range(max_retries):
                    try:
                        # Appliquer le rate limiting global avant la requête
                        self._enforce_rate_limit()
                        
                        response = self.client.post(
                            f"{self.base_url}/chat/completions",
                            json=payload
                        )
                        response.raise_for_status()
                        
                        data = response.json()
                        result = data['choices'][0]['message']['content']
                        duration = time.time() - start_time
                        
                        # LOG COMPLET ET EXHAUSTIF
                        log_llm_complete_exchange(
                            agent_name=agent_context.get('agent_name', 'Unknown') if agent_context else 'Unknown',
                            model=self.model,
                            messages=messages,
                            response=result,
                            parameters=params,
                            tokens_used=data.get('usage', {}).get('total_tokens'),
                            duration=duration,
                            context=agent_context or {}
                        )
                        
                        # Log l'interaction (ancien système pour compatibilité)
                        log_llm_interaction(
                            self.logger,
                            prompt=prompt,
                            response=result,
                            model=self.model,
                            tokens_used=data.get('usage', {}).get('total_tokens'),
                            duration=duration
                        )
                        
                        return result
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Tentative {attempt + 1} échouée: {str(e)}")
                            time.sleep(retry_delay)
                        else:
                            raise
                            
            except Exception as e:
                self.logger.error(f"Erreur lors de la génération DeepSeek: {str(e)}")
                raise


    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        agent_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """DeepSeek avec historique."""
        
        with _llm_execution_lock:
            start_time = time.time()
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                **kwargs
            }
            
            try:
                # Retry logic
                max_retries = default_config['general']['max_retries']
                retry_delay = default_config['general']['retry_delay']
                
                for attempt in range(max_retries):
                    try:
                        # Appliquer le rate limiting global avant la requête
                        self._enforce_rate_limit()
                        
                        response = self.client.post(
                            f"{self.base_url}/chat/completions",
                            json=payload
                        )
                        response.raise_for_status()
                        
                        data = response.json()
                        result = data['choices'][0]['message']['content']
                        duration = time.time() - start_time
                        
                        # LOG COMPLET ET EXHAUSTIF
                        log_llm_complete_exchange(
                            agent_name=agent_context.get('agent_name', 'Unknown') if agent_context else 'Unknown',
                            model=self.model,
                            messages=messages,
                            response=result,
                            parameters=kwargs,
                            tokens_used=data.get('usage', {}).get('total_tokens'),
                            duration=duration,
                            context=agent_context or {}
                        )
                        
                        return result
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Tentative {attempt + 1} échouée: {str(e)}")
                            time.sleep(retry_delay)
                        else:
                            raise
                            
            except Exception as e:
                self.logger.error(f"Erreur lors de la génération DeepSeek avec messages: {str(e)}")
                raise
    



    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Génère une réponse JSON avec DeepSeek.
        """
        json_prompt = f"{prompt}\n\nRéponds uniquement avec un JSON valide."
        
        if schema:
            json_prompt += f"\n\nSchéma attendu:\n{json.dumps(schema, indent=2)}"
        
        if system_prompt:
            system_prompt += "\nRéponds toujours en JSON valide."
        else:
            system_prompt = "Tu réponds toujours en JSON valide uniquement."
        
        kwargs.setdefault('temperature', 0.3)
        
        response = self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            **kwargs
        )
        
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            return json.loads(cleaned.strip())
            
        except json.JSONDecodeError:
            self.logger.error("Erreur de parsing JSON")
            return {"error": "Invalid JSON", "raw": response}
    
    def __del__(self):
        """Ferme le client HTTP."""
        if hasattr(self, 'client'):
            self.client.close()


class LLMFactory:
    """
    Factory pour créer des connecteurs LLM avec pattern singleton.
    Réutilise les instances existantes pour économiser les ressources.
    """
    
    _connectors = {
        'mistral': MistralConnector,
        'deepseek': DeepSeekConnector,
        'mistral-embed': MistralEmbedConnector
    }
    
    # Registre des instances créées (singleton par modèle)
    _instances = {}
    _lock = threading.Lock()
    
    @classmethod
    def create(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMConnector:
        """
        Crée ou récupère un connecteur LLM (singleton par modèle).
        
        Args:
            provider: Fournisseur (mistral, deepseek)
            model: Modèle spécifique
            **kwargs: Arguments supplémentaires (ignorés pour le cache)
            
        Returns:
            LLMConnector: Instance du connecteur (partagée)
        """
        # Déterminer le provider depuis le modèle si non fourni
        if not provider and model:
            provider = cls._detect_provider(model)
        
        # Utiliser le provider par défaut si non fourni
        if not provider:
            default_model = default_config['llm']['default_model']
            provider = cls._detect_provider(default_model)
            model = model or default_model
        
        if provider not in cls._connectors:
            raise ValueError(f"Provider non supporté: {provider}")
        
        # Clé de cache basée sur le modèle uniquement
        cache_key = f"{provider}:{model}"
        
        # Thread-safe singleton pattern
        with cls._lock:
            if cache_key not in cls._instances:
                connector_class = cls._connectors[provider]
                # Créer une nouvelle instance (adapter les paramètres selon le connecteur)
                if provider == 'mistral-embed':
                    # MistralEmbedConnector ne prend pas de paramètre model
                    cls._instances[cache_key] = connector_class()
                else:
                    # Les autres connecteurs prennent un paramètre model
                    cls._instances[cache_key] = connector_class(model=model)
            
            return cls._instances[cache_key]
    
    @classmethod
    def _detect_provider(cls, model_name: str) -> str:
        """
        Détecte le provider depuis le nom du modèle.
        """
        if 'mistral' in model_name or 'codestral' in model_name or 'magistral' in model_name:
            return 'mistral'
        elif 'deepseek' in model_name:
            return 'deepseek'
        else:
            # Par défaut, utiliser Mistral
            return 'mistral'
    
    @classmethod
    def register_connector(cls, name: str, connector_class: type):
        """
        Enregistre un nouveau connecteur.
        
        Args:
            name: Nom du provider
            connector_class: Classe du connecteur
        """
        cls._connectors[name] = connector_class
    
    @classmethod
    def clear_cache(cls):
        """Nettoie le cache des instances (utile pour les tests)."""
        with cls._lock:
            # Fermer les connexions existantes
            for instance in cls._instances.values():
                if hasattr(instance, 'client') and hasattr(instance.client, 'close'):
                    try:
                        instance.client.close()
                    except:
                        pass  # Ignorer les erreurs de fermeture
            cls._instances.clear()
    
    @classmethod 
    def get_instance_count(cls) -> int:
        """Retourne le nombre d'instances en cache (utile pour les tests)."""
        return len(cls._instances)