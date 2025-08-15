"""
Service LLM léger pour extraction de mots-clés et résumé de contexte.
Utilisé par le système RAG pour améliorer la qualité des recherches.
"""

from typing import Optional
from core.llm_connector import LLMFactory
from config import default_config
from utils.logger import get_project_logger


class LightweightLLMService:
    """
    Service utilitaire pour tâches de synthèse rapides avec LLM léger.
    Pas d'état, pas de RAG - juste génération directe.
    """
    
    def __init__(self, project_name: str = "System"):
        self.logger = get_project_logger(project_name, "LightweightLLM")
        self.config = default_config.get('rag', {}).get('auto_context_injection', {})
        
        # Configuration extraction de mots-clés
        self.keyword_config = self.config.get('keyword_extraction', {})
        self.keyword_enabled = self.keyword_config.get('enabled', False)
        self.keyword_model = self.keyword_config.get('model', 'mistral-small-latest')
        self.keyword_max_tokens = self.keyword_config.get('max_tokens', 30)
        self.keyword_temperature = self.keyword_config.get('temperature', 0.1)
        self.keyword_timeout = self.keyword_config.get('timeout', 10)
        
        # Configuration résumé de contexte
        self.summary_config = self.config.get('context_summarization', {})
        self.summary_enabled = self.summary_config.get('enabled', False)
        self.summary_model = self.summary_config.get('model', 'mistral-small-latest')
        self.summary_max_input = self.summary_config.get('max_input_length', 25000)
        self.summary_max_output = self.summary_config.get('max_output_length', 2000)
        self.summary_temperature = self.summary_config.get('temperature', 0.2)
        self.summary_timeout = self.summary_config.get('timeout', 15)
        
        self.logger.info(f"LightweightLLMService initialisé - Keywords: {self.keyword_enabled}, Summary: {self.summary_enabled}")
    
    def extract_keywords(self, prompt: str) -> str:
        """
        Extrait des mots-clés intelligents pour recherche RAG.
        
        Args:
            prompt: Prompt original de l'agent
            
        Returns:
            str: Mots-clés séparés par espaces, ou prompt original si service désactivé
        """
        if not self.keyword_enabled:
            self.logger.debug("Extraction de mots-clés désactivée, utilisation prompt original")
            return prompt[:100]  # Fallback simple
        
        if not prompt or len(prompt) < 10:
            return ""
        
        # Tronquer si trop long pour éviter les timeouts
        prompt_truncated = prompt[:2000] if len(prompt) > 2000 else prompt
        
        extraction_prompt = f'''Extrait 3-5 mots-clés techniques pour recherche dans une base de connaissances de développement logiciel.

Prompt: "{prompt_truncated}"

Règles:
- Privilégie les termes techniques et spécialisés
- Ignore les mots vides (le, la, avec, pour, etc.)
- Ignore les verbes génériques (faire, créer, analyser)
- Priorise: technologies, patterns, concepts métier
- Format: mots séparés par espaces, minuscules
- RÉPONSE DIRECTE UNIQUEMENT: pas d'explication, juste les mots-clés

Mots-clés:'''
        
        try:
            llm = LLMFactory.create(model=self.keyword_model)
            
            result = llm.generate(
                prompt=extraction_prompt,
                max_tokens=self.keyword_max_tokens,
                temperature=self.keyword_temperature
            )
            
            # Nettoyer et valider le résultat
            keywords = result.strip().lower()
            if not keywords or len(keywords) > 200:  # Validation basique
                self.logger.warning(f"Résultat d'extraction invalide: '{keywords[:50]}...'")
                return prompt[:100]  # Fallback
            
            self.logger.debug(f"Mots-clés extraits: '{keywords}'")
            return keywords
            
        except Exception as e:
            self.logger.error(f"Erreur extraction mots-clés: {str(e)}")
            return prompt[:100]  # Fallback
    
    def summarize_context(self, context: str) -> str:
        """
        Résume un contexte long pour injection RAG optimisée.
        
        Args:
            context: Contexte original (potentiellement très long)
            
        Returns:
            str: Contexte résumé, ou contexte original si pas besoin/service désactivé
        """
        if not self.summary_enabled:
            return context
        
        # Vérifier que le contexte n'est pas vide
        if not context or len(context) < 50:
            return context
        
        # Tronquer si dépasse la limite d'entrée
        context_truncated = context[:self.summary_max_input] if len(context) > self.summary_max_input else context
        
        summarization_prompt = f'''Résume ce contexte technique en gardant uniquement les informations essentielles pour un agent de développement.

Contexte original:
{context_truncated}

Consignes:
- Garde les spécifications techniques précises
- Conserve les exemples de code importants  
- Supprime les répétitions et détails non-critiques
- Structure en points clés avec sources
- Maximum {self.summary_max_output} caractères
- RÉPONSE DIRECTE UNIQUEMENT: pas de préambule comme "Voici le résumé:", juste le contenu résumé

Résumé structuré:'''
        
        try:
            llm = LLMFactory.create(model=self.summary_model)
            
            result = llm.generate(
                prompt=summarization_prompt,
                max_tokens=self.summary_max_output // 3,  # Estimation tokens ≈ chars/3
                temperature=self.summary_temperature
            )
            
            summary = result.strip()
            
            # Validation basique
            if len(summary) < 50:
                self.logger.warning("Résumé trop court, utilisation contexte original")
                return context
            
            self.logger.debug(f"Contexte résumé: {len(context)} -> {len(summary)} caractères")
            return summary
            
        except Exception as e:
            self.logger.error(f"Erreur résumé contexte: {str(e)}")
            return context  # Fallback vers contexte original


# Instance globale partagée (singleton simple)
_lightweight_service = None


def get_lightweight_llm_service(project_name: str = "System") -> LightweightLLMService:
    """Récupère l'instance partagée du service LLM léger."""
    global _lightweight_service
    if _lightweight_service is None:
        _lightweight_service = LightweightLLMService(project_name)
    return _lightweight_service