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
        prompt_truncated = prompt[:20000] if len(prompt) > 20000 else prompt
        
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
            if not keywords or len(keywords) > 500:  # Validation basique
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
    
    def summarize_conversation(self, conversation_text: str) -> str:
        """
        Résume un historique de conversation multi-agents pour compression mémoire.
        
        Args:
            conversation_text: Historique de conversation à compresser
            
        Returns:
            str: Conversation résumée, ou conversation originale si pas besoin/service désactivé
        """
        if not self.summary_enabled:
            return conversation_text
        
        # Vérifier que la conversation n'est pas vide
        if not conversation_text or len(conversation_text) < 100:
            return conversation_text
        
        # Tronquer si dépasse la limite d'entrée
        context_truncated = conversation_text[:self.summary_max_input] if len(conversation_text) > self.summary_max_input else conversation_text
        
        conversation_prompt = f'''Résume cette conversation multi-agents en préservant les informations techniques essentielles pour la continuité du développement.

Conversation à résumer:
{context_truncated}

Instructions de compression équilibrée:
- GARDE ABSOLUMENT: spécifications techniques, schémas de données, noms de colonnes, formats JSON, API endpoints, contraintes projet
- Garde aussi: décisions prises, jalons approuvés, fichiers créés/modifiés, problèmes résolus, découvertes importantes
- Supprime: détails de debugging, répétitions, échanges de validation routine, thinking blocks verbeux
- Format: puces détaillées avec contexte technique préservé
- Maximum {self.summary_max_output // 1.5} caractères (compression modérée)
- PRIORITÉ ABSOLUE: préserver tous les détails techniques, schémas, formats de données, contraintes

Résumé technique détaillé:'''

        try:
            llm = LLMFactory.create(model=self.summary_model)
            
            result = llm.generate(
                prompt=conversation_prompt,
                max_tokens=self.summary_max_output // 3,  # Compression modérée (vs //6)
                temperature=0.2  # Légèrement plus créatif pour préserver détails
            )
            
            summary = result.strip()
            
            # Validation moins stricte pour préserver plus de contenu
            if len(summary) < 100:
                self.logger.warning("Résumé conversation trop court, utilisation original")
                return conversation_text
            
            self.logger.debug(f"Conversation résumée: {len(conversation_text)} -> {len(summary)} caractères")
            return summary
            
        except Exception as e:
            self.logger.error(f"Erreur résumé conversation: {str(e)}")
            return conversation_text  # Fallback vers conversation originale
    
    def summarize_constraints(self, project_charter: str, task_description: str) -> str:
        """
        CYCLE COGNITIF HYBRIDE - Phase d'Alignement
        Extrait les contraintes critiques du Project Charter pour une tâche spécifique.
        
        Args:
            project_charter: Contenu du Project Charter complet
            task_description: Description de la tâche spécifique à accomplir
            
        Returns:
            str: contraintes critiques ciblées pour la tâche
        """
        self.logger.debug(f"Génération contraintes avec modèle léger démarrée pour tâche: {task_description[:50]}...")
        
        if not self.keyword_enabled:
            # Service désactivé - retour Charter tronqué
            self.logger.info("LightweightLLM désactivé, utilisation Charter tronqué")
            return f"Directive projet: {project_charter[:300]}..."
        
        if not project_charter or not task_description:
            self.logger.warning("Project Charter ou tâche vide pour summarize_constraints")
            return "Aucune directive de projet disponible."
        
        alignment_prompt = f"""Extrait les  contraintes du Project Charter suivant pour accomplir la tâche spécifique ci-dessous.

PROJECT CHARTER:
{project_charter}

TÂCHE SPÉCIFIQUE:
{task_description}

Instructions:
- Identifie uniquement les contraintes CRITIQUES pour cette tâche
- Maximum 3 points, format liste
- Privilégie: domaine métier, stack technique, livrables clés
- Ignore les détails généraux
- RÉPONSE DIRECTE UNIQUEMENT: pas d'explication, juste les contraintes

Contraintes critiques:"""
        
        try:
            # Génération avec modèle léger et rapide
            llm = LLMFactory.create(model=self.keyword_model)
            result = llm.generate(
                prompt=alignment_prompt,
                max_tokens=800,  # Suffisant pour contraintes détaillées
                temperature=0.1   # Très factuel
            )
            
            # Validation et nettoyage
            constraints = result.strip()
            if not constraints:
                self.logger.warning("Résultat d'alignement vide")
                return f"Directive projet: {project_charter[:200]}..."  # Fallback Charter tronqué
            
            # Validation basique : contenu généré valide
            self.logger.debug(f"Contraintes alignées générées: {len(constraints)} caractères")
            return constraints
                
        except Exception as e:
            self.logger.error(f"Erreur phase d'alignement: {str(e)}")
            return f"Directive projet: {project_charter[:200]}..."  # Fallback robuste


# Instance globale partagée (singleton simple)
_lightweight_service = None


def get_lightweight_llm_service(project_name: str = "System") -> LightweightLLMService:
    """Récupère l'instance partagée du service LLM léger."""
    global _lightweight_service
    if _lightweight_service is None:
        _lightweight_service = LightweightLLMService(project_name)
    return _lightweight_service