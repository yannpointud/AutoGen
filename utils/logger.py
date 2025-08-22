"""
Système de logging centralisé pour la plateforme.
Utilise JSON Lines pour faciliter l'analyse et le RAG.
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Formateur JSON personnalisé pour enrichir les logs.
    """
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """
        Ajoute des champs personnalisés aux logs.
        
        Args:
            log_record: Dictionnaire du log
            record: Record de logging
            message_dict: Dictionnaire du message
        """
        super().add_fields(log_record, record, message_dict)
        
        # Ajouter des métadonnées
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['logger_name'] = record.name
        log_record['level'] = record.levelname
        
        # Ajouter le contexte si disponible
        if hasattr(record, 'agent_name'):
            log_record['agent_name'] = record.agent_name
        if hasattr(record, 'project_name'):
            log_record['project_name'] = record.project_name
        if hasattr(record, 'phase'):
            log_record['phase'] = record.phase


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    project_name: Optional[str] = None
) -> logging.Logger:
    """
    Configure et retourne un logger.
    
    Args:
        name: Nom du logger
        log_file: Fichier de log (optionnel)
        level: Niveau de logging
        project_name: Nom du projet pour les logs spécifiques
        
    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Éviter la duplication des handlers
    if logger.handlers:
        return logger
    
    # Formatter JSON
    json_formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    
    # Handler pour la console (format lisible)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Handler pour fichier si spécifié
    if log_file:
        # Créer le dossier si nécessaire
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotation des logs (10MB max, 5 fichiers de backup)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    # Handler global pour tous les logs
    global_log_dir = Path("logs")
    global_log_dir.mkdir(exist_ok=True)
    
    global_log_file = global_log_dir / f"platform_{datetime.now().strftime('%Y%m%d')}.jsonl"
    global_handler = RotatingFileHandler(
        global_log_file,
        maxBytes=50*1024*1024,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    global_handler.setFormatter(json_formatter)
    global_handler.setLevel(logging.DEBUG)  # Capturer tout dans le log global
    logger.addHandler(global_handler)
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    Adaptateur pour ajouter du contexte aux logs.
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        """
        Initialise l'adaptateur.
        
        Args:
            logger: Logger de base
            extra: Contexte supplémentaire
        """
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Ajoute le contexte aux logs.
        
        Args:
            msg: Message de log
            kwargs: Arguments du log
            
        Returns:
            tuple: Message et kwargs enrichis
        """
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        for key, value in self.extra.items():
            kwargs['extra'][key] = value
        
        return msg, kwargs


def get_project_logger(project_name: str, agent_name: Optional[str] = None) -> LoggerAdapter:
    """
    Retourne un logger configuré pour un projet spécifique.
    
    Args:
        project_name: Nom du projet
        agent_name: Nom de l'agent (optionnel)
        
    Returns:
        LoggerAdapter: Logger avec contexte
    """
    # Chemin du log du projet
    project_log_dir = Path("projects") / project_name / "logs"
    project_log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = project_log_dir / f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    # Créer le logger de base
    logger_name = f"{project_name}.{agent_name}" if agent_name else project_name
    base_logger = setup_logger(logger_name, log_file)
    
    # Contexte
    context = {
        'project_name': project_name
    }
    if agent_name:
        context['agent_name'] = agent_name
    
    return LoggerAdapter(base_logger, context)


def log_llm_interaction(
    logger: logging.Logger,
    prompt: str,
    response: str,
    model: str,
    tokens_used: Optional[int] = None,
    duration: Optional[float] = None
) -> None:
    """
    Log une interaction avec un LLM.
    
    Args:
        logger: Logger à utiliser
        prompt: Prompt envoyé
        response: Réponse reçue
        model: Modèle utilisé
        tokens_used: Nombre de tokens utilisés
        duration: Durée de la requête
    """
    extra = {
        'llm_model': model,
        'prompt_length': len(prompt),
        'response_length': len(response),
        'interaction_type': 'llm_call'
    }
    
    if tokens_used:
        extra['tokens_used'] = tokens_used
    if duration:
        extra['duration_seconds'] = duration
    
    logger.info(
        f"LLM interaction with {model}",
        extra=extra
    )
    
    # Log détaillé en debug
    logger.debug(
        f"LLM prompt: {prompt[:200]}...",
        extra={'full_prompt': prompt}
    )
    logger.debug(
        f"LLM response: {response[:200]}...",
        extra={'full_response': response}
    )


def log_llm_complete_exchange(
    agent_name: str,
    model: str,
    messages: list[Dict[str, str]],
    response: str,
    parameters: Dict[str, Any],
    tokens_used: Optional[int] = None,
    duration: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log COMPLET et EXHAUSTIF d'un échange LLM avec tous les détails.
    
    Args:
        agent_name: Nom de l'agent qui fait l'appel
        model: Modèle LLM utilisé
        messages: Messages complets envoyés au LLM (system, user, assistant)
        response: Réponse complète du LLM
        parameters: Tous les paramètres de génération (temperature, max_tokens, etc.)
        tokens_used: Nombre de tokens utilisés
        duration: Durée de la requête en secondes
        context: Contexte additionnel (milestone, task_id, etc.)
    """
    # Logger spécialisé pour les échanges LLM DEBUG
    # Extraire le nom du projet du contexte si disponible
    project_name = context.get('project_name') if context else None
    llm_debug_logger = setup_llm_debug_logger(project_name)
    
    # Compteur séquentiel global pour tracer l'ordre des appels
    if not hasattr(log_llm_complete_exchange, '_sequence_counter'):
        log_llm_complete_exchange._sequence_counter = 0
    log_llm_complete_exchange._sequence_counter += 1
    
    # Préparer l'entrée de log complète
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'sequence_id': log_llm_complete_exchange._sequence_counter,
        'agent_name': agent_name,
        'model': model,
        'direction': 'REQUEST',
        'messages_sent': messages,  # TOUT le contenu envoyé
        'parameters': parameters,   # TOUS les paramètres
        'prompt_total_length': sum(len(msg.get('content', '')) for msg in messages),
        'context': context or {}
    }
    
    # Log de la DEMANDE (ce qui est envoyé au LLM)
    # Utiliser DEBUG pour s'assurer que c'est loggué même si le niveau global est INFO
    prompt_length = log_entry['prompt_total_length']
    estimated_tokens = prompt_length // 3
    llm_debug_logger.debug(
        f"🚀 LLM REQUEST | Agent: {agent_name} | Model: {model} | Chars: {prompt_length} | Est.Tokens: {estimated_tokens}",
        extra=log_entry
    )
    
    # Log de la RÉPONSE (ce qui est reçu du LLM)
    response_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'sequence_id': log_llm_complete_exchange._sequence_counter,
        'agent_name': agent_name,
        'model': model,
        'direction': 'RESPONSE',
        'response_content': response,  # TOUTE la réponse
        'response_length': len(response),
        'tokens_used': tokens_used,
        'duration_seconds': duration,
        'context': context or {}
    }
    
    llm_debug_logger.debug(
        f"📥 LLM RESPONSE | Agent: {agent_name} | Model: {model} | Tokens: {tokens_used} | Duration: {duration:.2f}s",
        extra=response_entry
    )


def setup_llm_debug_logger(project_name: Optional[str] = None) -> logging.Logger:
    """
    Crée un logger spécialisé pour les échanges LLM DEBUG.
    Logs séparés pour faciliter l'analyse.
    
    Args:
        project_name: Nom du projet pour logs spécifiques au projet
    """
    from config import default_config
    
    logger_name = f"LLM_DEBUG_{project_name}" if project_name else "LLM_DEBUG"
    
    # Éviter la duplication
    if logger_name in logging.Logger.manager.loggerDict:
        return logging.getLogger(logger_name)
    
    logger = logging.getLogger(logger_name)
    
    # Respecter le niveau de log de la configuration
    config_log_level = default_config.get('monitoring', {}).get('log_level', 'INFO')
    level = getattr(logging, config_log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Formatter JSON spécialisé pour LLM
    json_formatter = CustomJsonFormatter(
        '%(timestamp)s %(sequence_id)s %(agent_name)s %(direction)s %(message)s'
    )
    
    # Handler pour fichier dédié aux échanges LLM
    if project_name:
        # Logs dans le dossier du projet
        llm_debug_dir = Path("projects") / project_name / "logs" / "llm_debug"
    else:
        # Logs globaux si pas de projet spécifique
        llm_debug_dir = Path("logs/llm_debug")
    
    llm_debug_dir.mkdir(parents=True, exist_ok=True)
    
    llm_debug_file = llm_debug_dir / f"llm_exchanges_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    file_handler = RotatingFileHandler(
        llm_debug_file,
        maxBytes=100*1024*1024,  # 100MB par fichier
        backupCount=20,
        encoding='utf-8'
    )
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    # Handler console pour voir en temps réel (si DEBUG activé)
    if level <= logging.DEBUG:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - LLM_DEBUG - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
    
    return logger


def parse_json_logs(log_file: Path) -> list[Dict[str, Any]]:
    """
    Parse un fichier de logs JSON Lines.
    
    Args:
        log_file: Chemin du fichier de logs
        
    Returns:
        list: Liste des entrées de log
    """
    logs = []
    
    if not log_file.exists():
        return logs
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                logs.append(log_entry)
            except json.JSONDecodeError:
                continue
    
    return logs
