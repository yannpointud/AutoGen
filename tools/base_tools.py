"""
Outils communs à tous les agents (extraits de base_agent).
"""

from typing import Dict, Any
from datetime import datetime

from agents.base_agent import ToolResult


def _analyze_manual_content_for_completion(content: str) -> str:
    """Analyse sémantique d'un rapport manuel pour déterminer le self_assessment."""
    content_lower = content.lower()
    
    # Mots-clés de succès/completion
    success_keywords = ['terminé', 'succès', 'réussi', 'complété', '100%', 'fini', 'achevé', 'complet']
    error_keywords = ['erreur', 'échec', 'problème', 'impossible', 'failed', 'échoué', 'bloqué']
    
    if any(word in content_lower for word in error_keywords):
        return 'failed'
    elif any(word in content_lower for word in success_keywords):
        return 'compliant'
    else:
        return 'unknown'  # Cas ambigus


def _generate_message_from_structured_report(structured_report: Dict[str, Any]) -> str:
    """Génère un message descriptif depuis un rapport structuré."""
    self_assessment = structured_report.get('self_assessment', 'unknown')
    artifacts_count = len(structured_report.get('artifacts_created', []))
    
    if self_assessment == 'compliant':
        return f"Tâche terminée avec succès ({artifacts_count} artefacts créés)"
    elif self_assessment == 'partial':
        return f"Tâche partiellement terminée ({artifacts_count} artefacts créés)"
    elif self_assessment == 'failed':
        return "Tâche échouée - aucun artefact créé"
    else:
        return f"État de la tâche indéterminé ({artifacts_count} artefacts créés)"


def tool_search_context(agent, parameters: Dict[str, Any]) -> ToolResult:
    """Recherche du contexte pertinent dans le RAG."""
    try:
        if not agent.rag_engine:
            return ToolResult('error', error="RAG engine non disponible")
        
        query = parameters.get('query', '')
        top_k = int(parameters.get('top_k', 5))
        
        results = agent.rag_engine.search(query, top_k=top_k)
        
        # Formater les résultats
        formatted_results = []
        for r in results:
            formatted_results.append({
                'text': r.get('chunk_text', '')[:200],
                'source': r.get('source', 'unknown'),
                'score': r.get('score', 0)
            })
        
        return ToolResult('success', result=formatted_results)
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_send_message_to_agent(agent, parameters: Dict[str, Any]) -> ToolResult:
    """Envoie un message/question à un autre agent."""
    try:
        if not agent.communication_enabled:
            return ToolResult('error', error="Communication inter-agents désactivée")
        
        agent_name = parameters.get('agent_name', '').lower()
        message = parameters.get('message', '')
        
        task_id = agent.state.get('current_task_id', 'unknown')
        
        # Vérifier la limite d'échanges
        exchanges_count = agent.current_exchanges.get(task_id, 0)
        if exchanges_count >= agent.max_exchanges:
            return ToolResult('error', error=f"Limite d'échanges atteinte ({agent.max_exchanges})")
        
        # Obtenir l'agent via le superviseur
        if not agent.supervisor:
            return ToolResult('error', error="Pas de superviseur pour la communication")
        
        colleague = agent.supervisor.get_agent(agent_name)
        if not colleague:
            return ToolResult('error', error=f"Agent {agent_name} non trouvé")
        
        # Envoyer le message
        response = colleague.answer_colleague(agent.name, message)
        
        # Mettre à jour le compteur
        agent.current_exchanges[task_id] = exchanges_count + 1
        
        return ToolResult('success', result={'response': response, 'from': agent_name})
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_share_discovery(agent, parameters: Dict[str, Any]) -> ToolResult:
    """Partage une découverte importante dans la mémoire de travail."""
    try:
        discovery = parameters.get('discovery', '')
        importance = parameters.get('importance', 'normal')
        
        if not agent.rag_engine:
            return ToolResult('error', error="RAG engine non disponible")
        
        # Valider l'importance
        valid_importance = ['low', 'normal', 'high', 'critical']
        if importance not in valid_importance:
            importance = 'normal'
        
        # Créer le message
        prefix = {
            'critical': '🚨 CRITIQUE',
            'high': '⚠️ IMPORTANT',
            'normal': 'ℹ️ Info',
            'low': '💡 Note'
        }.get(importance, 'ℹ️ Info')
        
        message = f"{prefix} - {agent.name}: {discovery}"
        
        # Indexer dans la mémoire de travail
        agent.rag_engine.index_to_working_memory(
            message,
            {
                'type': 'discovery',
                'agent_name': agent.name,
                'importance': importance,
                'milestone': agent.current_milestone_id or 'unknown'
            }
        )
        
        return ToolResult('success', result={'discovery_shared': True})
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_report_to_supervisor(agent, parameters: Dict[str, Any]) -> ToolResult:
    """Envoie un rapport au superviseur."""
    try:
        if not agent.supervisor:
            return ToolResult('error', error="Pas de superviseur assigné")
        
        report_type = parameters.get('report_type', 'progress')
        content = parameters.get('content', {})
        
        # Normaliser le format du content
        if isinstance(content, str):
            # Format manuel → Normaliser
            normalized_content = {
                'type': 'manual',
                'message': content,
                'self_assessment': _analyze_manual_content_for_completion(content)
            }
        else:
            # Format automatique → Ajouter type et message si manquants
            normalized_content = dict(content)  # Copie
            normalized_content['type'] = 'automatic'
            if 'message' not in normalized_content:
                normalized_content['message'] = _generate_message_from_structured_report(normalized_content)
        
        # Construire le rapport
        report = {
            'type': report_type,
            'agent': agent.name,
            'timestamp': datetime.now().isoformat(),
            'task_id': agent.state.get('current_task_id'),
            'content': normalized_content
        }
        
        # Envoyer au superviseur
        agent.supervisor.receive_report(agent.name, report)
        
        # Logger dans le RAG si important
        if report_type in ['issue', 'completion']:
            if agent.rag_engine:
                agent.rag_engine.index_to_working_memory(
                    f"Rapport {agent.name} → Superviseur: {report_type}",
                    {
                        'type': 'supervisor_report',
                        'agent_name': agent.name,
                        'report_type': report_type,
                        'milestone': agent.current_milestone_id
                    }
                )
        
        return ToolResult('success', result={'report_sent': True})
        
    except Exception as e:
        return ToolResult('error', error=str(e))