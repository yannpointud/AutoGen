"""
Outils spécifiques à l'agent Supervisor.
"""

from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import json


def tool_assign_agents_to_milestone(agent, parameters: Dict[str, Any]):
    """Assigne des agents à un jalon."""
    from agents.base_agent import ToolResult
    
    try:
        milestone_id = parameters.get('milestone_id', '')
        agents_list = parameters.get('agents', [])
        
        # Valider les agents
        valid_agents = ['analyst', 'developer']
        agents_to_assign = [a for a in agents_list if a in valid_agents]
        
        # Trouver le jalon
        milestone = None
        for m in agent.milestones:
            if str(m.get('id')) == str(milestone_id) or m.get('milestone_id') == milestone_id:
                milestone = m
                break
        
        if not milestone:
            return ToolResult('error', error=f"Jalon {milestone_id} non trouvé")
        
        # Assigner les agents
        milestone['agents_required'] = agents_to_assign
        
        return ToolResult('success', result={
            'milestone': milestone_id,
            'agents_assigned': agents_to_assign
        })
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_get_progress_report(agent, parameters: Dict[str, Any]):
    """Génère un rapport de progression."""
    from agents.base_agent import ToolResult
    
    try:
        include_details = parameters.get('include_details', 'false').lower() == 'true'
        
        completed = agent.project_state['milestones_completed']
        total = len(agent.milestones)
        
        report = {
            'project_name': agent.project_name,
            'status': agent.project_state['status'],
            'progress_percentage': (completed / total * 100) if total > 0 else 0,
            'completed_milestones': completed,
            'total_milestones': total,
            'current_milestone': agent.milestones[agent.current_milestone_index]['name'] 
                               if agent.current_milestone_index < total else 'Terminé',
            'started_at': agent.project_state['started_at'],
            'phase': agent.project_state['current_phase']
        }
        
        if include_details:
            report['milestones'] = []
            for m in agent.milestones:
                report['milestones'].append({
                    'id': m['id'],
                    'name': m['name'],
                    'status': m.get('status', 'pending'),
                    'agents': m.get('agents_required', [])
                })
        
        # Sauvegarder le rapport
        report_path = Path("projects") / agent.project_name / "progress_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        
        return ToolResult('success', result=report, artifact=str(report_path))
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_add_milestone(agent, parameters: Dict[str, Any]):
    """Ajoute un nouveau jalon au plan."""
    from agents.base_agent import ToolResult
    
    try:
        after_id = parameters.get('after_milestone_id')
        name = parameters.get('name', '')
        description = parameters.get('description', '')
        agents_required = parameters.get('agents_required', [])
        deliverables = parameters.get('deliverables', [])
        
        if not name or not description:
            return ToolResult('error', error="Nom et description requis")
        
        # Générer un nouvel ID
        new_id = max([m['id'] for m in agent.milestones], default=0) + 1
        
        # Créer le nouveau jalon
        new_milestone = {
            'id': new_id,
            'milestone_id': f'milestone_{new_id}',
            'name': name,
            'description': description,
            'agents_required': [a for a in agents_required if a in ['analyst', 'developer']],
            'deliverables': deliverables,
            'estimated_duration': 'À estimer',
            'dependencies': [],
            'status': 'pending'
        }
        
        # Trouver la position d'insertion
        if after_id:
            insert_pos = None
            for i, m in enumerate(agent.milestones):
                if str(m['id']) == str(after_id) or m['milestone_id'] == after_id:
                    insert_pos = i + 1
                    break
            if insert_pos is None:
                return ToolResult('error', error=f"Jalon {after_id} non trouvé")
            agent.milestones.insert(insert_pos, new_milestone)
        else:
            # Ajouter à la fin
            agent.milestones.append(new_milestone)
        
        # Mettre à jour dans le RAG
        agent._update_plan_in_rag(f"Nouveau jalon ajouté: {name}")
        
        return ToolResult('success', result={
            'milestone_added': new_milestone,
            'total_milestones': len(agent.milestones)
        })
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_modify_milestone(agent, parameters: Dict[str, Any]):
    """Modifie un jalon existant."""
    from agents.base_agent import ToolResult
    
    try:
        milestone_id = parameters.get('milestone_id', '')
        changes = parameters.get('changes', {})
        
        if not milestone_id:
            return ToolResult('error', error="ID du jalon requis")
        
        # Trouver le jalon
        milestone = None
        for m in agent.milestones:
            if str(m['id']) == str(milestone_id) or m['milestone_id'] == milestone_id:
                milestone = m
                break
        
        if not milestone:
            return ToolResult('error', error=f"Jalon {milestone_id} non trouvé")
        
        # Vérifier si le jalon peut être modifié
        if milestone['status'] == 'completed':
            return ToolResult('error', error="Impossible de modifier un jalon terminé")
        
        # Appliquer les modifications
        allowed_fields = ['name', 'description', 'agents_required', 'deliverables', 'estimated_duration']
        modifications = []
        
        for field, new_value in changes.items():
            if field in allowed_fields:
                old_value = milestone.get(field)
                milestone[field] = new_value
                modifications.append(f"{field}: {old_value} → {new_value}")
        
        # Mettre à jour dans le RAG
        agent._update_plan_in_rag(f"Jalon {milestone['name']} modifié: {', '.join(modifications)}")
        
        return ToolResult('success', result={
            'milestone_modified': milestone,
            'changes_applied': modifications
        })
        
    except Exception as e:
        return ToolResult('error', error=str(e))


def tool_remove_milestone(agent, parameters: Dict[str, Any]):
    """Supprime un jalon (seulement si pas commencé)."""
    from agents.base_agent import ToolResult
    
    try:
        milestone_id = parameters.get('milestone_id', '')
        
        if not milestone_id:
            return ToolResult('error', error="ID du jalon requis")
        
        # Trouver le jalon
        milestone_index = None
        milestone = None
        for i, m in enumerate(agent.milestones):
            if str(m['id']) == str(milestone_id) or m['milestone_id'] == milestone_id:
                milestone_index = i
                milestone = m
                break
        
        if milestone is None:
            return ToolResult('error', error=f"Jalon {milestone_id} non trouvé")
        
        # Vérifier si le jalon peut être supprimé
        if milestone['status'] in ['in_progress', 'completed']:
            return ToolResult('error', error="Impossible de supprimer un jalon commencé ou terminé")
        
        # Supprimer le jalon
        removed_milestone = agent.milestones.pop(milestone_index)
        
        # Ajuster current_milestone_index si nécessaire
        if milestone_index <= agent.current_milestone_index:
            agent.current_milestone_index = max(0, agent.current_milestone_index - 1)
        
        # Mettre à jour dans le RAG
        agent._update_plan_in_rag(f"Jalon supprimé: {removed_milestone['name']}")
        
        return ToolResult('success', result={
            'milestone_removed': removed_milestone,
            'total_milestones': len(agent.milestones)
        })
        
    except Exception as e:
        return ToolResult('error', error=str(e))


