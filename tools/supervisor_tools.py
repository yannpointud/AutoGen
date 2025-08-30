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
        for m in agent._milestone_manager.milestones:
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
        total = len(agent._milestone_manager.milestones)
        
        # Calculer la progression réelle en tenant compte des jalons partiels
        fully_completed = len([m for m in agent._milestone_manager.milestones if m.get('status') == 'completed'])
        partially_completed = len([m for m in agent._milestone_manager.milestones if m.get('status') == 'partially_completed'])
        
        # Les jalons partiels comptent pour 0.7 de la progression
        effective_completion = fully_completed + (partially_completed * 0.7)
        
        report = {
            'project_name': agent.project_name,
            'status': agent.project_state['status'],
            'progress_percentage': (effective_completion / total * 100) if total > 0 else 0,
            'completed_milestones': fully_completed,
            'partially_completed_milestones': partially_completed,
            'total_milestones': total,
            'current_milestone': agent._milestone_manager.get_current_milestone()['name'] 
                               if agent._milestone_manager.get_current_milestone() else 'Terminé',
            'started_at': agent.project_state['started_at'],
            'phase': agent.project_state['current_phase']
        }
        
        if include_details:
            report['milestones'] = []
            for m in agent._milestone_manager.milestones:
                milestone_details = {
                    'id': m['id'],
                    'name': m['name'],
                    'status': m.get('status', 'pending'),
                    'agents': m.get('agents_required', [])
                }
                
                # Ajouter des détails pour les jalons partiellement complétés
                if m.get('status') == 'partially_completed':
                    milestone_details['partial_completion_reason'] = m.get('partial_completion_reason', 'Raison non spécifiée')
                    milestone_details['correction_attempts'] = m.get('correction_attempts', 0)
                    milestone_details['completion_level'] = 'partial'
                
                report['milestones'].append(milestone_details)
        
        # Sauvegarder le rapport (interne - pas un livrable)
        report_path = Path("projects") / agent.project_name / "progress_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        
        return ToolResult('success', result=report)
        
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
        
        # Dans le système immutable, on utilise insert_correction ou add_milestone
        if after_id:
            # Vérifier que le jalon de référence existe
            anchor = agent._milestone_manager.find_milestone(after_id)
            if not anchor:
                return ToolResult('error', error=f"Jalon {after_id} non trouvé")
            
            # Pour l'instant, utiliser insert_correction comme mécanisme d'insertion
            new_milestone = agent._milestone_manager.insert_correction_after_current(
                name=name,
                description=description,
                agents_required=[a for a in agents_required if a in ['analyst', 'developer']],
                deliverables=deliverables,
                estimated_duration='À estimer',
                dependencies=[]
            )
        else:
            # Ajouter à la fin via le manager
            new_milestone = agent._milestone_manager.add_milestone(
                name=name,
                description=description,
                agents_required=[a for a in agents_required if a in ['analyst', 'developer']],
                deliverables=deliverables,
                estimated_duration='À estimer',
                dependencies=[]
            )
        
        # Mettre à jour dans le RAG
        agent._update_plan_in_rag(f"Nouveau jalon ajouté: {name}")
        
        return ToolResult('success', result={
            'milestone_added': new_milestone,
            'total_milestones': len(agent._milestone_manager.milestones)
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
        
        # Trouver le jalon via le manager
        milestone = agent._milestone_manager.find_milestone(milestone_id)
        
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
    """OBSOLÈTE: Dans le système immutable, les jalons ne sont plus supprimés."""
    from agents.base_agent import ToolResult
    
    return ToolResult('error', error="Suppression de jalons non supportée dans le système immutable. Utilisez des corrections à la place.")


def tool_add_correction(agent, parameters: Dict[str, Any]):
    """Ajoute un jalon de correction après le jalon courant."""
    from agents.base_agent import ToolResult
    
    try:
        name = parameters.get('name', '')
        description = parameters.get('description', '')
        
        if not name or not description:
            return ToolResult('error', error="Nom et description requis")
        
        # Ajouter la correction via le manager immutable
        correction = agent._milestone_manager.insert_correction_after_current(
            name=name,
            description=description,
            agents_required=parameters.get('agents_required', ['developer']),
            deliverables=parameters.get('deliverables', ["Rapport de correction"])
        )
        
        # Mettre à jour dans le RAG
        agent._update_plan_in_rag(f"Jalon de correction ajouté: {name}")
        
        return ToolResult('success', result={
            'correction_added': correction,
            'inserted_after_current': True,
            'total_milestones': len(agent._milestone_manager.milestones)
        })
        
    except Exception as e:
        return ToolResult('error', error=str(e))


