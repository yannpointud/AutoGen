"""
Agent Superviseur utilisant une architecture orientée outils.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
from collections import defaultdict
import time

from agents.base_agent import BaseAgent, Tool, ToolResult
from agents.analyst import Analyst
from agents.developer import Developer

from core.llm_connector import LLMFactory
from core.rag_engine import RAGEngine
from config import default_config


class Supervisor(BaseAgent):
    """
    Agent Superviseur qui orchestre le projet en utilisant des outils.
    """
    
    def __init__(self, project_name: str, project_prompt: str, rag_engine: Optional[Any] = None):
        """
        Initialise le superviseur avec ses outils spécifiques.
        """
        supervisor_config = default_config['agents']['supervisor']
        llm_config = {
            'model': supervisor_config.get('model', default_config['llm']['default_model']),
            'temperature': 0.7
        }
        
        super().__init__(
            name="Supervisor",
            role=supervisor_config['role'],
            personality=supervisor_config['personality'],
            llm_config=llm_config,
            project_name=project_name,
            supervisor=None,
            rag_engine=rag_engine
        )
        
        self.project_prompt = project_prompt
        self.rag_singleton = RAGEngine(project_name=self.project_name)
        
        # Gestion des jalons et agents
        self.milestones = []
        self.agents = {}
        self.current_milestone_index = 0
        
        # État du projet
        self.project_state = {
            'status': 'initialized',
            'started_at': datetime.now().isoformat(),
            'milestones_completed': 0,
            'current_phase': 'planning'
        }
        
        # Configuration
        self.max_milestones = default_config.get('supervisor', {}).get('max_milestones', 10)
        self.min_milestones = default_config.get('supervisor', {}).get('min_milestones', 2)
        
        # Enregistrer les outils spécifiques
        self._register_supervisor_tools()
        
        self.logger.info(f"Superviseur initialisé pour le projet: {project_name}")
    
    def _register_supervisor_tools(self) -> None:
        """Enregistre les outils spécifiques au superviseur."""
        
        # assign_agents_to_milestone
        self.register_tool(
            Tool(
                "assign_agents_to_milestone",
                "Assigne des agents à un jalon spécifique",
                {
                    "milestone_id": "ID du jalon",
                    "agents": "Liste des agents à assigner (analyst/developer)"
                }
            ),
            self._tool_assign_agents_to_milestone
        )
        
        # get_progress_report
        self.register_tool(
            Tool(
                "get_progress_report",
                "Génère un rapport de progression du projet",
                {
                    "include_details": "Inclure les détails (true/false)"
                }
            ),
            self._tool_get_progress_report
        )
        
        # add_milestone
        self.register_tool(
            Tool(
                "add_milestone",
                "Ajoute un nouveau jalon au plan",
                {
                    "after_milestone_id": "ID du jalon après lequel insérer (optionnel)",
                    "name": "Nom du nouveau jalon",
                    "description": "Description détaillée du jalon",
                    "agents_required": "Liste des agents nécessaires",
                    "deliverables": "Liste des livrables attendus"
                }
            ),
            self._tool_add_milestone
        )
        
        # modify_milestone
        self.register_tool(
            Tool(
                "modify_milestone",
                "Modifie un jalon existant",
                {
                    "milestone_id": "ID du jalon à modifier",
                    "changes": "Dictionnaire des modifications à apporter"
                }
            ),
            self._tool_modify_milestone
        )
        
        # remove_milestone
        self.register_tool(
            Tool(
                "remove_milestone",
                "Supprime un jalon (seulement si pas commencé)",
                {
                    "milestone_id": "ID du jalon à supprimer"
                }
            ),
            self._tool_remove_milestone
        )
    
    def _tool_assign_agents_to_milestone(self, parameters: Dict[str, Any]) -> ToolResult:
        """Assigne des agents à un jalon."""
        try:
            milestone_id = parameters.get('milestone_id', '')
            agents_list = parameters.get('agents', [])
            
            # Valider les agents
            valid_agents = ['analyst', 'developer']
            agents_to_assign = [a for a in agents_list if a in valid_agents]
            
            # Trouver le jalon
            milestone = None
            for m in self.milestones:
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
    
    def _tool_get_progress_report(self, parameters: Dict[str, Any]) -> ToolResult:
        """Génère un rapport de progression."""
        try:
            include_details = parameters.get('include_details', 'false').lower() == 'true'
            
            completed = self.project_state['milestones_completed']
            total = len(self.milestones)
            
            report = {
                'project_name': self.project_name,
                'status': self.project_state['status'],
                'progress_percentage': (completed / total * 100) if total > 0 else 0,
                'completed_milestones': completed,
                'total_milestones': total,
                'current_milestone': self.milestones[self.current_milestone_index]['name'] 
                                   if self.current_milestone_index < total else 'Terminé',
                'started_at': self.project_state['started_at'],
                'phase': self.project_state['current_phase']
            }
            
            if include_details:
                report['milestones'] = []
                for m in self.milestones:
                    report['milestones'].append({
                        'id': m['id'],
                        'name': m['name'],
                        'status': m.get('status', 'pending'),
                        'agents': m.get('agents_required', [])
                    })
            
            # Sauvegarder le rapport
            report_path = Path("projects") / self.project_name / "progress_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            
            return ToolResult('success', result=report, artifact=str(report_path))
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def think(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Le superviseur analyse le projet et planifie les jalons.
        """
        self.update_state(
            status='thinking',
            current_task=task,
            current_task_id='project_planning'
        )
        
        project_prompt = task.get('prompt', self.project_prompt)
        
        # Prompt pour analyser et découper le projet
        analysis_prompt = f"""Tu es {self.name}, {self.role}.

Projet à analyser: {project_prompt}

Tu dois:
1. Analyser la complexité du projet
2. Décider du nombre de jalons approprié (entre {self.min_milestones} et {self.max_milestones})
3. Définir chaque jalon avec ses livrables
4. Déterminer quels agents sont nécessaires pour chaque jalon

Agents disponibles:
- analyst: pour l'analyse, la conception et la documentation
- developer: pour l'implémentation, les tests et la configuration

Réponds avec une analyse structurée et un plan de jalons clair.
"""
        
        try:
            # Générer l'analyse
            analysis = self.generate_with_context(
                prompt=analysis_prompt,
                temperature=0.6
            )
            
            # Créer les jalons basés sur l'analyse
            milestones = self._create_milestones_from_analysis(analysis, project_prompt)
            
            plan = {
                'task_id': 'supervisor_planning',
                'analysis': analysis,
                'milestones': milestones,
                'strategy': f"Projet découpé en {len(milestones)} jalons",
                'timestamp': datetime.now().isoformat()
            }
            
            # Sauvegarder les jalons
            self.milestones = milestones
            
            self.log_interaction('think', plan)
            return plan
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la planification: {str(e)}")
            # Plan minimal de fallback
            return self._create_fallback_plan(project_prompt)
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Le superviseur prépare l'exécution du projet.
        """
        self.update_state(status='acting', current_phase='preparation')
        
        result = {
            'task_id': plan.get('task_id'),
            'status': 'in_progress',
            'milestones_created': len(self.milestones),
            'agents_created': 0
        }
        
        try:
            # Créer les agents
            self.agents = self.create_agents()
            result['agents_created'] = len(self.agents)
            
            # Partager le plan via le RAG
            if self.rag_engine:
                plan_summary = f"Plan projet: {len(self.milestones)} jalons"
                self.rag_engine.index_to_working_memory(
                    plan_summary,
                    {
                        'type': 'project_plan',
                        'agent_name': self.name,
                        'milestone': 'planning'
                    }
                )
            
            result['status'] = 'ready'
            self.project_state['status'] = 'ready_to_execute'
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation: {str(e)}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def create_agents(self) -> Dict[str, BaseAgent]:
        """Crée les agents nécessaires."""
        agents = {}
        
        try:
            # Créer l'analyste
            analyst = Analyst(
                project_name=self.project_name,
                supervisor=self,
                rag_engine=self.rag_singleton
            )
            agents['analyst'] = analyst
            
            # Créer le développeur
            developer = Developer(
                project_name=self.project_name,
                supervisor=self,
                rag_engine=self.rag_singleton
            )
            agents['developer'] = developer
            
            self.logger.info(f"2 agents créés: analyst, developer")
            
        except Exception as e:
            self.logger.error(f"Erreur création agents: {str(e)}")
        
        return agents
    



    def orchestrate(self) -> Dict[str, Any]:
        """Orchestre l'exécution complète du projet."""
        self.logger.info("Début de l'orchestration du projet")
        self.update_state(status='orchestrating', current_phase='execution')
        
        orchestration_result = {
            'project_name': self.project_name,
            'started_at': datetime.now().isoformat(),
            'milestones_results': []
        }
        
        try:
            while self.current_milestone_index < len(self.milestones):
                milestone = self.milestones[self.current_milestone_index]
                
                # Exécution du jalon
                self.logger.info(f"Exécution du jalon {milestone['id']}: {milestone['name']}")
                for agent in self.agents.values():
                    agent.update_state(current_milestone_id=milestone['milestone_id'])
                
                milestone_result = self._execute_milestone(milestone)
                orchestration_result['milestones_results'].append(milestone_result)
                
                # Bilan et journalisation du jalon terminé
                self.logger.info(f"Bilan du jalon {milestone['id']}...")
                summary = self._generate_milestone_summary(milestone_result)
                self._create_journal_entry(
                    'milestone_completed',
                    summary,
                    {
                        'milestone_name': milestone['name'],
                        'agents': milestone['agents_required'],
                        'artifacts': [artifact for task in milestone_result.get('tasks_completed', []) for artifact in task.get('artifacts', [])]
                    }
                )
                
                # Mise à jour de l'état et passage au suivant
                self.current_milestone_index += 1
                self.project_state['milestones_completed'] += 1
                milestone['status'] = 'completed'

                # Pause stratégique entre jalons
                if self.current_milestone_index < len(self.milestones):
                    self.logger.info("Traitement fin de jalon")
                    time.sleep(5)        

            orchestration_result['status'] = 'completed'
            orchestration_result['ended_at'] = datetime.now().isoformat()
            
            # Générer le résumé final
            self._generate_project_summary(orchestration_result)
            
        except Exception as e:
            self.logger.error(f"Erreur pendant l'orchestration: {str(e)}")
            orchestration_result['status'] = 'error'
            orchestration_result['error'] = str(e)
        
        return orchestration_result
    





    def _execute_milestone(self, milestone: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute un jalon avec les agents assignés."""
        result = {
            'milestone_id': milestone['id'],
            'milestone_name': milestone['name'],
            'start_time': datetime.now().isoformat(),
            'agents_involved': milestone['agents_required'],
            'tasks_completed': []
        }
        
        # Pour chaque agent requis
        for agent_type in milestone['agents_required']:
            if agent_type not in self.agents:
                continue
            
            agent = self.agents[agent_type]
            
            # Créer la tâche pour l'agent
            task = {
                'milestone': milestone['name'],
                'milestone_id': milestone['milestone_id'],
                'description': milestone['description'],
                'deliverables': milestone.get('deliverables', []),
                'project_prompt': self.project_prompt
            }
            
            try:
                # L'agent réfléchit
                agent_plan = agent.think(task)
                
                # L'agent exécute
                agent_result = agent.act(agent_plan)
                
                result['tasks_completed'].append({
                    'agent': agent_type,
                    'plan': agent_plan,
                    'result': agent_result,
                    'artifacts': agent_result.get('artifacts', [])
                })
                
            except Exception as e:
                self.logger.error(f"Erreur agent {agent_type}: {str(e)}")
                result['tasks_completed'].append({
                    'agent': agent_type,
                    'error': str(e),
                    'status': 'error'
                })
        
        result['end_time'] = datetime.now().isoformat()
        result['status'] = 'completed'
        
        # La génération du résumé et l'entrée de journal sont maintenant gérées par l'orchestrateur.
        
        return result
    
    def _create_milestones_from_analysis(self, analysis: str, project_prompt: str) -> List[Dict[str, Any]]:
        """Crée les jalons à partir de l'analyse."""
        # Prompt pour créer les jalons structurés
        milestone_prompt = f"""Basé sur cette analyse:
{analysis[:1000]}

Crée entre {self.min_milestones} et {self.max_milestones} jalons pour ce projet.

Réponds uniquement avec un JSON valide:
{{
    "milestones": [
        {{
            "id": 1,
            "name": "Nom du jalon",
            "description": "Description détaillée",
            "agents_required": ["analyst", "developer"],
            "deliverables": ["livrable1", "livrable2"],
            "estimated_duration": "durée estimée",
            "dependencies": []
        }}
    ]
}}
"""
        
        try:
            response = self.generate_json_with_context(
                prompt=milestone_prompt,
                temperature=0.5
            )
            
            milestones = response.get('milestones', [])
            
            # Valider et enrichir
            for i, m in enumerate(milestones):
                m['id'] = m.get('id', i + 1)
                m['milestone_id'] = f"milestone_{m['id']}"
                m['status'] = 'pending'
                m['agents_required'] = [a for a in m.get('agents_required', []) 
                                       if a in ['analyst', 'developer']]
                if not m['agents_required']:
                    m['agents_required'] = ['analyst']
            
            return milestones[:self.max_milestones]

        except Exception as e:
            # Logguer l'erreur spécifique pour le débogage
            self.logger.error(f"Échec de la génération des jalons depuis l'analyse, utilisation du plan de secours. Erreur: {e}")
            # Fallback
            return self._get_default_milestones()
    
    def _get_default_milestones(self) -> List[Dict[str, Any]]:
        """Retourne des jalons par défaut."""
        return [
            {
                'id': 1,
                'milestone_id': 'milestone_1',
                'name': 'Analyse et Conception',
                'description': 'Analyser les besoins et concevoir l\'architecture',
                'agents_required': ['analyst'],
                'deliverables': ['requirements.md', 'architecture.md'],
                'estimated_duration': '2 heures',
                'dependencies': [],
                'status': 'pending'
            },
            {
                'id': 2,
                'milestone_id': 'milestone_2',
                'name': 'Implémentation',
                'description': 'Implémenter le code et les tests',
                'agents_required': ['developer'],
                'deliverables': ['src/', 'tests/'],
                'estimated_duration': '4 heures',
                'dependencies': [1],
                'status': 'pending'
            },
            {
                'id': 3,
                'milestone_id': 'milestone_3',
                'name': 'Documentation et Finalisation',
                'description': 'Finaliser la documentation et les configurations',
                'agents_required': ['analyst', 'developer'],
                'deliverables': ['README.md', 'config/'],
                'estimated_duration': '1 heure',
                'dependencies': [2],
                'status': 'pending'
            }
        ]
    
    def _create_fallback_plan(self, project_prompt: str) -> Dict[str, Any]:
        """Crée un plan de secours."""
        milestones = self._get_default_milestones()
        
        return {
            'task_id': 'supervisor_planning_fallback',
            'analysis': 'Plan par défaut appliqué',
            'milestones': milestones,
            'strategy': 'Approche standard en 3 phases',
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_project_summary(self, orchestration_result: Dict[str, Any]) -> None:
        """Génère un résumé du projet."""
        # Compter les artifacts
        total_artifacts = sum(
            len(task.get('artifacts', []))
            for milestone in orchestration_result['milestones_results']
            for task in milestone.get('tasks_completed', [])
        )
        
        summary = f"""# Résumé du Projet {self.project_name}

## Vue d'ensemble
- **Prompt initial**: {self.project_prompt[:200]}...
- **Jalons complétés**: {self.project_state['milestones_completed']}/{len(self.milestones)}
- **Artifacts créés**: {total_artifacts}

## Jalons exécutés
"""
        
        for milestone in self.milestones:
            status = "✅" if milestone['status'] == 'completed' else "⏳"
            summary += f"\n{status} **{milestone['name']}**\n"
            summary += f"   - Agents: {', '.join(milestone['agents_required'])}\n"
            summary += f"   - Livrables: {', '.join(milestone.get('deliverables', []))}\n"
        
        # Sauvegarder
        summary_path = Path("projects") / self.project_name / "PROJECT_SUMMARY.md"
        summary_path.write_text(summary, encoding='utf-8')
        
        self.logger.info("Résumé du projet généré")
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Retourne un agent par son nom."""
        agent_name_lower = agent_name.lower()
        if agent_name_lower == 'supervisor':
            return self
        return self.agents.get(agent_name_lower)
    
    def get_all_agents_with_roles(self) -> Dict[str, str]:
        """Retourne tous les agents avec leurs rôles."""
        agents_roles = {'supervisor': self.role}
        
        for name, agent in self.agents.items():
            agents_roles[name] = agent.role
        
        return agents_roles
    
    def handle_escalation(self, from_agent: str, issue: str) -> str:
        """Gère une escalade d'un agent."""
        self.logger.warning(f"Escalade de {from_agent}: {issue[:100]}...")
        
        # Analyser et fournir une solution
        escalation_prompt = f"""En tant que superviseur, résous cette escalade:

Agent: {from_agent}
Problème: {issue}

Fournis une solution concrète et actionnable.
"""
        
        try:
            guidance = self.generate_with_context(
                prompt=escalation_prompt,
                temperature=0.6,
            )
            return guidance

        except Exception as e: 
            self.logger.error(f"Échec critique de la gestion d'escalade: {e}", exc_info=True)
            return "Erreur interne lors de la tentative de résolution. Appliquer la procédure de secours par défaut : privilégier la simplicité et la complétion de la tâche."       
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Génère un rapport de progression."""
        # Utiliser l'outil pour générer le rapport
        result = self._tool_get_progress_report({'include_details': 'true'})
        return result.result if result.status == 'success' else {}
    
    def communicate(self, message: str, recipient: Optional[BaseAgent] = None) -> str:
        """Communication du superviseur."""
        self.update_state(status='communicating')
        
        if recipient:
            self.logger.info(f"Communication vers {recipient.name}: {message[:100]}...")
            recipient.receive_message(self.name, message)
            return f"Message transmis à {recipient.name}"
        else:
            # Broadcast
            for agent in self.agents.values():
                agent.receive_message(self.name, message)
            return f"Message diffusé à {len(self.agents)} agents"
    
    def _tool_add_milestone(self, parameters: Dict[str, Any]) -> ToolResult:
        """Ajoute un nouveau jalon au plan."""
        try:
            after_id = parameters.get('after_milestone_id')
            name = parameters.get('name', '')
            description = parameters.get('description', '')
            agents_required = parameters.get('agents_required', [])
            deliverables = parameters.get('deliverables', [])
            
            if not name or not description:
                return ToolResult('error', error="Nom et description requis")
            
            # Générer un nouvel ID
            new_id = max([m['id'] for m in self.milestones], default=0) + 1
            
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
                for i, m in enumerate(self.milestones):
                    if str(m['id']) == str(after_id) or m['milestone_id'] == after_id:
                        insert_pos = i + 1
                        break
                if insert_pos is None:
                    return ToolResult('error', error=f"Jalon {after_id} non trouvé")
                self.milestones.insert(insert_pos, new_milestone)
            else:
                # Ajouter à la fin
                self.milestones.append(new_milestone)
            
            # Mettre à jour dans le RAG
            self._update_plan_in_rag(f"Nouveau jalon ajouté: {name}")
            
            return ToolResult('success', result={
                'milestone_added': new_milestone,
                'total_milestones': len(self.milestones)
            })
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _tool_modify_milestone(self, parameters: Dict[str, Any]) -> ToolResult:
        """Modifie un jalon existant."""
        try:
            milestone_id = parameters.get('milestone_id', '')
            changes = parameters.get('changes', {})
            
            if not milestone_id:
                return ToolResult('error', error="ID du jalon requis")
            
            # Trouver le jalon
            milestone = None
            for m in self.milestones:
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
            self._update_plan_in_rag(f"Jalon {milestone['name']} modifié: {', '.join(modifications)}")
            
            return ToolResult('success', result={
                'milestone_modified': milestone,
                'changes_applied': modifications
            })
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _tool_remove_milestone(self, parameters: Dict[str, Any]) -> ToolResult:
        """Supprime un jalon (seulement si pas commencé)."""
        try:
            milestone_id = parameters.get('milestone_id', '')
            
            if not milestone_id:
                return ToolResult('error', error="ID du jalon requis")
            
            # Trouver le jalon
            milestone_index = None
            milestone = None
            for i, m in enumerate(self.milestones):
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
            removed_milestone = self.milestones.pop(milestone_index)
            
            # Ajuster current_milestone_index si nécessaire
            if milestone_index <= self.current_milestone_index:
                self.current_milestone_index = max(0, self.current_milestone_index - 1)
            
            # Mettre à jour dans le RAG
            self._update_plan_in_rag(f"Jalon supprimé: {removed_milestone['name']}")
            
            return ToolResult('success', result={
                'milestone_removed': removed_milestone,
                'total_milestones': len(self.milestones)
            })
            
        except Exception as e:
            return ToolResult('error', error=str(e))
    
    def _update_plan_in_rag(self, change_description: str) -> None:
        """Met à jour le plan dans le RAG après modification."""
        if self.rag_engine:
            plan_summary = f"Plan modifié: {change_description}. Jalons actuels: {len(self.milestones)}"
            self.rag_engine.index_to_working_memory(
                plan_summary,
                {
                    'type': 'plan_modification',
                    'agent_name': self.name,
                    'change': change_description,
                    'milestone_count': len(self.milestones)
                }
            )
    
    def _evaluate_plan_after_interaction(self, interaction_type: str, content: str) -> None:
        """Évalue si le plan doit être modifié après une interaction avec un agent."""
        try:
            # Construire le contexte pour l'évaluation
            current_milestone = None
            if self.current_milestone_index < len(self.milestones):
                current_milestone = self.milestones[self.current_milestone_index]
            
            remaining_milestones = self.milestones[self.current_milestone_index:]
            
            evaluation_prompt = f"""Tu es le superviseur du projet {self.project_name}.

Une interaction vient d'avoir lieu:
Type: {interaction_type}
Contenu: {content}

État actuel du projet:
- Jalon actuel: {current_milestone['name'] if current_milestone else 'Tous terminés'}
- Jalons restants: {len(remaining_milestones)}
- Jalons terminés: {self.project_state['milestones_completed']}

Plan actuel des jalons restants:
{self._format_milestones_for_evaluation(remaining_milestones)}

À la lumière de cette nouvelle information, analyse si:
1. Le plan actuel est-il toujours adapté ?
2. Des jalons doivent-ils être ajoutés, modifiés ou supprimés ?
3. L'ordre des jalons est-il optimal ?

Réponds avec un JSON:
{{
    "plan_needs_change": true/false,
    "reasoning": "explication de ton analyse",
    "suggested_changes": [
        {{
            "action": "add/modify/remove",
            "milestone_id": "id si modification/suppression",
            "details": {{}}
        }}
    ]
}}

Sois conservateur : ne propose des changements que si vraiment nécessaire."""
            
            # Générer l'évaluation
            evaluation = self.generate_json_with_context(
                prompt=evaluation_prompt,
                temperature=0.6
            )
            
            # Traiter les résultats
            if evaluation.get('plan_needs_change', False):
                self.logger.info(f"Évaluation: le plan nécessite des modifications - {evaluation.get('reasoning', '')}")
                self._apply_plan_changes(evaluation.get('suggested_changes', []))
                self._update_journal_de_bord(interaction_type, content, evaluation)
            else:
                self.logger.debug(f"Évaluation: le plan reste valide - {evaluation.get('reasoning', '')}")
                # Créer une entrée de journal même si pas de modification
                self._create_journal_entry(
                    'plan_evaluation',
                    content,
                    {
                        'trigger': interaction_type,
                        'status': 'stable',
                        'reasoning': evaluation.get('reasoning', '')
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation du plan: {str(e)}")
    
    def _format_milestones_for_evaluation(self, milestones: List[Dict[str, Any]]) -> str:
        """Formate les jalons pour l'évaluation."""
        if not milestones:
            return "Aucun jalon restant"
        
        formatted = []
        for m in milestones:
            formatted.append(f"- {m['id']}: {m['name']} ({m['status']}) - {m['description']}")
        return '\n'.join(formatted)
    
    def _apply_plan_changes(self, suggested_changes: List[Dict[str, Any]]) -> None:
        """Applique les modifications suggérées au plan."""
        for change in suggested_changes:
            action = change.get('action')
            try:
                if action == 'add':
                    details = change.get('details', {})
                    result = self._tool_add_milestone(details)
                    if result.status == 'success':
                        self.logger.info(f"Jalon ajouté: {details.get('name')}")
                
                elif action == 'modify':
                    milestone_id = change.get('milestone_id')
                    details = change.get('details', {})
                    result = self._tool_modify_milestone({
                        'milestone_id': milestone_id,
                        'changes': details
                    })
                    if result.status == 'success':
                        self.logger.info(f"Jalon {milestone_id} modifié")
                
                elif action == 'remove':
                    milestone_id = change.get('milestone_id')
                    result = self._tool_remove_milestone({'milestone_id': milestone_id})
                    if result.status == 'success':
                        self.logger.info(f"Jalon {milestone_id} supprimé")
                        
            except Exception as e:
                self.logger.error(f"Erreur lors de l'application du changement {action}: {str(e)}")
    
    def _update_journal_de_bord(self, interaction_type: str, content: str, evaluation: Dict[str, Any]) -> None:
        """Met à jour le journal de bord avec les modifications de plan."""
        try:
            journal_path = Path("projects") / self.project_name / "docs" / "JOURNAL_DE_BORD.md"
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Préparer l'entrée
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"""
## {timestamp} - Modification de Plan

**Déclencheur**: {interaction_type}
**Contenu**: {content[:200]}{'...' if len(content) > 200 else ''}

**Analyse**: {evaluation.get('reasoning', 'Non spécifiée')}

**Modifications appliquées**:
"""
            for change in evaluation.get('suggested_changes', []):
                entry += f"- {change.get('action', 'unknown').upper()}: {change.get('details', {})}\n"
            
            entry += f"\n**État du projet**: {self.project_state['milestones_completed']}/{len(self.milestones)} jalons terminés\n\n---\n"
            
            # Ajouter au journal
            if journal_path.exists():
                with open(journal_path, 'a', encoding='utf-8') as f:
                    f.write(entry)
            else:
                with open(journal_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Journal de Bord - Projet {self.project_name}\n\n")
                    f.write(entry)
            
            # Ré-indexer dans le RAG avec priorité
            if self.rag_engine:
                self.rag_engine.index_to_working_memory(
                    f"Modification de plan: {evaluation.get('reasoning', '')}",
                    {
                        'type': 'project_journal',
                        'agent_name': self.name,
                        'preserve': True,
                        'priority': 'high'
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour du journal de bord: {str(e)}")
    
    def receive_report(self, agent_name: str, report: Dict[str, Any]) -> None:
        """Reçoit un rapport d'un autre agent et évalue si le plan doit être modifié."""
        super().receive_report(agent_name, report)

        # Évaluer le plan après réception du rapport
        report_content = f"Agent {agent_name} rapport {report.get('type', 'status')}: {str(report.get('content', {}))}"
        self._evaluate_plan_after_interaction('report', report_content)
    

    
    def _create_journal_entry(self, entry_type: str, content: str, details: Dict[str, Any] = None) -> None:
        """Crée une entrée dans le journal de bord."""
        try:
            journal_path = Path("projects") / self.project_name / "docs" / "JOURNAL_DE_BORD.md"
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Préparer l'entrée
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if entry_type == "milestone_completed":
                milestone_name = details.get('milestone_name', 'Unknown')
                artifacts = details.get('artifacts', [])
                entry = f"""
## {timestamp} - Jalon Terminé: {milestone_name}

**Résultat**: Jalon complété avec succès
**Artifacts créés**: {len(artifacts)} fichiers
**Agents impliqués**: {', '.join(details.get('agents', []))}

**Résumé des réalisations**:
{content}

**État du projet**: {self.project_state['milestones_completed']}/{len(self.milestones)} jalons terminés

---
"""
            
            elif entry_type == "plan_evaluation":
                entry = f"""
## {timestamp} - Évaluation de Plan

**Déclencheur**: {details.get('trigger', 'Unknown')}
**Contenu**: {content[:200]}{'...' if len(content) > 200 else ''}

**Analyse**: Plan évalué comme {details.get('status', 'stable')}
**État du projet**: {self.project_state['milestones_completed']}/{len(self.milestones)} jalons terminés

---
"""
            
            else:
                entry = f"""
## {timestamp} - {entry_type.title()}

{content}

**État du projet**: {self.project_state['milestones_completed']}/{len(self.milestones)} jalons terminés

---
"""
            
            # Ajouter au journal
            if journal_path.exists():
                with open(journal_path, 'a', encoding='utf-8') as f:
                    f.write(entry)
            else:
                header = f"""# Journal de Bord - Projet {self.project_name}

**Objectif**: {self.project_prompt[:200]}{'...' if len(self.project_prompt) > 200 else ''}
**Début du projet**: {self.project_state['started_at']}

---
"""
                with open(journal_path, 'w', encoding='utf-8') as f:
                    f.write(header)
                    f.write(entry)
            
            # Ré-indexer le journal complet dans le RAG avec priorité
            if self.rag_engine:
                journal_content = journal_path.read_text(encoding='utf-8')
                self.rag_engine.index_to_working_memory(
                    f"Journal de bord mis à jour: {entry_type}",
                    {
                        'type': 'project_journal',
                        'agent_name': self.name,
                        'preserve': True,
                        'priority': 'high',
                        'content_preview': content[:100]
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la création d'entrée de journal: {str(e)}")
    
    def _generate_milestone_summary(self, milestone_result: Dict[str, Any]) -> str:
        """Génère un résumé intelligent d'un jalon terminé."""
        try:
            milestone_name = milestone_result.get('milestone_name', 'Unknown')
            agents_involved = milestone_result.get('agents_involved', [])
            tasks_completed = milestone_result.get('tasks_completed', [])
            
            # Collecter les informations importantes
            decisions = []
            problems = []
            artifacts = []
            
            for task in tasks_completed:
                if task.get('artifacts'):
                    artifacts.extend(task['artifacts'])
                if task.get('result', {}).get('decisions'):
                    decisions.extend(task['result']['decisions'])
                if task.get('result', {}).get('issues'):
                    problems.extend(task['result']['issues'])
            
            # Construire le prompt pour le résumé
            summary_prompt = f"""Génère un résumé concis du jalon terminé:

Jalon: {milestone_name}
Agents: {', '.join(agents_involved)}
Artifacts créés: {len(artifacts)}
Tâches: {len(tasks_completed)}

Informations détaillées:
{str(milestone_result)[:1000]}

Fournis un résumé structuré couvrant:
1. Les principales réalisations
2. Les décisions importantes prises
3. Les problèmes rencontrés et résolus
4. Les points d'attention pour les jalons futurs

Maximum 200 mots, style professionnel."""
            
            summary = self.generate_with_context(
                prompt=summary_prompt,
                temperature=0.5
            )
            
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du résumé de jalon: {str(e)}")
            return f"Jalon {milestone_result.get('milestone_name', 'Unknown')} terminé avec {len(milestone_result.get('tasks_completed', []))} tâches."