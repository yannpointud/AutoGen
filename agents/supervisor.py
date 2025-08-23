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
from tools.supervisor_tools import (
    tool_assign_agents_to_milestone,
    tool_get_progress_report,
    tool_add_milestone,
    tool_modify_milestone,
    tool_remove_milestone
)


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
        self.max_corrections = supervisor_config.get('max_corrections', 3)
       
        # RAG singleton pour les agents
        self.rag_singleton = rag_engine
        
        # Gestion des jalons et agents
        self.milestones = []
        self.agents = {}
        self.current_milestone_index = 0
        
        # Buffer des rapports par jalon
        self.current_milestone_reports = []
        self.current_milestone_id = None
        
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
            lambda params: tool_assign_agents_to_milestone(self, params)
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
            lambda params: tool_get_progress_report(self, params)
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
            lambda params: tool_add_milestone(self, params)
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
            lambda params: tool_modify_milestone(self, params)
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
            lambda params: tool_remove_milestone(self, params)
        )
    
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
        
        # === LOGIQUE CONDITIONNELLE ===
        is_initial_planning = not self.milestones
        
        # Choisir la méthode de génération appropriée
        if is_initial_planning:
            self.logger.info("Détection de la planification initiale. Génération sans RAG.")
            generate = self._generate_pure
        else:
            self.logger.info("Détection d'une ré-analyse. Utilisation du mode de génération complet avec RAG.")
            generate = self.generate_with_context
        # ==============================
        
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
            # Utilise la méthode de génération choisie (avec ou sans RAG)
            analysis = generate(
                prompt=analysis_prompt,
                temperature=0.6
            )
            
            # PHASE 1: Création du Project Charter
            self.logger.info("Création du Project Charter...")
            charter_prompt = f"""Reformule UNIQUEMENT la demande suivante en Project Charter structuré. 

PROMPT ORIGINAL: {project_prompt}

RÈGLES STRICTES:
- UNIQUEMENT reformuler/structurer ce qui est dans le PROMPT ORIGINAL
- INTERDICTION d'ajouter des détails non mentionnés
- INTERDICTION d'inventer des exemples ou spécifications
- Longueur max: {len(project_prompt) * 2} caractères
- Si une information n'existe pas dans le PROMPT ORIGINAL, écrire "Non spécifié"

Format obligatoire:
## Objectifs
- [Ce qui est demandé dans le prompt original seulement]

## Contraintes Techniques  
- [Seulement si mentionnées dans le prompt original, sinon "Non spécifié"]

## Contraintes Métier
- [Seulement si mentionnées dans le prompt original, sinon "Non spécifié"] 

## Spécifications
- [Seulement les éléments explicites du prompt original]

## Livrables
- [Seulement ce qui est explicitement demandé]

## Critères de Succès
- [Seulement si définis dans le prompt original, sinon "Fonctionnalité opérationnelle"]

"""
            
            try:
                # Utiliser la même logique conditionnelle pour le Charter
                project_charter_content = generate(
                    prompt=charter_prompt,
                    temperature=0.2  # Température basse pour synthèse factuelle
                )
                
                # Sauvegarder uniquement en fichier (source unique de vérité)
                charter_path = Path("projects") / self.project_name / "docs" / "PROJECT_CHARTER.md"
                charter_path.parent.mkdir(parents=True, exist_ok=True)
                charter_path.write_text(project_charter_content, encoding='utf-8')
                self.logger.info(f"Project Charter sauvegardé: {charter_path}")
                
                # Architecture unifiée : pas de stockage mémoire ni indexation RAG
                self.logger.info("Project Charter disponible uniquement via fichier (architecture unifiée)")
                    
            except Exception as e:
                self.logger.error(f"ÉCHEC CRITIQUE: Impossible de créer le Project Charter: {e}")
                raise RuntimeError(f"PROJET COMPROMIS: Échec de création du Project Charter pour {self.project_name}")
            
            # Créer les jalons basés sur l'analyse
            milestones = self._create_milestones_from_analysis(analysis, project_prompt, use_pure_generation=is_initial_planning)
            
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
                
                # PHASE 2: Vérification intelligente du jalon
                self.logger.info(f"Vérification du jalon {milestone['id']}...")
                verification_decision = self._verify_milestone_completion(milestone, milestone_result)
                
                # PHASE 3: Application de la décision de vérification
                self._apply_verification_decision(verification_decision, milestone)

                # Pause stratégique entre jalons
                #if self.current_milestone_index < len(self.milestones):
                #    self.logger.info("Traitement fin de jalon")
                #    time.sleep(5)        

            orchestration_result['status'] = 'completed'
            orchestration_result['ended_at'] = datetime.now().isoformat()
            
            # Générer le résumé final
            self._generate_project_summary(orchestration_result)
            
        except Exception as e:
            self.logger.error(f"Erreur pendant l'orchestration: {str(e)}")
            orchestration_result['status'] = 'error'
            orchestration_result['error'] = str(e)
        
        return orchestration_result
    
    def _generate_pure(self, prompt: str, **kwargs) -> str:
        """
        Effectue un appel LLM direct sans injection de contexte RAG.
        À utiliser uniquement pour la planification initiale.
        """
        self.logger.info("Exécution d'une génération LLM 'pure' (sans RAG) pour la planification.")
        llm = LLMFactory.create(model=self.llm_config['model'])
        
        agent_context = {
            'agent_name': self.name,
            'task_id': self.state.get('current_task_id'),
            'project_name': self.project_name,
            'agent_role': self.role
        }
        
        # Fusion des configurations : kwargs > config de l'agent
        # Exclure 'model' car déjà passé à LLMFactory.create()
        final_params = {k: v for k, v in self.llm_config.items() if k != 'model'}
        final_params.update(kwargs)
        
        messages = [{"role": "user", "content": prompt}]
        
        response = llm.generate_with_messages(
            messages=messages, 
            agent_context=agent_context, 
            **final_params
        )
        
        # Correction robuste: gérer le format de réponse structuré du modèle magistral
        if isinstance(response, list):
            # Extraire le contenu "text" de la réponse structurée
            text_content = None
            for item in response:
                if isinstance(item, str) and item.startswith('text="'):
                    # Format: text="contenu réel..."
                    text_content = item[6:]  # Enlever 'text="'
                    if text_content.endswith('"'):
                        text_content = text_content[:-1]  # Enlever '"' final
                    break
                elif isinstance(item, str) and 'text=' in item:
                    # Autre format possible
                    text_start = item.find('text="') + 6
                    text_end = item.rfind('"')
                    if text_start > 5 and text_end > text_start:
                        text_content = item[text_start:text_end]
                        break
                elif hasattr(item, 'text'):
                    # Format objet avec attribut text
                    text_content = item.text
                    break
            
            if text_content:
                response = text_content
                self.logger.info(f"Réponse directe structurée extraite: {len(response)} caractères")
            else:
                # Fallback: joindre tous les éléments
                response = '\n'.join(str(item) for item in response)
                self.logger.warning(f"Réponse directe liste non structurée, jointure: {len(response)} caractères")
        elif not isinstance(response, str):
            # Forcer la conversion en chaîne pour tous les autres types
            response = str(response)
        
        return response





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
    
    def _create_milestones_from_analysis(self, analysis: str, project_prompt: str, use_pure_generation: bool = False) -> List[Dict[str, Any]]:
        """Crée les jalons à partir de l'analyse."""
        # Prompt pour créer les jalons structurés
        milestone_prompt = f"""Basé sur la demande initiale ET le Project Charter formalisé:

--- DEMANDE INITIALE DE L'UTILISATEUR ---
{project_prompt}
--- FIN DE LA DEMANDE INITIALE ---

--- PROJECT CHARTER FORMALISÉ ---
{self._get_project_charter_from_file()}
--- FIN DU PROJECT CHARTER ---

--- ANALYSE INTERMÉDIAIRE ---
{analysis}
--- FIN DE L'ANALYSE ---

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
            if use_pure_generation:
                # Appel direct et pur pour le JSON
                json_prompt = f"{milestone_prompt}\n\nRéponds uniquement avec un JSON valide."
                raw_response = self._generate_pure(prompt=json_prompt, temperature=0.5)
                # Utiliser la méthode utilitaire partagée
                response = self._parse_json_from_llm_response(raw_response)
            else:
                # Utilise la méthode standard avec RAG pour la ré-analyse
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
            if milestone['status'] == 'completed':
                status = "✅"
            elif milestone['status'] == 'partially_completed':
                status = "⚠️"
            else:
                status = "⏳"
            
            summary += f"\n{status} **{milestone['name']}**\n"
            summary += f"   - Agents: {', '.join(milestone['agents_required'])}\n"
            summary += f"   - Livrables: {', '.join(milestone.get('deliverables', []))}\n"
            
            # Ajouter des détails pour les jalons partiellement complétés
            if milestone['status'] == 'partially_completed':
                reason = milestone.get('partial_completion_reason', 'Raison non spécifiée')
                attempts = milestone.get('correction_attempts', 0)
                summary += f"   - ⚠️ **Statut**: Partiellement complété ({attempts} tentatives de correction)\n"
                summary += f"   - **Raison**: {reason}\n"
        
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
        result = self.tools['get_progress_report']({'include_details': 'true'})
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
                    result = self.tools['add_milestone'](details)
                    if result.status == 'success':
                        self.logger.info(f"Jalon ajouté: {details.get('name')}")
                
                elif action == 'modify':
                    milestone_id = change.get('milestone_id')
                    details = change.get('details', {})
                    result = self.tools['modify_milestone']({
                        'milestone_id': milestone_id,
                        'changes': details
                    })
                    if result.status == 'success':
                        self.logger.info(f"Jalon {milestone_id} modifié")
                
                elif action == 'remove':
                    milestone_id = change.get('milestone_id')
                    result = self.tools['remove_milestone']({'milestone_id': milestone_id})
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
        """Reçoit un rapport d'un autre agent et gère le buffer par jalon."""
        super().receive_report(agent_name, report)

        # Détecter le jalon actuel depuis le rapport
        reported_milestone = self._extract_milestone_from_report(agent_name, report)
        
        # Reset du buffer si changement de jalon
        if reported_milestone != self.current_milestone_id:
            self.logger.info(f"Changement de jalon détecté: {self.current_milestone_id} → {reported_milestone}")
            self._reset_milestone_buffer(reported_milestone)
        
        # Stocker tous les rapports dans le buffer
        self.current_milestone_reports.append(report)
        self.logger.debug(f"Rapport de {agent_name} stocké dans le buffer ({len(self.current_milestone_reports)} rapports)")
        
        # Évaluer UNIQUEMENT sur rapport automatique
        content = report.get('content', {})
        if content.get('type') == 'automatic':
            self.logger.info(f"Rapport automatique de {agent_name} reçu - Déclenchement de l'évaluation du jalon")
            self._evaluate_milestone_with_context()
            # Ne pas reset le buffer ici - garde historique pour debugging
    
    def _extract_milestone_from_report(self, agent_name: str, report: Dict[str, Any]) -> str:
        """Extrait l'ID du jalon depuis un rapport."""
        # Essayer d'abord depuis le task_id
        task_id = report.get('task_id')
        if task_id and hasattr(self, 'agents') and agent_name in self.agents:
            agent = self.agents[agent_name]
            if hasattr(agent, 'current_milestone_id'):
                return agent.current_milestone_id
        
        # Fallback: utiliser l'index du jalon actuel
        if self.current_milestone_index < len(self.milestones):
            return self.milestones[self.current_milestone_index].get('milestone_id', f'milestone_{self.current_milestone_index}')
        
        return 'unknown'
    
    def _reset_milestone_buffer(self, new_milestone_id: str) -> None:
        """Reset le buffer des rapports pour un nouveau jalon."""
        self.current_milestone_id = new_milestone_id
        self.current_milestone_reports = []
        self.logger.debug(f"Buffer des rapports réinitialisé pour le jalon {new_milestone_id}")
    
    def _evaluate_milestone_with_context(self) -> None:
        """Évalue le jalon avec le contexte complet de tous les rapports."""
        try:
            # Séparer les rapports par type
            automatic_reports = [r for r in self.current_milestone_reports if r.get('content', {}).get('type') == 'automatic']
            manual_reports = [r for r in self.current_milestone_reports if r.get('content', {}).get('type') == 'manual']
            
            self.logger.info(f"Évaluation du jalon {self.current_milestone_id} avec {len(automatic_reports)} rapports automatiques et {len(manual_reports)} rapports manuels")
            
            # Analyser le dernier rapport automatique pour la décision technique
            if automatic_reports:
                latest_automatic = automatic_reports[-1]
                technical_assessment = latest_automatic.get('content', {}).get('self_assessment', 'unknown')
                
                # Évaluer seulement si problème technique détecté
                if technical_assessment != 'compliant':
                    self.logger.info(f"Évaluation nécessaire - Assessment technique: {technical_assessment}")
                    self._trigger_plan_evaluation_with_context(latest_automatic, manual_reports)
                else:
                    self.logger.info(f"Jalon {self.current_milestone_id} terminé avec succès - Pas d'évaluation nécessaire")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation du jalon: {str(e)}")
    
    def _trigger_plan_evaluation_with_context(self, automatic_report: Dict[str, Any], manual_reports: List[Dict[str, Any]]) -> None:
        """Déclenche l'évaluation du plan avec le contexte complet."""
        # Construire le contexte enrichi
        context_parts = [f"Rapport technique: {automatic_report.get('content', {}).get('message', 'Non spécifié')}"]
        
        if manual_reports:
            context_parts.append("Contexte des communications:")
            for i, manual_report in enumerate(manual_reports, 1):
                agent_name = manual_report.get('agent', 'Unknown')
                message = manual_report.get('content', {}).get('message', 'Non spécifié')
                context_parts.append(f"  {i}. {agent_name}: {message}")
        
        enriched_context = "\n".join(context_parts)
        self.logger.info(f"Déclenchement de l'évaluation avec contexte enrichi")
        self._evaluate_plan_after_interaction('milestone_completion', enriched_context)

    
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
            
            elif entry_type == "milestone_partially_completed":
                milestone_name = details.get('milestone_name', 'Unknown')
                attempts = details.get('correction_attempts', 0)
                entry = f"""
## {timestamp} - Jalon Partiellement Complété: {milestone_name}

**Résultat**: ⚠️ Jalon complété partiellement après {attempts} tentatives de correction
**Raison**: {content}
**Statut**: Fonctionnel mais incomplet

**Impact**: Le projet continue mais ce jalon pourrait nécessiter une attention future
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
    
    def _get_project_charter_from_file(self) -> str:
        """
        Architecture unifiée: Récupère le Project Charter depuis le fichier uniquement.
        Tous les agents (y compris Supervisor) fonctionnent de la même façon.
        """
        try:
            charter_path = Path("projects") / self.project_name / "docs" / "PROJECT_CHARTER.md"
            if charter_path.exists():
                charter = charter_path.read_text(encoding='utf-8')
                if charter and len(charter) > 50:
                    self.logger.info(f"Project Charter récupéré depuis le fichier: {charter_path}")
                    return charter
                else:
                    raise ValueError("Project Charter fichier vide ou trop court")
            else:
                raise FileNotFoundError(f"Project Charter non trouvé: {charter_path}")
                
        except Exception as e:
            self.logger.error(f"PROJET COMPROMIS: Impossible de lire le Project Charter: {str(e)}")
            raise RuntimeError(f"PROJET COMPROMIS: Project Charter inaccessible pour {self.project_name}: {str(e)}")
    
    def _verify_milestone_completion(self, milestone: Dict[str, Any], milestone_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 2: Vérification intelligente d'un jalon terminé.
        """
        try:
            # Collecter les rapports structurés des agents
            structured_reports = []
            for task in milestone_result.get('tasks_completed', []):
                if 'result' in task and 'structured_report' in task['result']:
                    structured_reports.append(task['result']['structured_report'])
            
            # Évaluation rapide basée sur les auto-évaluations
            if structured_reports:
                compliant_reports = [r for r in structured_reports if r.get('self_assessment') == 'compliant']
                partial_reports = [r for r in structured_reports if r.get('self_assessment') == 'partial']
                failed_reports = [r for r in structured_reports if r.get('self_assessment') == 'failed']
                
                # Logique de décision rapide
                if len(compliant_reports) == len(structured_reports):
                    # Tous les agents sont conformes
                    self.logger.info("Validation rapide réussie: tous les rapports sont conformes")
                    return {
                        'decision': 'approve',
                        'reason': f'Tous les agents ({len(compliant_reports)}) rapportent une conformité complète',
                        'confidence': 0.9,
                        'evaluation_type': 'fast'
                    }
                elif failed_reports:
                    # Au moins un échec critique
                    self.logger.warning(f"Validation rapide détecte {len(failed_reports)} échec(s) critique(s)")
                    # Passer à l'évaluation approfondie
                    return self._deep_milestone_evaluation(milestone, milestone_result, structured_reports)
                elif partial_reports or len(structured_reports) > 0:
                    # Succès partiels ou rapports disponibles - évaluation nuancée
                    self.logger.info(f"Validation avec {len(structured_reports)} rapports disponibles, {len(partial_reports)} succès partiels")
                    return self._deep_milestone_evaluation(milestone, milestone_result, structured_reports)
            
            # Vérifier si des agents ont terminé même sans rapports structurés
            agents_completed = milestone_result.get('agents_completed', [])
            if agents_completed:
                self.logger.info(f"Agents terminés détectés: {agents_completed}, validation par résultats")
                return {
                    'decision': 'approve',
                    'confidence': 0.8,
                    'reason': f'Completion confirmée pour agents: {", ".join(agents_completed)}',
                    'evaluation_type': 'agent_completion_based'
                }
            
            # Cas vraiment ambigus - évaluation approfondie
            self.logger.warning("Pas de rapports structurés fiables, lancement évaluation approfondie")
            return self._deep_milestone_evaluation(milestone, milestone_result, structured_reports)
            
        except Exception as e:
            self.logger.error(f"Erreur vérification jalon: {e}")
            # En cas d'erreur, approuver pour continuer
            return {
                'decision': 'approve',
                'reason': f'Erreur lors de la vérification: {e}',
                'confidence': 0.1,
                'evaluation_type': 'error_fallback'
            }
    
    def _deep_milestone_evaluation(self, milestone: Dict[str, Any], milestone_result: Dict[str, Any], 
                                  structured_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        PHASE 2: Évaluation approfondie utilisant le Project Charter et l'IA.
        """
        try:
            # Récupérer le Project Charter
            project_charter = self._get_project_charter_from_file()
            
            # Construire le contexte pour l'évaluation
            evaluation_context = f"""
PROJECT CHARTER:
{project_charter}

JALON ÉVALUÉ: {milestone['name']}
Description: {milestone['description']}
Livrables attendus: {', '.join(milestone.get('deliverables', []))}

RÉSULTATS OBTENUS:
Agents impliqués: {', '.join(milestone_result.get('agents_involved', []))}
Artefacts créés: {len([artifact for task in milestone_result.get('tasks_completed', []) for artifact in task.get('artifacts', [])])}

RAPPORTS D'AUTO-ÉVALUATION:
"""
            for report in structured_reports:
                evaluation_context += f"- Agent {report.get('agent_name', 'unknown')}: {report.get('self_assessment', 'unknown')} (confiance: {report.get('confidence_level', 0):.1f})\n"
                if report.get('issues_encountered'):
                    evaluation_context += f"  Problèmes: {'; '.join(report['issues_encountered'])}\n"
            
            # Prompt d'évaluation intelligente
            evaluation_prompt = f"""Tu es le superviseur du projet. Évalue si ce jalon respecte les objectifs du Project Charter.

{evaluation_context}

Analyse:
1. Les livrables attendus sont-ils créés selon le Charter?
2. Y a-t-il des déviations par rapport aux objectifs?
3. Les problèmes rencontrés compromettent-ils la suite?

Réponds avec un JSON:
{{
    "decision": "approve|request_rework|adjust_plan",
    "reason": "explication détaillée",
    "confidence": 0.0-1.0,
    "recommended_action": "action spécifique si nécessaire"
}}"""
            
            # Génération de l'évaluation
            evaluation_response = self.generate_json_with_context(
                prompt=evaluation_prompt,
                temperature=0.4
            )
            
            # Validation et enrichissement de la réponse
            decision = evaluation_response.get('decision', 'approve')
            if decision not in ['approve', 'request_rework', 'adjust_plan']:
                decision = 'approve'  # Sécurité
            
            result = {
                'decision': decision,
                'reason': evaluation_response.get('reason', 'Évaluation IA complétée'),
                'confidence': max(0.0, min(1.0, evaluation_response.get('confidence', 0.5))),
                'evaluation_type': 'deep_ai',
                'recommended_action': evaluation_response.get('recommended_action', ''),
                'charter_compliance': 'evaluated'
            }
            
            self.logger.info(f"Évaluation approfondie terminée: {decision} (confiance: {result['confidence']:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation approfondie: {e}")
            # Fallback conservateur
            return {
                'decision': 'approve',
                'reason': f'Erreur évaluation approfondie: {e}',
                'confidence': 0.2,
                'evaluation_type': 'error_fallback'
            }
    
    def _apply_verification_decision(self, verification: Dict[str, Any], current_milestone: Dict[str, Any]) -> None:
        """
        PHASE 3: Applique la décision de vérification en utilisant les outils existants.
        """
        decision = verification.get('decision', 'approve')
        reason = verification.get('reason', 'Aucune raison spécifiée')
        confidence = verification.get('confidence', 0.5)
        
        # Marquer le milestone avec les informations de vérification
        current_milestone['verification_status'] = decision
        current_milestone['verification_timestamp'] = datetime.now().isoformat()
        current_milestone['verification_confidence'] = confidence
        
        # Gestion des tentatives de correction pour éviter boucles infinites
        correction_count = current_milestone.get('correction_attempts', 0)
        max_corrections = self.max_corrections
        
        if decision == 'approve':
            # Jalon approuvé - continuer normalement
            self.logger.info(f"Jalon '{current_milestone['name']}' approuvé (confiance: {confidence:.2f})")
            
            # Journalisation de l'approbation
            self._create_journal_entry(
                'milestone_approved',
                reason,
                {
                    'milestone_name': current_milestone['name'],
                    'confidence': confidence,
                    'evaluation_type': verification.get('evaluation_type', 'unknown')
                }
            )
            
            # Avancer au jalon suivant
            self.current_milestone_index += 1
            self.project_state['milestones_completed'] += 1
            current_milestone['status'] = 'completed'
            
        elif decision == 'request_rework' and correction_count < max_corrections:
            # Vérifier le nombre de corrections imbriquées pour éviter les boucles infinies
            supervisor_config = default_config.get('agents', {}).get('supervisor', {})
            max_nested_corrections = supervisor_config.get('max_nested_corrections', 2)
            correction_depth = current_milestone['name'].count('Correction:')
            if correction_depth >= max_nested_corrections:
                self.logger.error(f"Trop de corrections imbriquées ({correction_depth}), approbation forcée pour éviter boucle infinie")
                self._force_milestone_approval(current_milestone, "Limite de corrections imbriquées atteinte")
                return
            
            # Demande de correction
            self.logger.warning(f"Correction requise pour '{current_milestone['name']}' (tentative {correction_count + 1}/{max_corrections})")
            
            # Incrémenter le compteur de corrections
            current_milestone['correction_attempts'] = correction_count + 1
            
            # Ajouter un jalon de correction via l'outil existant
            correction_result = self.tools['add_milestone']({
                'after_milestone_id': current_milestone['id'],
                'name': f"Correction: {current_milestone['name']}",
                'description': f"Action corrective requise: {reason}",
                'agents_required': current_milestone['agents_required'],
                'deliverables': ["Rapport de correction", "Artifacts corrigés"]
            })
            
            if correction_result.status == 'success':
                self.logger.info("Jalon de correction ajouté avec succès")
                # NE PAS incrémenter current_milestone_index - le prochain jalon sera la correction
            else:
                self.logger.error(f"Échec ajout jalon correction: {correction_result.error}")
                # Forcer l'approbation en cas d'échec
                self._force_milestone_approval(current_milestone, "Échec ajout correction")
            
        elif decision == 'request_rework' and correction_count >= max_corrections:
            # Trop de tentatives de correction - stratégie d'échec gracieuse
            self.logger.error(f"Trop de tentatives de correction ({correction_count}), passage en mode partiellement complété")
            self._mark_milestone_partially_completed(current_milestone, f"Limite de corrections atteinte ({max_corrections}) - Jalon partiellement fonctionnel")
            
        elif decision == 'adjust_plan':
            # SÉQUENCE CORRIGÉE POUR ÉVITER DE SAUTER DES JALONS
            self.logger.info(f"Ajustement du plan requis suite au jalon '{current_milestone['name']}'")
            
            # ÉTAPE 1: Finaliser le jalon déclencheur AVANT adjust_plan
            # Ceci évite le conflit d'index avec _force_milestone_approval
            current_milestone['status'] = 'completed'
            current_milestone['triggered_plan_adjustment'] = True
            current_milestone['adjustment_reason'] = reason
            
            # Avancer l'index et incrémenter le compteur
            self.current_milestone_index += 1
            self.project_state['milestones_completed'] += 1
            
            # Journaliser la finalisation du jalon déclencheur
            self._create_journal_entry(
                'milestone_completed_trigger_adjustment',
                f"Jalon '{current_milestone['name']}' complété et déclenche l'ajustement: {reason}",
                {
                    'milestone_name': current_milestone['name'],
                    'trigger_reason': reason,
                    'milestones_completed': self.project_state['milestones_completed']
                }
            )
            
            # ÉTAPE 2: Maintenant ajuster le plan 
            # adjust_plan va recalculer current_milestone_index = len(completed_milestones)
            # ce qui le repositionne correctement sur le premier nouveau jalon
            self.adjust_plan(reason)
            
            # Note: Pas besoin de _force_milestone_approval - nous avons géré manuellement
            # L'atomicité de l'opération est garantie
            
        else:
            # Cas non géré - approuver par défaut
            self.logger.warning(f"Décision non reconnue '{decision}', approbation par défaut")
            self._force_milestone_approval(current_milestone, f"Décision inconnue: {decision}")
    
    def _force_milestone_approval(self, milestone: Dict[str, Any], reason: str) -> None:
        """
        PHASE 3: Force l'approbation d'un jalon et continue l'orchestration.
        """
        self.logger.info(f"Approbation forcée du jalon '{milestone['name']}': {reason}")
        
        # Journaliser l'approbation forcée
        self._create_journal_entry(
            'milestone_forced_approval',
            reason,
            {'milestone_name': milestone['name']}
        )
        
        # Marquer comme complété et avancer
        milestone['status'] = 'completed'
        milestone['forced_approval'] = True
        milestone['forced_approval_reason'] = reason
        
        self.current_milestone_index += 1
        self.project_state['milestones_completed'] += 1
    
    def _mark_milestone_partially_completed(self, milestone: Dict[str, Any], reason: str) -> None:
        """
        Marque un jalon comme partiellement complété après plusieurs échecs.
        Stratégie d'échec gracieuse pour éviter les blocages.
        """
        self.logger.warning(f"Jalon '{milestone['name']}' marqué comme partiellement complété: {reason}")
        
        # Journaliser la completion partielle
        self._create_journal_entry(
            'milestone_partially_completed',
            reason,
            {
                'milestone_name': milestone['name'],
                'correction_attempts': milestone.get('correction_attempts', 0),
                'status': 'partial'
            }
        )
        
        # Marquer comme partiellement complété et avancer
        milestone['status'] = 'partially_completed'
        milestone['partial_completion_reason'] = reason
        milestone['completion_level'] = 'partial'
        
        self.current_milestone_index += 1
        self.project_state['milestones_completed'] += 1
        
        # Créer une note pour les jalons suivants
        if self.rag_engine:
            self.rag_engine.index_to_working_memory(
                f"Jalon {milestone['name']} partiellement complété: {reason}",
                {
                    'type': 'milestone_partial_completion',
                    'agent_name': self.name,
                    'priority': 'high',
                    'milestone_name': milestone['name']
                }
            )
    
    def adjust_plan(self, reason: str) -> None:
        """
        Réajuste le plan complet suite à une décision adjust_plan.
        Préserve les jalons complétés et régénère les jalons restants.
        
        ATTENTION: Cette méthode assume que le jalon déclencheur est déjà finalisé
        avant l'appel (status='completed', index incrémenté).
        
        Gestion des cas limites :
        - 0 nouveaux jalons générés : le projet se termine naturellement
        - Tous jalons complétés : le projet se termine 
        - Jalons partially_completed : conservés comme les completed
        """
        try:
            self.logger.info(f"Début adjust_plan: {reason}")
            
            # 1. Sauvegarder les jalons complétés/partiels avec toutes leurs métadonnées
            completed_milestones = []
            for m in self.milestones:
                if m.get('status') in ['completed', 'partially_completed']:
                    completed_milestones.append(m.copy())  # Préserver toutes les métadonnées
            
            completed_names = [m['name'] for m in completed_milestones]
            self.logger.info(f"Jalons préservés ({len(completed_milestones)}): {completed_names}")
            
            # 2. Prompt de régénération (réutilise pattern _create_milestones_from_analysis)
            regeneration_prompt = f"""Le plan de projet initial doit être ajusté à cause de : {reason}

Les jalons suivants sont déjà complétés et immuables : {completed_names}

En te basant sur le prompt original et le Project Charter, régénère la suite du plan (les jalons restants) pour atteindre l'objectif final.

--- DEMANDE INITIALE DE L'UTILISATEUR ---
{self.project_prompt}
--- FIN DE LA DEMANDE INITIALE ---

--- PROJECT CHARTER FORMALISÉ ---
{self._get_project_charter_from_file()}
--- FIN DU PROJECT CHARTER ---

Réponds uniquement avec un JSON contenant la nouvelle liste des jalons futurs:
{{
    "milestones": [
        {{
            "id": "sera recalculé automatiquement", 
            "name": "Nom du jalon", 
            "description": "Description détaillée", 
            "agents_required": ["analyst", "developer"], 
            "deliverables": ["livrable1", "livrable2"],
            "estimated_duration": "durée estimée",
            "dependencies": []
        }}
    ]
}}"""

            # 3. Générer nouveaux jalons (même logique conditionnelle que création initiale)
            try:
                # Utiliser génération pure ou avec contexte selon l'état
                is_initial_context = len(completed_milestones) == 0
                if is_initial_context:
                    raw_response = self._generate_pure(prompt=regeneration_prompt, temperature=0.5)
                    new_milestones_data = self._parse_json_from_llm_response(raw_response)
                else:
                    new_milestones_data = self.generate_json_with_context(
                        prompt=regeneration_prompt, temperature=0.5
                    )
            except Exception as e:
                self.logger.error(f"Erreur génération nouveaux jalons: {e}")
                # Fallback: garder le plan existant et continuer
                self._create_journal_entry(
                    'adjust_plan_failed',
                    f"Échec régénération jalons: {e}. Plan conservé.",
                    {'trigger_reason': reason, 'error': str(e)}
                )
                return
            
            # 4. Traitement et validation des nouveaux jalons
            new_milestones_raw = new_milestones_data.get('milestones', [])
            
            # Cas limite : 0 nouveaux jalons = projet se termine naturellement
            if not new_milestones_raw:
                self.logger.info("Aucun nouveau jalon généré - projet se termine")
                self.milestones = completed_milestones
                self.current_milestone_index = len(completed_milestones)  # Index au-delà du dernier
                self._create_journal_entry(
                    'plan_adjustment_no_new_milestones',
                    f"Plan ajusté sans nouveaux jalons: {reason}. Projet se termine.",
                    {'trigger_reason': reason, 'completed_milestones_count': len(completed_milestones)}
                )
                return
            
            # Générer IDs séquentiels pour éviter conflits
            next_id = max([m['id'] for m in completed_milestones], default=0) + 1
            new_milestones = []
            
            for i, m in enumerate(new_milestones_raw):
                validated_milestone = {
                    'id': next_id + i,
                    'milestone_id': f'milestone_{next_id + i}',
                    'name': m.get('name', f'Jalon {next_id + i}'),
                    'description': m.get('description', ''),
                    'agents_required': [a for a in m.get('agents_required', ['analyst']) 
                                      if a in ['analyst', 'developer']],
                    'deliverables': m.get('deliverables', []),
                    'estimated_duration': m.get('estimated_duration', 'À estimer'),
                    'dependencies': m.get('dependencies', []),
                    'status': 'pending'
                }
                
                # Validation supplémentaire
                if not validated_milestone['agents_required']:
                    validated_milestone['agents_required'] = ['analyst']
                
                new_milestones.append(validated_milestone)
            
            # 5. Reconstruction complète de la liste des jalons
            self.milestones = completed_milestones + new_milestones
            
            # 6. **CRITIQUE**: Recalcul de l'index courant basé sur les jalons complétés
            # L'index doit pointer sur le premier nouveau jalon
            self.current_milestone_index = len(completed_milestones)
            
            self.logger.info(f"Index recalculé: {self.current_milestone_index} (premier nouveau jalon)")
            
            # 7. Journalisation complète avec métriques détaillées
            self._create_journal_entry(
                'plan_adjustment',
                f"Plan ajusté: {reason}. {len(completed_milestones)} jalons préservés, {len(new_milestones)} nouveaux jalons générés.",
                {
                    'trigger_reason': reason,
                    'completed_milestones_count': len(completed_milestones),
                    'new_milestones_count': len(new_milestones),
                    'total_milestones': len(self.milestones),
                    'current_index': self.current_milestone_index,
                    'completed_milestones': completed_names,
                    'new_milestones': [m['name'] for m in new_milestones]
                }
            )
            
            # 8. Mise à jour RAG avec contexte enrichi
            self._update_plan_in_rag(f"Plan ajusté ({reason}): {len(new_milestones)} nouveaux jalons")
            
            self.logger.info(f"adjust_plan terminé avec succès: {len(new_milestones)} nouveaux jalons, index={self.current_milestone_index}")
            
        except Exception as e:
            self.logger.error(f"ERREUR CRITIQUE adjust_plan: {e}", exc_info=True)
            # Stratégie de fallback: journaliser et continuer avec plan existant
            self._create_journal_entry(
                'adjust_plan_critical_error',
                f"Erreur critique lors de l'ajustement: {e}. Plan conservé pour éviter corruption.",
                {'trigger_reason': reason, 'critical_error': str(e)}
            )