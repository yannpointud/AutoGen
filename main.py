#!/usr/bin/env python3
"""
Point d'entrée principal de la plateforme multi-agents.
Ce script gère l'interface utilisateur en ligne de commande (CLI),
la sélection de projet, la configuration de l'exécution et
l'orchestration du démarrage du Superviseur.
"""

import sys
import os
from pathlib import Path
import time
from typing import Dict, Any, Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from core.project_manager import ProjectManager
from agents.supervisor import Supervisor
from core.metrics_visualizer import MetricsVisualizer
from utils.logger import setup_logger

# Charger les variables d'environnement
load_dotenv()

# Console Rich pour l'affichage
console = Console()
logger = setup_logger("main")

# Exemples de projets prédéfinis
PROJECT_TEMPLATES = {
    "1": {
        "name": "MLPricePredictor",
        "description": "API ML pour prédiction prix immobilier",
        "prompt": """Nom du Projet : MLPricePredictor

Objectif Général :
Créer une API REST complète en Python pour prédire le prix de biens immobiliers en se basant sur leurs caractéristiques. Le projet doit couvrir l'ensemble du cycle de vie du Machine Learning, de la préparation des données au déploiement d'une API de prédiction.

Stack Technique Imposée :
- Framework API : FastAPI
- Serveur Web : Uvicorn
- Bibliothèques ML : Scikit-learn, Pandas
- Validation de Données : Pydantic (intégré à FastAPI)
- Tests : Pytest

Source des Données (Description) :
Le modèle devra être entraîné sur un fichier CSV qui sera nommé housing_data.csv. Ce fichier contiendra les colonnes suivantes :
- surface_m2 (numérique, ex: 85.0)
- nb_pieces (entier, ex: 4)
- nb_sdb (entier, ex: 2)
- etage (entier, ex: 3)
- ville (catégorielle, ex: "Paris", "Lyon", "Marseille")
- a_renover (booléen, ex: True/False)
- prix_m2 (numérique, la cible à prédire, ex: 9500.0)

NOTE IMPORTANTE POUR LES AGENTS : Le fichier housing_data.csv n'existe pas. Une des premières tâches du Développeur sera de générer un fichier CSV de données factices (dummy_data.csv) avec une centaine de lignes respectant ce format pour permettre l'entraînement et les tests.

Fonctionnalités Détaillées et Exigences :

1. Script de Préparation des Données (data_processing.py) :
- Charger les données depuis le fichier CSV
- Gérer les valeurs manquantes de manière simple (remplacer par moyenne/médiane pour numérique, mode pour catégoriel)
- Effectuer l'ingénierie des caractéristiques (feature engineering) :
  * Transformer les variables catégorielles (ville) en variables numériques avec One-Hot Encoding
  * Transformer la variable booléenne (a_renover) en 0 ou 1
- Séparer les données en caractéristiques (X) et en cible (y)

2. Script d'Entraînement du Modèle (model_training.py) :
- Utiliser les fonctions de data_processing.py
- Diviser les données en ensemble d'entraînement et de test (80% / 20%)
- Utiliser RandomForestRegressor de Scikit-learn
- Entraîner le modèle sur l'ensemble d'entraînement
- Évaluer la performance (score R² et erreur quadratique moyenne - MSE)
- CRUCIAL : Sauvegarder le modèle entraîné et l'encodeur One-Hot dans des fichiers (model.pkl et encoder.pkl avec joblib)

3. API REST avec FastAPI (main.py) :
- Charger le modèle et l'encodeur sauvegardés au démarrage
- Créer un endpoint POST /predict
- Accepter JSON d'entrée avec caractéristiques du bien (sans le prix)
- Utiliser modèle Pydantic pour validation des données d'entrée
- Exemple JSON entrée : {"surface_m2": 120.0, "nb_pieces": 5, "nb_sdb": 2, "etage": 4, "ville": "Paris", "a_renover": false}
- Transformer les données avec l'encodeur chargé puis prédire avec le modèle
- Retourner la prédiction du prix total (prix_m2 prédit * surface_m2) en JSON
- Exemple JSON sortie : {"prediction_prix_total": 1140000.0}

4. Tests Unitaires (tests/test_api.py) :
- Créer au moins un test pour l'endpoint /predict
- Vérifier statut 200 OK et présence de prediction_prix_total avec valeur numérique

5. Documentation et Configuration :
- Générer requirements.txt avec toutes les dépendances
- Créer README.md détaillé avec installation, lancement API et exemple curl"""
    },
    "2": {
        "name": "Calculator",
        "description": "Calculatrice scientifique",
        "prompt": """Développer une calculatrice en Python avec :
- Opérations de base (+, -, *, /)
- Interface en ligne de commande interactive
- Tests unitaires complets
- Documentation des fonctions"""
    },
    "3": {
        "name": "FileOrganizer",
        "description": "Organisateur de fichiers automatique",
        "prompt": """Créer un outil d'organisation automatique de fichiers qui :
- Trie les fichiers par type (documents, images, vidéos, etc.)
- Renomme les fichiers selon des patterns configurables
- Détecte et gère les doublons
- Crée une structure de dossiers organisée
- Génère un rapport des actions effectuées
- Mode simulation (dry-run) avant exécution
- Configuration via fichier YAML"""
    },
    "4": {
        "name": "ChatBot",
        "description": "Chatbot assistant simple",
        "prompt": """Créer un chatbot assistant capable de :
- Répondre à des questions simples
- Maintenir le contexte de conversation
- Intégrer avec une base de connaissances
- Sauvegarder l'historique des conversations
- Personnalités configurables
- Interface CLI interactive
- Tests des différents scénarios"""
    }
}


class AutoGenMain:
    """Gestionnaire de démarrage rapide."""
    
    def __init__(self):
        self.console = console
        self.project_manager = ProjectManager()
    
    def display_welcome(self):
        """Affiche l'écran de bienvenue."""
        welcome_text = """
        🚀 Plateforme Multi-Agents IA - AutoGen
        
        Orchestrez votre équipe d'agents IA pour créer des projets complets.
        Choisissez un template ou créez votre propre projet !
        """
        
        self.console.print(Panel(welcome_text, title="Bienvenue", style="bold blue"))
    
    def check_api_key(self) -> bool:
        """Vérifie la présence des clés API."""
        mistral_key = os.getenv("MISTRAL_API_KEY")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not mistral_key and not deepseek_key:
            self.console.print("[bold red]❌ Aucune clé API trouvée ![/bold red]")
            self.console.print("\nPour configurer :")
            self.console.print("1. Copiez .env.example vers .env")
            self.console.print("2. Ajoutez votre clé API Mistral ou DeepSeek")
            self.console.print("\nObtenez une clé sur :")
            self.console.print("- Mistral: https://console.mistral.ai/")
            self.console.print("- DeepSeek: https://platform.deepseek.com/")
            return False
        
        if mistral_key:
            self.console.print("[green]✅ Clé API Mistral détectée[/green]")
        if deepseek_key:
            self.console.print("[green]✅ Clé API DeepSeek détectée[/green]")
        
        return True
    
    def display_templates(self):
        """Affiche les templates disponibles."""
        table = Table(title="Templates de Projets Disponibles", show_header=True)
        table.add_column("N°", style="cyan", width=4)
        table.add_column("Nom", style="magenta", width=20)
        table.add_column("Description", style="white")
        
        for key, template in PROJECT_TEMPLATES.items():
            table.add_row(key, template["name"], template["description"])
        
        table.add_row("0", "Custom", "Créer votre propre projet")
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")
    
    def get_project_choice(self) -> Dict[str, Any]:
        """Obtient le choix de projet de l'utilisateur."""
        choice = Prompt.ask(
            "[bold cyan]Choisissez un template (0, 1, 2, 3, 4)[/bold cyan]",
            choices=["0", "1", "2", "3", "4"],
            default="0"
        )
        
        if choice == "0":
            # Projet custom
            name = Prompt.ask("[bold cyan]Nom du projet[/bold cyan]")
            self.console.print("\n[bold cyan]Description du projet[/bold cyan]")
            self.console.print("[dim]Décrivez ce que vous voulez créer...[/dim]")
            
            lines = []
            self.console.print("[dim](Appuyez sur Entrée deux fois pour terminer)[/dim]\n")
            
            empty_lines = 0
            while empty_lines < 2:
                line = input()
                if line:
                    lines.append(line)
                    empty_lines = 0
                else:
                    empty_lines += 1
            
            prompt = '\n'.join(lines).strip()
            
            return {
                "name": name,
                "description": "Projet personnalisé",
                "prompt": prompt
            }
        else:
            return PROJECT_TEMPLATES[choice]
    
    def display_execution_options(self) -> Dict[str, bool]:
        """Affiche les options d'exécution."""
        self.console.print("\n[bold]Options d'exécution[/bold]")
        
        options = {
            "auto_execute": Confirm.ask(
                "Exécuter automatiquement tous les jalons ?",
                default=True
            ),
            "generate_metrics": Confirm.ask(
                "Générer le dashboard de métriques ?",
                default=True
            ),
            "verbose": Confirm.ask(
                "Mode verbose (afficher tous les détails) ?",
                default=True
            )
        }
        
        return options
    
    def execute_project(self, project: Dict[str, Any], options: Dict[str, bool]):
        """Exécute le projet avec les options choisies."""
        project_name = project["name"]
        project_prompt = project["prompt"]
        
        # Créer la structure du projet
        self.console.print(f"\n[bold]📁 Création du projet {project_name}...[/bold]")
        project_path = self.project_manager.create_project_structure(project_name)
        
        if not project_path:
            self.console.print("[red]❌ Échec de la création du projet[/red]")
            return
        
        self.console.print(f"[green]✅ Projet créé dans : {project_path}[/green]")
        
        # Initialiser le superviseur
        self.console.print("\n[bold]🤖 Initialisation de l'équipe d'agents...[/bold]")
        
        try:
            # 1. Importer la classe RAGEngine si ce n'est pas déjà fait
            from core.rag_engine import RAGEngine

            # 2. Créer l'instance du moteur RAG pour ce projet
            rag_engine = RAGEngine(project_name=project_name)
            self.console.print("[green]✅ Moteur RAG initialisé[/green]")

            # 3. Passer l'instance RAG au Superviseur lors de sa création
            supervisor = Supervisor(
                project_name=project_name, 
                project_prompt=project_prompt,
                rag_engine=rag_engine  # <-- Ajout crucial
            )
            self.console.print("[green]✅ Superviseur initialisé et connecté au RAG[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Erreur : {str(e)}[/red]")
            return




        # Planification
        self.console.print("\n[bold]📋 Planification du projet...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Analyse en cours...", total=None)
            
            plan = supervisor.think({'prompt': project_prompt})
            progress.update(task, completed=True)
        
        # Afficher l'analyse
        if plan.get('project_analysis'):
            analysis = plan['project_analysis']
            self.console.print("\n[bold]📊 Analyse du projet :[/bold]")
            self.console.print(f"  • Type : {analysis.get('project_type', 'N/A')}")
            self.console.print(f"  • Complexité : {analysis.get('complexity', 'N/A')}")
            self.console.print(f"  • Durée estimée : {analysis.get('estimated_duration', 'N/A')}")
        
        # Préparation
        self.console.print("\n[bold]👥 Préparation de l'équipe...[/bold]")
        prep_result = supervisor.act(plan)
        
        self.console.print(f"[green]✅ {prep_result['agents_created']} agents créés[/green]")
        self.console.print(f"[green]✅ {prep_result['milestones_created']} jalons définis[/green]")
        
        # Afficher les jalons
        self.console.print("\n[bold]🎯 Jalons du projet :[/bold]")
        for i, milestone in enumerate(supervisor.milestones, 1):
            self.console.print(f"\n{i}. [cyan]{milestone['name']}[/cyan]")
            self.console.print(f"   {milestone.get('description', 'N/A')}")
            self.console.print(f"   Agents : {', '.join(milestone['agents_required'])}")
        
        # Exécution automatique si demandée
        if options["auto_execute"]:
            if Confirm.ask("\n[bold yellow]Lancer l'exécution automatique ?[/bold yellow]", default=True):
                self.console.print("\n[bold]🚀 Lancement de l'orchestration complète...[/bold]")
                self.console.print("[dim]Les agents vont maintenant travailler. Cela peut prendre plusieurs minutes.[/dim]")
                
                # Lancement de l'orchestration complète
                final_results = supervisor.orchestrate()
                
                self.console.print("\n[bold green]✨ Orchestration terminée ![/bold green]")
                
                # Afficher un résumé détaillé des résultats
                milestone_results = final_results.get('milestones_results', [])
                completed_count = sum(1 for r in milestone_results if r.get('status') == 'completed')
                self.console.print(f"  • {completed_count}/{len(milestone_results)} jalons complétés.")
                
                # Ajouter le rapport de progression détaillé du superviseur
                try:
                    report = supervisor.get_progress_report()
                    self.console.print(f"  • Progression : {report['progress_percentage']:.0f}%")
                    self.console.print(f"  • Statut : {report['status']}")
                    self.console.print(f"  • Modèle LLM : {report.get('llm_model', 'N/A')}")
                except Exception as e:
                    self.console.print(f"  • [dim]Rapport détaillé non disponible: {str(e)}[/dim]")
        
        # Génération des métriques
        if options["generate_metrics"]:
            self.console.print("\n[bold]📊 Génération du dashboard...[/bold]")
            
            visualizer = MetricsVisualizer(project_name)
            dashboard_path = visualizer.generate_dashboard(supervisor.rag_singleton)
            
            self.console.print(f"[green]✅ Dashboard généré : {dashboard_path}[/green]")
            
            # Ouvrir automatiquement le dashboard
            if Confirm.ask("Ouvrir le dashboard dans le navigateur ?", default=True):
                import webbrowser
                webbrowser.open(f"file://{Path(dashboard_path).absolute()}")
        
        # Résumé final
        self.console.print("\n[bold green]🎉 Projet terminé ![/bold green]")
        self.console.print(f"\n📁 Fichiers du projet : {project_path}")
        self.console.print("📄 Artefacts générés :")
        
        # Lister les fichiers créés
        for folder in ['docs', 'src', 'tests']:
            folder_path = project_path / folder
            if folder_path.exists():
                files = list(folder_path.glob('*'))
                if files:
                    self.console.print(f"\n  [{folder}/]")
                    for file in files[:5]:  # Limiter l'affichage
                        self.console.print(f"    • {file.name}")
                    if len(files) > 5:
                        self.console.print(f"    ... et {len(files) - 5} autres fichiers")
    
    def run(self):
        """Lance le processus de démarrage rapide."""
        try:
            # Bienvenue
            self.display_welcome()
            
            # Vérifier les clés API
            if not self.check_api_key():
                return
            
            # Afficher les templates
            self.display_templates()
            
            # Obtenir le choix
            project = self.get_project_choice()
            
            # Confirmer
            self.console.print("\n[bold]Projet sélectionné :[/bold]")
            self.console.print(f"Nom : [cyan]{project['name']}[/cyan]")
            self.console.print(f"Description : {project['description']}")
            
            if not Confirm.ask("\nContinuer avec ce projet ?", default=True):
                self.console.print("[yellow]Annulé par l'utilisateur[/yellow]")
                return
            
            # Options d'exécution
            options = self.display_execution_options()
            
            # Exécuter
            self.execute_project(project, options)
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrompu par l'utilisateur[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Erreur : {str(e)}[/red]")
            logger.error(f"Erreur dans main: {str(e)}", exc_info=True)


def main():
    """Point d'entrée principal."""
    autogen = AutoGenMain()
    autogen.run()


if __name__ == "__main__":
    main()
