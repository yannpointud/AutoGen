"""
Interface en ligne de commande pour l'interaction avec l'utilisateur.
Gère les entrées/sorties et l'affichage des informations.
"""

import re
from typing import Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text


class CLIInterface:
    """
    Gère l'interface en ligne de commande pour l'interaction utilisateur.
    """
    
    def __init__(self):
        """Initialise la console Rich pour un affichage amélioré."""
        self.console = Console()
    
    def display_welcome(self) -> None:
        """Affiche le message de bienvenue."""
        welcome_text = Text("🤖 Plateforme Multi-Agents IA 🤖", style="bold cyan")
        panel = Panel(
            welcome_text,
            title="Bienvenue",
            subtitle="Conception, Développement, Test & Documentation",
            style="bold blue"
        )
        self.console.print(panel)
        self.console.print()
    
    def get_project_name(self) -> str:
        """
        Demande et valide le nom du projet.
        
        Returns:
            str: Nom du projet validé
        """
        while True:
            name = Prompt.ask(
                "[bold cyan]Nom du projet[/bold cyan]",
                default="MyAwesomeApp"
            )
            
            # Valider le nom (alphanumériques, underscores et tirets)
            if re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
                return name
            else:
                self.display_error(
                    "Le nom du projet doit commencer par une lettre et "
                    "contenir uniquement des lettres, chiffres, tirets et underscores."
                )
    
    def get_project_prompt(self) -> str:
        """
        Demande la description du projet.
        
        Returns:
            str: Description/prompt du projet
        """
        self.console.print("\n[bold cyan]Description du projet[/bold cyan]")
        self.console.print("[dim]Décrivez ce que vous souhaitez créer...[/dim]")
        
        # Utiliser une entrée multiligne
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
        
        if not prompt:
            prompt = Prompt.ask(
                "[yellow]Aucune description fournie. Entrez une description[/yellow]"
            )
        
        return prompt
    
    def display_project_created(self, project_name: str, project_path: str) -> None:
        """
        Affiche un message de confirmation de création du projet.
        
        Args:
            project_name: Nom du projet
            project_path: Chemin du projet créé
        """
        success_panel = Panel(
            f"✅ Projet '[bold green]{project_name}[/bold green]' créé avec succès!\n\n"
            f"📁 Chemin: {project_path}",
            title="Succès",
            style="green"
        )
        self.console.print(success_panel)
    
    def display_info(self, message: str) -> None:
        """
        Affiche un message d'information.
        
        Args:
            message: Message à afficher
        """
        self.console.print(f"[blue]ℹ️  {message}[/blue]")
    
    def display_error(self, message: str) -> None:
        """
        Affiche un message d'erreur.
        
        Args:
            message: Message d'erreur à afficher
        """
        self.console.print(f"[red]❌ {message}[/red]")
    
    def display_warning(self, message: str) -> None:
        """
        Affiche un message d'avertissement.
        
        Args:
            message: Message d'avertissement à afficher
        """
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")
    
    def ask_confirmation(self, question: str, default: bool = True) -> bool:
        """
        Demande une confirmation à l'utilisateur.
        
        Args:
            question: Question à poser
            default: Valeur par défaut
            
        Returns:
            bool: Réponse de l'utilisateur
        """
        return Confirm.ask(f"[cyan]{question}[/cyan]", default=default)
    
    def display_progress(self, message: str) -> None:
        """
        Affiche un message de progression.
        
        Args:
            message: Message de progression
        """
        self.console.print(f"[dim]⏳ {message}...[/dim]")
