# Changelog

Toutes les modifications notables de ce projet sont documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Versioning Sémantique](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-08-16

### 🎉 Ajouté
- **Système de compression mémoire conversationnelle** : Nouveau système automatique de compression des historiques de conversation
  - **Seuil configurable** : `conversation_compression_threshold` (50000 chars par défaut)
  - **Mémoire courte préservée** : Conservation des `conversation_memory_size` derniers messages (2 par défaut)
  - **Compression intelligente** : Utilisation du service LLM léger pour résumer les anciens échanges
  - **Logs détaillés** : Affichage précis des tailles avant/après compression avec différence

### 🔧 Amélioré
- **Logs LLM debug** : Redirection des logs vers `projects/{project}/logs/llm_debug/` pour isolation par projet
- **Gestion mémoire agents** : Suppression de la troncature automatique (maxlen) au profit du système de compression
- **Configuration centralisée** : Nouveaux paramètres de compression dans `default_config.yaml`

### 🐛 Corrigé
- **Double-comptage contexte RAG** : Correction du bug causant l'augmentation de taille lors de la compression
- **Calcul taille prompt** : Méthode `_calculate_final_prompt_size()` pour mesures précises

### 📋 Technique
- Modules modifiés : `agents/base_agent.py`, `utils/logger.py`, `config/default_config.yaml`, `supervisor.py`
- Nouveau système : Compression via `lightweight_llm_service.summarize_context()`
- Architecture : Intégration complète avec le système RAG existant

## [1.1.0] - 2025-08-15

### ✨ Ajouté
- **Boucle de Gouvernance Renforcée** : Transformation du superviseur en garant actif de conformité projet
  - **Project Charter automatique** : Génération et préservation d'une charte projet structurée lors de la planification
  - **Vérification intelligente des jalons** : Évaluation conditionnelle (rapide/approfondie) basée sur les auto-évaluations des agents
  - **Auto-correction dynamique** : Ajout automatique de jalons correctifs en cas de non-conformité détectée
  - **Rapports structurés** : Génération automatique de rapports d'auto-évaluation par les agents (compliant/partial/failed)

### 🔧 Corrigé
- **Format de réponse LLM** : Gestion robuste des réponses structurées du modèle `magistral-medium-latest` (format liste avec thinking/text)
- **Stabilité RAG** : Protection contre les erreurs de type lors de l'indexation de contenu non-chaîne

### 🏗️ Amélioré
- **Préservation contextuelle** : Le Project Charter est automatiquement marqué `preserve: True` dans le RAG pour éviter la compression
- **Journalisation enrichie** : Traçabilité complète des décisions de vérification et actions correctives
- **Performance optimisée** : Vérification rapide pour les jalons conformes, évaluation approfondie uniquement si nécessaire

### 📋 Technique
- Nouveaux fichiers : `test_governance_implementation.py` (validation complète)
- Modules modifiés : `agents/supervisor.py`, `agents/base_agent.py`
- Tests : 4/5 tests de validation passants
- Compatibilité : Aucune modification breaking, utilise les outils existants

## [1.0.0] - 2025-08-14

### 🎉 Ajouté
- **Documentation professionnelle** : README.md complet avec badges et structure claire
- **Changelog structuré** : Suivi des versions selon les standards open source
- **Interface CLI améliorée** : Sélection interactive avec 5 templates prédéfinis via Rich
- **Système de rate limiting global** : Protection contre les surcharges API

### 🔧 Amélioré
- **Architecture RAG optimisée** : Injection automatique de contexte intelligent
- **Logs structurés JSON** : Format JSON Lines pour analyse et debug LLM
- **Communication inter-agents** : Limite d'échanges configurable par tâche
- **Parser JSON renforcé** : Stratégies multiples de récupération de données

### 🐛 Corrigé
- **Singleton LLM Factory** : Réutilisation des instances pour économiser les ressources
- **Threading sécurisé** : Verrous pour éviter les race conditions
- **Mémoire de travail RAG** : Évite les fuites mémoire lors des recherches

### 🎉 Ajouté (versions précédentes consolidées)
- **Plateforme multi-agents complète** : Supervisor, Analyst, Developer
- **Moteur RAG avec FAISS** : Recherche vectorielle pour contexte intelligent
- **Support multi-LLM** : Mistral AI et DeepSeek avec factory pattern
- **Système de checkpoints** : Sauvegarde et restauration d'état des projets
- **Configuration YAML flexible** : Personnalisation complète des agents et modèles
- **Communication inter-agents** : Messages structurés entre agents
- **Mémoire de travail partagée** : Découvertes indexées dans le RAG
- **Guidelines dynamiques** : Configuration des comportements par agent
- **Métriques de performance** : Suivi détaillé des opérations
- **Agent Developer** : Spécialisé en génération de code fonctionnel
- **Système d'outils avancé** : `implement_code`, `create_tests`, `create_project_file`
- **Templates de code** : Support Python, JavaScript, Java avec frameworks
- **Validation de code** : Vérification syntaxique et complétude
- **Agent Analyst** : Analyse des besoins et spécifications techniques
- **Extraction de mots-clés** : Service LLM léger pour recherche RAG
- **Configuration multi-environnement** : Support conda et virtualenv
- **Résumé intelligent** : Contexte RAG condensé automatiquement
- **Architecture Agent-Supervisor** : Coordination centralisée des tâches
- **LLM Factory Pattern** : Support multi-provider avec cache
- **Logs JSON structurés** : Traçabilité complète des opérations
- **Connecteur Mistral AI** : Intégration API première version
- **Structure de projet** : Organisation modulaire de base
- **Génération de tests** : Framework automatique pour projets générés

### 🔧 Amélioré (consolidation)
- **Architecture orientée outils** : Agents basés sur des outils atomiques
- **Logging professionnel** : Système centralisé avec rotation automatique
- **Structure modulaire** : Séparation claire des responsabilités
- **Génération de code** : Instructions strictes contre les placeholders
- **Gestion des tâches** : Attribution automatique selon les compétences
- **Parser JSON robuste** : Récupération intelligente des outils
- **Qualité du code généré** : Zéro placeholder, code prêt à l'exécution
- **Tests automatiques** : Génération de tests unitaires réels
- **Documentation intégrée** : Docstrings et commentaires techniques
- **Performance RAG** : Cache intelligent et limitation de contexte
- **Modularité** : Séparation des services core et agents
- **Configuration** : YAML structuré avec validation
- **Configuration centralisée** : Fichier YAML unique
- **Gestion d'erreurs** : Try-catch généralisé avec logs détaillés

---

## 🏷️ Types de changements

- **🎉 Ajouté** : Nouvelles fonctionnalités
- **🔧 Amélioré** : Modifications de fonctionnalités existantes  
- **🐛 Corrigé** : Corrections de bugs
- **📚 Documentation** : Changements de documentation uniquement
- **🔒 Sécurité** : Corrections de vulnérabilités
- **⚠️ Déprécié** : Fonctionnalités qui seront supprimées
- **🗑️ Supprimé** : Fonctionnalités supprimées

## 🤝 Comment contribuer

1. Consultez le [README.md](README.md) pour les instructions de contribution
2. Respectez le format [Keep a Changelog](https://keepachangelog.com/)
3. Ajoutez vos changements dans la section `[Unreleased]` 
4. Utilisez les emojis pour catégoriser vos modifications
5. Décrivez clairement l'impact utilisateur de chaque changement

## 📋 Template pour nouvelles versions

```markdown
## [X.Y.Z] - YYYY-MM-DD

### 🎉 Ajouté
- Nouvelle fonctionnalité A
- Nouvelle fonctionnalité B

### 🔧 Amélioré  
- Amélioration de X
- Optimisation de Y

### 🐛 Corrigé
- Correction du bug Z
- Fix de la régression W

### 📚 Documentation
- Mise à jour README
- Nouveaux exemples
```