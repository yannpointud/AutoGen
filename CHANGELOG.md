# Changelog

Toutes les modifications notables de ce projet sont documentées dans ce fichier.

## [1.7.0] - 2025-09-01
### 🎉 Ajouté
  - **Dashboard visualisation logs debug** : Ajout d'un fichier HTML capable d'importer les logs de llm_debug dans un format lisible
### 🔧 Amélioré
  - **Centralisation logs debug** : désormais dans dossier projects/***/logs

## [1.6.3] - 2025-08-31
### 🔧 Amélioré
  - **Parser JSON robuste** : Ajout de `dirtyjson` pour réparer JSON malformé des LLMs + nettoyage multilignes pour `json5` + warnings sur réparations imparfaites
  - **LightweightSvc Parser** Ajout également d'un appel lightweightSvc pour demander a llm leger de corriger json en dernier recours
  - **Dépendances** : Ajout `dirtyjson>=1.0.8` dans requirements.txt et conda-requirements.yml

## [1.6.2] - 2025-08-31
### 🐛 Corrigé
  - **Update tests unitaires** : Suite au dernieres modifications, refonte de plusieurs tests unitaires 

## [1.6.1] - 2025-08-30
### 🐛 Corrigé
  - **Bug critique escalation system** : Correction du mapping report_type dans base_agent.py - les agents envoyaient toujours 'completion' au lieu de mapper correctement 'failed' → 'issue', 'partial' → 'progress'
  - **Amélioration prompt auto-évaluation** : Clarification des critères de completion avec seuils explicites (100%, 25%, 75%) pour réduire l'ambiguïté des évaluations
  ### **Important** : les tests unitaires devront etres adaptés pour refleter la nouvelle logique

## [1.6.0] - 2025-08-29
### 🔧 Amélioré
  - **Changement architecture autoevaluation** : Nouveau systeme d'appel LLM lightweight pour autoevaluation a la fin d'un think-act pour report au superviseur
  - **Centralisation parsing** : Fichier `json_parser.py` crée pour centraliser logique parsing

## [1.5.0] - 2025-08-27

### 🔧 Amélioré
  - **Refacto complète gestion des jalons** : Remplacement double gestion ID par IDs séquentiels unifiés avec renumbering automatique
  - **Interface utilisateur superviseur** : 3 choix clairs + analyse LLM intelligente pour ajustements de plan
  - **Timing d'évaluation corrigé** : Évaluations uniquement en fin de jalon au lieu de pendant l'exécution
  - **Navigation des jalons** : Correction avancement automatique après insertion corrections et modifications de plan
  - **Système corrections simplifié** : Compteur global unique remplace les multiples compteurs de corrections
  - ## **Important** : Le systeme devra etre amélioré pour doter les agents de meilleurs outils de vérification

### 🐛 Corrigé
  - **Échec insertion corrections** : Correction erreur "Jalon X non trouvé" causée par références vers jalons supprimés
  - **Race conditions jalons** : Synchronisation compteurs et journalisation pour cohérence des progressions
  - **Interface ambiguë** : "Valider jalon" créait des corrections au lieu d'approuver les jalons

## [1.4.0] - 2025-08-26

### 🐛 Corrigé
  - **Timeout LLM hs** : Correction de mauvaise gestion du parametre de timeout. 
  - **logique défaillante dans base_agent.py** : Correction verifie l'existence de fichier dans self_assessment

### 🔧 Amélioré  
  - **Mapping explicite des livrables** : Implémentation système obligatoire `fulfills_deliverable` pour tous les outils de création. 
  - Résout les faux positifs/négatifs dans la vérification des livrables par correspondance exacte au lieu de devinette de noms.

### 🎉 Ajouté
  - **Human in the loop** : Intégration du protocole d'escalade utilisateur apres echec de rework ou adjust_plan par le `supervisor.py`

## [1.3.2] - 2025-08-25

### 🐛 Corrigé
  - **Update tests unitaires** : Suite au dernieres modifications, refonte de plusieurs tests unitaires 

### 🎉 Ajouté
  - **Automatisation test unitaires** : `runtest.sh`

## [1.3.1] - 2025-08-22

### 🐛 Corrigé
  - **Retry LLM bug** : Correction du systeme de retry
  - **Config LLM mistral small par defaut** : Pour itération rapide et économies de cout de dev
  - **Bonus RAG manquants** : Ajout des pondérations manquantes et suppression des pondérations orphelines

### 🔧 Amélioré  
  - **Amelioration de l'affichage du logging** : Config RICH pour colorisation

## [1.3.0] - 2025-08-22

### 🎉 Ajouté
- **Fonctionnalité adjust_plan du superviseur**
  - Ajout de la capacité au superviseur a modifier les prochains jalons au lieu de simplement en ajouter en cas de probleme
  - Analyse fine des rapports reçus non conformes
  - Meilleure gestion des rapports manuel et automatiques recus
  - Evaluation lors de fin de jalon des rapports
  - ## **Important** : Actuellement le systeme peut boucler sur des echecs répétés si il ne parvient pas a contourner l'erreur par un meilleur plan
  - ## Bug conservé pour etre résolu plus tard par une interaction utilisateur a developper

## [1.2.6] - 2025-08-22

### 🐛 Corrigé
- **Generation Projet Charter inadéquate avec d'autres modeles**
  - Modification du prompt de génération
- **BUG qui empechait l'envoi de rapport au superviseur**
  - Erreur d'indentation dans `base_agent.py` 😅
  - Erreurs de parsing JSON

### 🎉 Ajouté
  - Précision du nombre de caracteres sur chaque REQUEST (DEBUG) et estimation tokens
  - Parametre de limitation du nombre de corrections initié par superviseur pour eviter boucle infinie

## [1.2.5] - 2025-08-19

### 🐛 Corrigé
- **Ratelimiter ne gérait pas la concurrence**
  - Serialisation des appels LLM pour eliminer les erreurs d'acces API concurrents

### 🔧 Amélioré
- **Architecture des tools**
  - Descriptifs des tools du prompt user -> prompt systeme
  - Ventilation du code des tools dans un dossier `tools` pour chaque agent

## [1.2.4] - 2025-08-18

### 🔧 Amélioré
- **Amélioration robustesse supervision des jalons**
  - Envoi automatique des rapports structurés au Supervisor après completion des tâches 
  - Logique de validation plus tolérante : accepte rapports partiels et détection agents terminés 
  - Réduction significative des warnings "Pas de rapports structurés fiables"
- **Parsing intelligent des réponses LLM** 
  - Gestion native des objets avec attribut `text` (format Mistral avancé)
  - Messages de debug au lieu de warnings pour conversions normales
  - Nouvelle méthode `_parse_json_from_llm_response()` pour extraction JSON depuis markdown 
  - Support des formats ````json`, blocs génériques, et JSON intégré dans texte

### 🐛 Corrigé
- **SUPERVISION : Warnings "Pas de rapports structurés fiables"**
  - Cause : Agents généraient des rapports mais ne les envoyaient pas systématiquement au Supervisor
  - Solution : Envoi automatique via `_tool_report_to_supervisor()` + validation améliorée
- **PARSING LLM : Warnings "Type inattendu <class 'str'>"** 
  - Cause : Réponses LLM dans formats objet/liste non gérées intelligemment
  - Solution : Détection format Mistral + extraction attribut `text` + parsing JSON robuste

## [1.2.3] - 2025-08-18

### 🔧 Amélioré
- **Injection Project Charter universelle dans tous les appels LLM**
  - Phase d'Action : `generate_with_context_enriched()` avec `critical_constraints` au lieu de `generate_with_context()`
  - Communication inter-agents : Injection automatique du Project Charter via `_get_project_charter_from_file()`
  - Génération JSON : Injection automatique du Project Charter via `_get_project_charter_from_file()`

### 🐛 Corrigé  
- **BUG CRITIQUE : Erreur appel LLM dans generate_with_context_enriched()**
  - `MistralConnector.generate() missing 1 required positional argument: 'prompt'`
  - Cause : Utilisation de `generate()` avec paramètre `messages` au lieu de `generate_with_messages()`
  - Solution : `base_agent.py:1446` - Remplacement par `llm.generate_with_messages()` + `agent_context`
- **ARCHITECTURE : Project Charter manquant en Phase 2 du cycle cognitif**
  - Phase 1 (Alignment) : ✅ Charter injecté via `generate_with_context_enriched()`
  - Phase 2 (Action) : ❌ Charter absent car utilisation de `generate_with_context()`
  - Solution : Uniformisation avec `generate_with_context_enriched()` partout
  - Résultat : Order prompts LLM conforme - Mission → Project Charter → RAG → Outils

## [1.2.2] - 2025-08-17

### 🔧 Amélioré
- **Injection RAG optimisée avec répartition automatique**
  - Nouveau paramètre `max_document_size: 50000` configurable pour l'indexation RAG (vs 10KB hardcodé)
  - Messages de log détaillés lors de troncature : nom du fichier, tailles avant/après
  - Support de fichiers 5× plus volumineux (ex: `rag_engine.py` indexé complètement)
- **Système d'injection RAG unifié et simplifié**
  -  Suppression paramètre redondant `max_results` (utilisait `top_k` à la place)
  -  Calcul automatique de `chars_per_chunk = max_context_length ÷ top_k` (5000 ÷ 5 = 1000 chars/chunk vs 300 hardcodé)
  -  Utilisation complète de l'espace disponible : 5000 chars vs ~900 chars précédemment
  -  Configuration cohérente : un seul paramètre `top_k` contrôle recherche ET injection
- **Suppression limite arbitraire d'injection RAG**
  - Suppression du check `len(prompt) > 10000` qui bloquait l'injection sur prompts longs
  - MLPricePredictor déblocé : injection RAG maintenant fonctionnelle sur prompts 15-20KB
  - Protection maintenue via `max_context_length`, `top_k`, et timeouts

### 🐛 Corrigé
- **BUG CRITIQUE : Variable utilisée avant définition**
  - `source` utilisée ligne 500 avant définition ligne 504 dans `rag_engine.py:index_document()`
  - Causait des `NameError` lors de troncature de gros documents
  - Solution : déplacement des définitions de métadonnées avant utilisation
- **BUG : Hiérarchie des prompts système vs utilisateur inversée**
  - **Problème** : Le contexte RAG était injecté en `role: "system"` (priorité maximale), écrasant les instructions spécifiques des jalons transmises en `role: "user"`
  - **Cause racine** : Les agents recevaient le Project Charter complet via RAG système au lieu des instructions de jalon ciblées 
    - ✅ **Étape 1** : Déplacement du contexte RAG du prompt système vers le prompt utilisateur avec préfixe "Contexte projet pertinent :"
    - ✅ **Étape 2** : Création de prompts système spécifiques par agent : `"Tu es {AgentName}, {Role}.\nPersonnalité: {Personality}"`
  
### 🧪 Tests
- Tests unitaires pour vérifier troncature RAG et calculs automatiques
- Validation du fonctionnement : 5000 ÷ 5 = 1000 chars/chunk

## [1.2.1] - 2025-08-16

### 🐛 Corrigé
- **BUG CRITIQUE : Perte spécifications techniques dans Project Charter** 
  - **Problème** : La fonction `summarize_constraints()` réduisait le Project Charter à 100 tokens génériques, supprimant toutes les spécifications détaillées (colonnes de données, exemples JSON, stack technique)
  - **Cause racine** : Prompt de génération du Project Charter avec instruction "Sois concis" qui encourageait la simplification
  - **Solution** : 
    - ✅ Remplacement de `summarize_constraints()` par transmission du Project Charter COMPLET aux agents
    - ✅ Nouveau prompt exhaustif avec instructions "PRÉSERVE INTÉGRALEMENT tous les détails techniques..."
    - ✅ Augmentation des limites de taille (+50%) : `max_context_length: 2250`, `max_context_tokens: 3000`
  - **Impact** : Les agents reçoivent maintenant les spécifications complètes au lieu de contraintes génériques, éliminant la génération de code inadéquat

### 🔧 Amélioré  
- **Parsing JSON robuste** : Nouvelle stratégie `_strategy_progressive_parse()` pour gérer les réponses LLM avec code volumineux
- **Optimisation prompts Developer** : Limitation automatique des JSON à 4000 caractères pour éviter les échecs de parsing
- **MetricsVisualizer refondu** : Collecte de vraies métriques depuis les logs actuels avec dashboard HTML autonome
- **Métriques tokens précises** : Calcul tokens d'entrée + sortie (conversion chars→tokens /3) avec affichage détaillé

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