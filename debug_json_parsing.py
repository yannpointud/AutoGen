#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.json_parser import get_json_parser

# Réponse LLM exacte du superviseur
llm_response = '''```json
{
    "decision": "request_rework",
    "success_rate": 0,
    "reason": "
    1. **Taux de réussite calculé**:
       - Agent impliqué: 1 (analyst)
       - Artefacts produits: 0/3 (Document de spécifications techniques, Liste des cas d'erreur, Maquette CLI)
       - Auto-évaluation: Absence de rapport => **échec implicite** (0% de réussite par agent).
       - Formule: (0 artefacts / 3 livrables attendus) × 100 = **0%**.
       - Seuil minimal de 90% non atteint.

    2. **Analyse critique**:
       - **Rôle unique et critique**: L'agent *analyst* était le seul responsable de ce jalon fondateur. Son échec (0 artefacts) signifie:
         - Aucune spécification technique formalisée (blocage pour les jalons suivants).
         - Aucune gestion d'erreur définie (risque de bugs en cascade).
         - Aucune maquette CLI (interface utilisateur non conçue).
       - **Impact**: Sans ces livrables, les jalons ultérieurs (développement, tests) **ne peuvent pas démarrer**, compromettant l'ensemble du projet.
       - **Contexte Project Charter**: Les spécifications initiales ('opérations de base', 'CLI interactive') sont **trop vagues** pour être implémentées directement. Ce jalon devait les préciser (ex: format des entrées, gestion des erreurs).

    3. **Recommandation stratégique**:
       - **Priorité absolue**: Re-traiter ce jalon avec:
         - Un document de spécifications **détaillé** (ex: syntaxe des commandes CLI, messages d'erreur standardisés).
         - Une liste **exhaustive** des cas d'erreur (ex: division par zéro, entrées non numériques).
         - Une maquette CLI **validée** (ex: flux de dialogue type).
       - **Risque si ignoré**: Développement basé sur des hypothèses → coûts de correction exponentiels plus tard.
    ",
    "confidence": 1.0,
    "agents_analysis": {
        "analyst": "
        - **status**: failure
        - **justification**:
          - **Livrables manquants**: 0/3 artefacts produits (document technique, cas d'erreur, maquette CLI).
          - **Blocage critique**: Sans ces éléments, le projet ne peut avancer conformément au *Project Charter* (ex: 'Interface CLI interactive' non spécifiée).
          - **Responsabilité totale**: Agent unique sur ce jalon → échec = échec du jalon.
          - **Action requise**: Réassigner la tâche avec:
            - **Échéance claire** pour les 3 livrables.
            - **Exemples concrets** à inclure (ex: '2 + 3' doit afficher '5', 'a + 1' doit afficher 'Erreur: entrée invalide').
            - **Validation intermédiaire** pour éviter un nouvel échec.
        "
    }
}
```'''

print("=== TEST PARSING JSON CENTRALISÉ ===")
parser = get_json_parser("Calculator.Debug")

print("\n1. Test avec parse_llm_response (ancien):")
result_old = parser.parse_llm_response(llm_response)
print(f"Résultat: {result_old}")
print(f"Keys: {list(result_old.keys()) if result_old else 'None'}")

print("\n2. Test avec parse_universal (nouveau):")
result_new = parser.parse_universal(llm_response, return_type='dict')
print(f"Résultat: {result_new}")
print(f"Keys: {list(result_new.keys()) if result_new else 'None'}")

if result_new:
    print(f"\n✅ SUCCÈS! Decision: {result_new.get('decision', 'MANQUANT')}")
    print(f"Success_rate: {result_new.get('success_rate', 'MANQUANT')}")
    print(f"Confidence: {result_new.get('confidence', 'MANQUANT')}")
    print(f"Reason preview: {str(result_new.get('reason', 'MANQUANT'))[:100]}...")
else:
    print("\n❌ ÉCHEC: Aucune stratégie n'a pu parser le JSON")