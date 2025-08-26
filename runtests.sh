#!/bin/bash

# Script robuste pour lancer les tests unitaires du projet.
#
# Ce script exécute la suite de tests et affiche un message de succès ou d'échec
# en fonction du résultat. Il propage également le code d'erreur, ce qui est
# essentiel pour l'intégration continue (CI/CD).

# NOTE: 'set -e' est intentionnellement omis ici pour nous permettre de capturer
# le code de sortie de pytest et d'afficher un message personnalisé.

echo " Lancement de la suite de tests complète..."
echo "--------------------------------------------------"

# Construire dynamiquement les options de couverture de test
# pour n'inclure que les dossiers qui existent réellement.
COV_OPTIONS=""
DIRECTORIES_TO_COVER=("core" "agents" "utils" "tools")

for dir in "${DIRECTORIES_TO_COVER[@]}"; do
  if [ -d "$dir" ]; then
    COV_OPTIONS="$COV_OPTIONS --cov=$dir"
  fi
done

# Exécuter pytest et stocker le résultat
pytest \
  -v \
  -x \
  --ignore=projects \
  --ignore=OLD \
  $COV_OPTIONS \
  --cov-report term-missing

# Capturer le code de sortie de la dernière commande (pytest)
PYTEST_EXIT_CODE=$?

echo "--------------------------------------------------"

# Vérifier le code de sortie et afficher le message approprié
if [ $PYTEST_EXIT_CODE -eq 0 ]; then
  echo "✅ Tous les tests sont passés avec succès."
else
  echo "❌ Des tests ont échoué. Veuillez consulter les erreurs ci-dessus."
  # Quitter avec le même code d'erreur pour que les systèmes CI/CD détectent l'échec
  exit $PYTEST_EXIT_CODE
fi