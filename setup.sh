#!/bin/bash
# Script d'installation complète pour AutoGen v6.5
# Crée l'environnement, installe les dépendances et vérifie l'installation

set -e  # Arrêter en cas d'erreur

echo "🚀 Installation AutoGen - Plateforme Multi-Agents IA v6.5"
echo "=========================================================="

# Fonction d'aide
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --check-only    Vérifier l'installation existante seulement"
    echo "  --recreate      Recréer l'environnement (supprime l'existant)"
    echo "  --help          Afficher cette aide"
    echo ""
    exit 0
}

# Parser les arguments
CHECK_ONLY=false
RECREATE=false

for arg in "$@"; do
    case $arg in
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --recreate)
            RECREATE=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "❌ Argument inconnu: $arg"
            show_help
            ;;
    esac
done

# ========================================
# ÉTAPE 1: Vérifications préliminaires
# ========================================

echo ""
echo "🔍 Vérifications préliminaires..."

# Vérifier conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda n'est pas installé ou pas dans le PATH"
    echo ""
    echo "📖 Pour installer conda:"
    echo "   - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "   - Anaconda: https://www.anaconda.com/products/distribution"
    exit 1
fi

echo "✅ Conda trouvé: $(conda --version)"

# Vérifier que nous ne sommes pas déjà dans l'environnement AutoGen
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}' | head -1)
if [ "$CURRENT_ENV" = "AutoGen" ] && [ "$CHECK_ONLY" = false ]; then
    echo "⚠️  Vous êtes dans l'environnement AutoGen"
    echo "   Désactivez-le d'abord: conda deactivate"
    exit 1
fi

# ========================================
# ÉTAPE 2: Mode vérification seulement
# ========================================

if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo "🔍 Mode vérification - Diagnostic de l'installation..."
    
    # Vérifier environnement AutoGen
    if ! conda env list | grep -q "AutoGen"; then
        echo "❌ Environnement conda 'AutoGen' non trouvé"
        echo "   Relancez sans --check-only pour l'installer"
        exit 1
    fi
    
    echo "✅ Environnement conda 'AutoGen' trouvé"
    
    # Activer l'environnement temporairement pour vérifications
    eval "$(conda shell.bash hook)"
    conda activate AutoGen
    
    # Vérifier Python
    PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(python -c 'import sys; print(sys.version_info >= (3,10))') == "True" ]]; then
        echo "✅ Python $PYTHON_VERSION (compatible)"
    else
        echo "❌ Python $PYTHON_VERSION (3.10+ requis)"
        exit 1
    fi
    
    # Vérifier dépendances critiques
    echo ""
    echo "📦 Vérification des dépendances critiques..."
    
    MISSING_DEPS=()
    
    # Fonction pour vérifier un package Python
    check_python_package() {
        local import_name=$1
        local display_name=$2
        
        if python -c "import $import_name" 2>/dev/null; then
            echo "   ✅ $display_name"
        else
            echo "   ❌ $display_name manquant"
            MISSING_DEPS+=("$display_name")
        fi
    }
    
    check_python_package "mistralai" "mistralai"
    check_python_package "httpx" "httpx"
    check_python_package "faiss" "faiss-cpu"
    check_python_package "numpy" "numpy"
    check_python_package "dotenv" "python-dotenv"
    check_python_package "yaml" "pyyaml"
    check_python_package "json5" "json5"
    check_python_package "pythonjsonlogger" "python-json-logger"
    check_python_package "rich" "rich"
    
    if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
        echo ""
        echo "❌ Dépendances manquantes: ${MISSING_DEPS[*]}"
        echo "   Installez avec: pip install -r requirements.txt"
        exit 1
    fi
    
    # Vérifier structure projet
    echo ""
    echo "📁 Vérification de la structure du projet..."
    
    MISSING_ITEMS=()
    
    # Dossiers requis
    for dir in "agents" "config" "core" "utils"; do
        if [ -d "$dir" ]; then
            echo "   ✅ $dir/"
        else
            echo "   ❌ $dir/ manquant"
            MISSING_ITEMS+=("$dir/")
        fi
    done
    
    # Fichiers requis
    for file in "main.py" "requirements.txt" "conda-requirements.yml" "config/default_config.yaml"; do
        if [ -f "$file" ]; then
            echo "   ✅ $file"
        else
            echo "   ❌ $file manquant"
            MISSING_ITEMS+=("$file")
        fi
    done
    
    if [ ${#MISSING_ITEMS[@]} -ne 0 ]; then
        echo ""
        echo "❌ Structure incomplète: ${MISSING_ITEMS[*]}"
        echo "   Vérifiez que vous êtes dans le bon répertoire"
        exit 1
    fi
    
    # Test d'import du projet
    echo ""
    echo "🧪 Test d'import du projet..."
    
    if python -c "from core.llm_connector import LLMFactory; from agents.base_agent import BaseAgent; print('✅ Imports principaux OK')" 2>/dev/null; then
        echo "✅ Le projet peut être importé sans erreur"
    else
        echo "❌ Erreur lors de l'import du projet"
        exit 1
    fi
    
    # Configuration .env
    if [ -f ".env" ]; then
        echo "✅ Fichier .env présent"
        if grep -q "MISTRAL_API_KEY.*=" .env && ! grep -q "MISTRAL_API_KEY=$" .env; then
            echo "✅ Clé API Mistral configurée"
        else
            echo "⚠️  Clé API Mistral non configurée dans .env"
        fi
    else
        echo "⚠️  Fichier .env manquant (copiez .env.example)"
    fi
    
    conda deactivate
    
    echo ""
    echo "=========================================================="
    echo "✅ Installation validée ! Environnement prêt à utilisé."
    echo ""
    echo "Pour lancer AutoGen:"
    echo "  conda activate AutoGen"
    echo "  python main.py"
    exit 0
fi

# ========================================
# ÉTAPE 3: Installation/Recréation
# ========================================

echo ""
echo "🛠️  Mode installation..."

# Supprimer l'environnement si --recreate
if [ "$RECREATE" = true ]; then
    echo ""
    echo "🗑️  Suppression de l'environnement existant..."
    conda env remove -n AutoGen 2>/dev/null || true
    echo "✅ Environnement supprimé"
fi

# Créer ou utiliser l'environnement existant
if conda env list | grep -q "AutoGen"; then
    echo ""
    echo "🔄 Environnement 'AutoGen' existant trouvé"
else
    echo ""
    echo "🆕 Création de l'environnement conda 'AutoGen'..."
    
    if [ -f "conda-requirements.yml" ]; then
        echo "📋 Utilisation du fichier conda-requirements.yml"
        conda env create -f conda-requirements.yml
    else
        echo "📋 Création manuelle de l'environnement"
        conda create -n AutoGen python=3.10 -y
    fi
    
    echo "✅ Environnement 'AutoGen' créé"
fi

# Activer l'environnement
echo ""
echo "🔌 Activation de l'environnement AutoGen..."
eval "$(conda shell.bash hook)"
conda activate AutoGen

# Vérifier Python
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION activé"

# Mettre à jour pip
echo ""
echo "⬆️  Mise à jour de pip..."
pip install --upgrade pip --quiet

# Installer/Mettre à jour les dépendances
echo ""
echo "📚 Installation des dépendances..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dépendances installées depuis requirements.txt"
else
    echo "❌ Fichier requirements.txt manquant"
    exit 1
fi

# Configuration .env
echo ""
echo "📄 Configuration du fichier .env..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Fichier .env créé depuis .env.example"
        echo "⚠️  N'oubliez pas de configurer vos clés API dans .env"
    else
        echo "⚠️  Ni .env ni .env.example trouvé"
        echo "   Créez un fichier .env avec vos clés API"
    fi
else
    echo "✅ Fichier .env existe déjà"
fi

# Créer les dossiers nécessaires
echo ""
echo "📁 Création des dossiers du projet..."
mkdir -p projects logs
echo "✅ Dossiers projects/ et logs/ créés"

# Désactiver l'environnement pour les tests
conda deactivate

# ========================================
# ÉTAPE 4: Validation finale
# ========================================

echo ""
echo "🧪 Validation de l'installation..."

# Réactiver pour les tests
conda activate AutoGen

# Test d'import rapide
if python -c "
try:
    from core.llm_connector import LLMFactory
    from agents.base_agent import BaseAgent
    from core.rag_engine import RAGEngine
    print('✅ Imports principaux réussis')
except Exception as e:
    print(f'❌ Erreur d\\'import: {e}')
    exit(1)
" 2>/dev/null; then
    echo "✅ Le projet est prêt à être utilisé"
else
    echo "❌ Problème lors du test d'import"
    exit 1
fi

conda deactivate

# ========================================
# ÉTAPE 5: Instructions finales
# ========================================

echo ""
echo "=========================================================="
echo "🎉 Installation terminée avec succès !"
echo ""
echo "📖 Pour utiliser AutoGen:"
echo "   conda activate AutoGen"
echo "   python main.py"
echo ""
echo "🔍 Pour vérifier l'installation:"
echo "   ./setup.sh --check-only"
echo ""
echo "⚙️  N'oubliez pas de configurer vos clés API dans le fichier .env"
echo "=========================================================="