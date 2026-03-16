#!/bin/bash
# Script de configuration d'environnement pour Artimir 2.0 (Agent 1)
# Optimisé pour Ubuntu 25.10 et NVIDIA RTX 5080 (Blackwell)

ENV_NAME="artimir_env"
PYTHON_VERSION="3.10"

echo "=========================================================="
echo "Initialisation de l'environnement Artimir 2.0 (Agent 1)"
echo "Architecture ciblée : NVIDIA RTX Series 50 (Blackwell)"
echo "=========================================================="

# 1. Création de l'environnement Conda
echo "[1/4] Création de l'environnement conda '${ENV_NAME}'..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 2. Installation de PyTorch avec le support CUDA 12.4+ (requis pour Blackwell)
echo "[2/4] Installation de PyTorch (CUDA 12.4)..."
# L'utilisation de cu124 garantit la compatibilité optimale avec les drivers récents
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Installation des dépendances du projet
echo "[3/4] Installation des modules complémentaires..."
pip install opencv-python-headless huggingface_hub numpy

# 4. Configuration finale
echo "[4/4] Configuration terminée."
echo "Pour activer l'environnement, tapez : conda activate ${ENV_NAME}"
echo "Testez l'allocation CUDA avec : python test_cuda.py"
echo "=========================================================="
