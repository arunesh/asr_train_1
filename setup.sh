#!/bin/bash
# Setup script for ASR Training Pipeline - Phase 1

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║       ASR Training Pipeline - Phase 1 Setup               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.9+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check CUDA availability (optional but recommended)
echo ""
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ CUDA available${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not found - will use CPU (training will be slow)${NC}"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/{raw,processed,manifests}
mkdir -p models/{whisper_hindi,whisper_punjabi}
mkdir -p training/{configs,scripts,logs}
mkdir -p optimization
mkdir -p evaluation/results
mkdir -p mobile/{android,ios}
mkdir -p notebooks

echo -e "${GREEN}✓ Directories created${NC}"

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x data/download_datasets.py
chmod +x data/preprocess.py
chmod +x training/scripts/train_whisper.py
chmod +x evaluation/evaluate.py
chmod +x optimization/*.py

echo -e "${GREEN}✓ Scripts are executable${NC}"

# Check Weights & Biases (optional)
echo ""
echo "Checking Weights & Biases setup..."
if command -v wandb &> /dev/null; then
    if wandb status &> /dev/null; then
        echo -e "${GREEN}✓ Weights & Biases configured${NC}"
    else
        echo -e "${YELLOW}⚠ Weights & Biases not logged in${NC}"
        echo "  Run: wandb login"
        echo "  Or set: export WANDB_API_KEY=your_key"
    fi
else
    echo -e "${YELLOW}⚠ wandb command not found${NC}"
fi

# Test imports
echo ""
echo "Testing critical imports..."
python3 -c "
import torch
import transformers
import librosa
import numpy
import pandas
print('✓ All critical packages import successfully')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Package imports successful${NC}"
else
    echo -e "${RED}✗ Import errors detected${NC}"
    exit 1
fi

# Print next steps
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "2. Download datasets:"
echo "   python data/download_datasets.py --language hindi"
echo "   python data/download_datasets.py --language punjabi"
echo ""
echo "3. Manually download Common Voice datasets:"
echo "   Visit: https://commonvoice.mozilla.org/datasets"
echo ""
echo "4. Preprocess data:"
echo "   python data/preprocess.py --language hindi"
echo "   python data/preprocess.py --language punjabi"
echo ""
echo "5. Start training:"
echo "   python training/scripts/train_whisper.py --config training/configs/whisper_tiny_hindi.yaml"
echo ""
echo "For GPU training, ensure CUDA is available!"
echo ""
