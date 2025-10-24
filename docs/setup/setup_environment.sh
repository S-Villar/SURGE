#!/bin/bash
# SURGE Environment Setup Script
# Quick setup for SURGE ML framework

set -e  # Exit on error

echo "🚀 SURGE Environment Setup"
echo "=========================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Ask user which environment to create
echo "Select environment type:"
echo "  1) Full CPU environment (recommended for most users)"
echo "  2) Minimal environment (lightweight, basic features)"
echo "  3) GPU environment (NVIDIA CUDA support)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        ENV_FILE="environment.yml"
        ENV_NAME="surge"
        echo "📦 Creating full CPU environment..."
        ;;
    2)
        ENV_FILE="environment_minimal.yml"
        ENV_NAME="surge-minimal"
        echo "📦 Creating minimal environment..."
        ;;
    3)
        ENV_FILE="environment_gpu.yml"
        ENV_NAME="surge-gpu"
        echo "📦 Creating GPU environment..."
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "Remove and recreate? [y/N]: " confirm
    if [[ $confirm == [yY] ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ${ENV_NAME}
    else
        echo "❌ Setup cancelled"
        exit 0
    fi
fi

# Create environment
echo "🔨 Creating conda environment from ${ENV_FILE}..."
conda env create -f ${ENV_FILE}

echo ""
echo "✅ Environment created successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the environment:"
echo "      conda activate ${ENV_NAME}"
echo ""
echo "   2. Verify installation:"
echo "      python -c \"import surge; surge._print_availability()\""
echo ""
echo "   3. Run tests:"
echo "      pytest tests/ -v"
echo ""
echo "   4. Try examples:"
echo "      python examples/simple_optuna_demo.py"
echo ""
echo "🎉 Setup complete! Happy modeling!"
