#!/bin/bash
# WSL2 GPU Environment Setup Script for RAPIDS cuDF
# This script sets up the complete environment for GPU-accelerated backtesting
# Requirements: WSL2, NVIDIA GPU (RTX 3060Ti or better), Windows NVIDIA drivers

set -e  # Exit on error

echo "=========================================="
echo "WSL2 GPU Environment Setup for RAPIDS cuDF"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in WSL2
print_info "Checking WSL2 environment..."
if ! grep -qi microsoft /proc/version; then
    print_error "This script must be run in WSL2"
    exit 1
fi
print_info "WSL2 detected ✓"

# Check GPU availability
print_info "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    print_info "GPU detected ✓"
else
    print_warning "nvidia-smi not found. GPU drivers may not be installed."
    print_info "Please ensure NVIDIA drivers are installed on Windows host."
fi

# Update system packages
print_info "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install required system packages
print_info "Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    vim

# Check if Miniconda is installed
if ! command -v conda &> /dev/null; then
    print_info "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    
    # Initialize conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    
    print_info "Miniconda installed ✓"
    print_warning "Please restart your terminal and run this script again."
    exit 0
else
    print_info "Conda already installed ✓"
fi

# Initialize conda for this session
eval "$(conda shell.bash hook)"

# Check if freqtrade-gpu environment exists
if conda env list | grep -q "freqtrade-gpu"; then
    print_warning "Environment 'freqtrade-gpu' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing environment..."
        conda env remove -n freqtrade-gpu -y
    else
        print_info "Using existing environment."
        conda activate freqtrade-gpu
        print_info "Environment activated ✓"
        exit 0
    fi
fi

# Create conda environment with Python 3.12
print_info "Creating conda environment 'freqtrade-gpu' with Python 3.12..."
conda create -n freqtrade-gpu python=3.12 -y

# Activate environment
print_info "Activating environment..."
conda activate freqtrade-gpu

# Install RAPIDS cuDF
print_info "Installing RAPIDS cuDF (this may take several minutes)..."
print_info "Using CUDA 12.5 compatible version..."

# Install RAPIDS using conda
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=24.12 \
    python=3.12 \
    cuda-version=12.5 \
    -y

if [ $? -eq 0 ]; then
    print_info "RAPIDS cuDF installed successfully ✓"
else
    print_error "Failed to install RAPIDS cuDF"
    exit 1
fi

# Verify cuDF installation
print_info "Verifying cuDF installation..."
python -c "import cudf; print(f'cuDF version: {cudf.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    print_info "cuDF verification successful ✓"
else
    print_error "cuDF verification failed"
    exit 1
fi

# Install Freqtrade
print_info "Installing Freqtrade 2025.12..."
pip install freqtrade==2025.12

if [ $? -eq 0 ]; then
    print_info "Freqtrade installed successfully ✓"
else
    print_error "Failed to install Freqtrade"
    exit 1
fi

# Install custom dependencies
print_info "Installing custom dependencies..."
if [ -f "requirements-custom.txt" ]; then
    pip install -r requirements-custom.txt
    print_info "Custom dependencies installed ✓"
else
    print_warning "requirements-custom.txt not found, skipping..."
fi

# Verify installations
print_info ""
print_info "=========================================="
print_info "Installation Summary"
print_info "=========================================="

echo ""
echo "Python version:"
python --version

echo ""
echo "Freqtrade version:"
freqtrade --version | head -n 5

echo ""
echo "cuDF version:"
python -c "import cudf; print(f'cuDF: {cudf.__version__}')"

echo ""
echo "Pandas version:"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

print_info ""
print_info "=========================================="
print_info "Setup Complete!"
print_info "=========================================="
print_info ""
print_info "To activate the environment in future sessions:"
print_info "  conda activate freqtrade-gpu"
print_info ""
print_info "To test the DataFrame backend:"
print_info "  python -m parallel_backtest.dataframe_backend"
print_info ""
print_info "To run performance benchmarks:"
print_info "  python tests/benchmark_dataframe_backend.py"
print_info ""
print_info "Happy GPU-accelerated trading! 🚀"
