#!/bin/bash
# Setup script for verl and Flash-RL
# Installs verl from flash-rl branch and Flash-RL

set -e

echo "=========================================="
echo "Setting up verl and Flash-RL"
echo "=========================================="

# Install vllm and flash-attn first
echo "Installing vllm==0.8.5..."
pip install vllm==0.8.5

echo "Installing flash-attn==2.7.4.post1..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

echo "Installing transformers (>=4.50.0 for AutoModelForVision2Seq support)..."
pip install "transformers>=4.50.0"

echo "Installing hydra-core..."
pip install hydra-core

echo "Installing torchdata..."
pip install torchdata

# Install required dependencies for verl (before Flash-RL)
echo "Installing verl dependencies (pandas, tensordict, codetiming, hydra-core, peft, pybind11, pylatexenc, torchdata, wandb)..."
pip install pandas "tensordict<=0.6.2" codetiming hydra-core peft pybind11 pylatexenc torchdata wandb

# Install Flash-RL FIRST (verl flash-rl branch requires it)
echo "Checking Flash-RL installation..."
python -c "import flash_rl; print('✓ Flash-RL is already installed')" || {
    echo "Flash-RL not found. Installing flash-llm-rl..."
    pip install flash-llm-rl
    python -c "import flash_rl; print('✓ Flash-RL installed successfully')" || {
        echo "ERROR: Failed to install Flash-RL"
        exit 1
    }
}

# Install verl from flash-rl branch (AFTER Flash-RL and dependencies)
echo "Checking verl installation..."
python -c "import verl.trainer" 2>/dev/null && echo "✓ verl is already installed" || {
    echo "verl not found. Installing verl..."
    
    # Get the script directory to find verl relative to it
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    VERL_DIR="$SCRIPT_DIR/verl"
    
    if [ ! -d "$VERL_DIR" ]; then
        echo "ERROR: verl directory not found at $VERL_DIR"
        echo "Please ensure verl is cloned in quantization/verl:"
        echo "  cd $SCRIPT_DIR"
        echo "  git clone -b flash-rl https://github.com/yaof20/verl"
        exit 1
    fi
    echo "Installing verl from $VERL_DIR..."
    cd "$VERL_DIR"
    pip uninstall -y verl 2>/dev/null || true  # Uninstall old verl if exists
    pip install --no-deps -e .
    cd "$SCRIPT_DIR"
    echo "Verifying verl installation..."
    python -c "import verl.trainer; import verl; import inspect; print('✓ verl installed successfully from:', inspect.getfile(verl))" || {
        echo "ERROR: Failed to install verl. Please check the output above."
        exit 1
    }
}

# Check vLLM LoRA support (but don't reinstall vLLM to avoid breaking flash-attn)
echo "Checking vLLM LoRA support..."
python -c "from vllm.lora.models import LoRAModel; print('✓ vLLM LoRA support OK')" 2>/dev/null || {
    VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    echo "WARNING: vLLM LoRA module not found. Current vLLM version: $VLLM_VERSION"
    echo "Note: If LoRA is needed, you may need to manually install compatible vLLM version."
    echo "But be careful - reinstalling vLLM may break flash-attn installation."
}

echo "Installing trl (required for PPO critic value head)..."
# trl requires numpy<2.0.0, so install compatible numpy first
pip install "numpy<2.0.0"
pip install "trl<=0.9.6"

echo "Installing datasets..."
pip install datasets

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="