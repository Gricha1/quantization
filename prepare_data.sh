#!/bin/bash
# Script to prepare GSM8K dataset for training
# Run this inside the Docker container

set -e  # Exit on error

echo "Preparing GSM8K dataset..."

# Create data directory
mkdir -p ~/data/gsm8k

# Check if data preprocessing script exists
VERL_DIR="/usr/home/workspace/verl"
if [ ! -f "$VERL_DIR/examples/data_preprocess/gsm8k.py" ]; then
    echo "❌ Error: Data preprocessing script not found at $VERL_DIR/examples/data_preprocess/gsm8k.py"
    echo "Make sure verl repository is cloned in /usr/home/workspace/verl"
    exit 1
fi

echo "Running data preprocessing..."
cd "$VERL_DIR"
python examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

# Verify data was created
if [ -f "$HOME/data/gsm8k/train.parquet" ] && [ -f "$HOME/data/gsm8k/test.parquet" ]; then
    echo "✅ Data preparation complete!"
    echo "Data location: ~/data/gsm8k"
    ls -lh ~/data/gsm8k/
else
    echo "❌ Error: Data files were not created"
    echo "Expected files:"
    echo "  - $HOME/data/gsm8k/train.parquet"
    echo "  - $HOME/data/gsm8k/test.parquet"
    exit 1
fi
