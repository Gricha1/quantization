#!/bin/bash
# Build script for Flash-RL Docker image
# Builds from SafeLLM directory context to access setup.py and other files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_PATH="$SCRIPT_DIR/Dockerfile.A100"

echo "Using Dockerfile: $DOCKERFILE_PATH"

docker build -f "$DOCKERFILE_PATH" -t flash_rl_quantization_img .
