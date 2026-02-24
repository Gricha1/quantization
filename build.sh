#!/bin/bash
# Build script for Flash-RL Docker image
# Builds from SafeLLM directory context to access setup.py and other files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_PATH="$SCRIPT_DIR/Dockerfile.A100"
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/gorbov_gv/safe_rl_nlp/SafeLLM}"

if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "ERROR: Workspace directory $WORKSPACE_DIR does not exist"
    echo "Please set WORKSPACE_DIR environment variable or ensure SafeLLM is at the default path"
    exit 1
fi

echo "Building Docker image from: $WORKSPACE_DIR"
echo "Using Dockerfile: $DOCKERFILE_PATH"

cd "$WORKSPACE_DIR"
docker build -f "$DOCKERFILE_PATH" -t flash_rl_quantization_img .
