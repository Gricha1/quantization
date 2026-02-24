#!/bin/bash
# Start script for Flash-RL Docker container
# Uses existing safe_llm_img to avoid rebuilding

if [ -z "$1" ]; then
    gpus=0
else
    gpus=$1
fi

if [ -z "$2" ]; then
    container_postfix=
else
    container_postfix=$2
fi

# Use existing safe_llm_img instead of new image
image_name=safe_llm_img
container_name=flash_rl_quantization_$container_postfix

echo "Image name: $image_name"
echo "Container name: $container_name"

# Determine quantization directory
QUANTIZATION_DIR="${QUANTIZATION_DIR:-/home/gorbov_gv/quantization}"

if [ ! -d "$QUANTIZATION_DIR" ]; then
    echo "WARNING: Quantization directory $QUANTIZATION_DIR does not exist"
fi

# Create logdir if it doesn't exist
LOG_DIR="$QUANTIZATION_DIR/logdir"
if [ -d "$LOG_DIR" ]; then
    echo "Log directory exists: $LOG_DIR"
else
    mkdir -p "$LOG_DIR"
    echo "Created log dir: $LOG_DIR"
fi

echo "GPUs in docker: $gpus"
echo "Quantization dir: $QUANTIZATION_DIR"
echo "Mounting quantization to /usr/home/workspace (will override image contents)"

# Mount quantization directory to /usr/home/workspace
# Bind mount will override any existing files in the image at that path
docker run -it --rm --name $container_name --memory="200g" --shm-size=8g --gpus '"device=0,1"' \
  --env COMET_API_KEY=$COMET_API_KEY \
  -v "$QUANTIZATION_DIR":/usr/home/workspace \
  -v "$LOG_DIR":/root/logdir \
  -w /usr/home/workspace \
  $image_name
