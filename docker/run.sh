#!/bin/bash
# Build and run the puffer-explore Docker container with GPU access.
#
# Usage:
#   ./docker/run.sh              # interactive shell
#   ./docker/run.sh probe        # run PufferLib probe script
#   ./docker/run.sh test         # run puffer-explore tests
#   ./docker/run.sh train        # run standalone training demo

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="puffer-explore"

# Build if image doesn't exist
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Building Docker image..."
    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_DIR"
fi

case "${1:-shell}" in
    probe)
        docker run --rm --gpus all \
            -v "$PROJECT_DIR:/workspace/puffer-explore" \
            "$IMAGE_NAME" \
            python3 /workspace/puffer-explore/docker/probe_pufferlib.py
        ;;
    test)
        docker run --rm --gpus all \
            -v "$PROJECT_DIR:/workspace/puffer-explore" \
            "$IMAGE_NAME" \
            bash -c "cd /workspace/puffer-explore && pip3 install -e . -q && pytest tests/ -v"
        ;;
    train)
        docker run --rm --gpus all \
            -v "$PROJECT_DIR:/workspace/puffer-explore" \
            "$IMAGE_NAME" \
            python3 /workspace/puffer-explore/scripts/train_pufferlib.py --explore rnd --total-timesteps 50000
        ;;
    shell|*)
        docker run --rm --gpus all -it \
            -v "$PROJECT_DIR:/workspace/puffer-explore" \
            "$IMAGE_NAME" \
            bash
        ;;
esac
