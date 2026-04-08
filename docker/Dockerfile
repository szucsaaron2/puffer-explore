FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1)
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# PufferLib (stable release from PyPI)
RUN pip3 install "pufferlib[cleanrl,atari]"

# Test deps
RUN pip3 install pytest numpy gymnasium

# Mount point for puffer-explore
WORKDIR /workspace

# Copy and install puffer-explore
COPY . /workspace/puffer-explore
RUN cd /workspace/puffer-explore && pip3 install -e .

CMD ["bash"]
