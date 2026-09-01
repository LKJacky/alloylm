FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install development tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    cmake \
    curl \
    g++ \
    gcc \
    git \
    htop \
    lsof \
    make \
    nano \
    pre-commit \
    procps \
    python3-venv \
    rsync \
    shfmt \
    sudo \
    tar \
    tmux \
    tree \
    unzip \
    vim \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Create the Python environment.
RUN python3 -m venv --system-site-packages /workspaces/.venv \
    && /workspaces/.venv/bin/python -m pip install --no-cache-dir --upgrade pip

# Install Node.js and AI tools.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install --global @github/copilot @anthropic-ai/claude-code @openai/codex \
    && rm -rf /var/lib/apt/lists/*
RUN /workspaces/.venv/bin/pip install --no-cache-dir \
    "hermes-node @ git+https://github.com/LKJacky/hermes_nodes.git#subdirectory=node"
# Install Python dependencies.
COPY requirements.txt /root/requirements.txt
RUN /workspaces/.venv/bin/pip install --no-cache-dir -r /root/requirements.txt
