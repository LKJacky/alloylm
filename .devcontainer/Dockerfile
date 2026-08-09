# syntax=docker.1ms.run/docker/dockerfile:1.7

FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 

# change mirror
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates
RUN  rm -f /etc/apt/sources.list.d/ubuntu.sources
ARG TARGETARCH
COPY .devcontainer/sources.list /tmp/sources.list
COPY .devcontainer/sources.list_x64 /tmp/sources.list_x64
RUN if [ "$TARGETARCH" = "arm64" ] || [ "$TARGETARCH" = "arm" ]; then \
    cp /tmp/sources.list /etc/apt/sources.list; \
    else \
    cp /tmp/sources.list_x64 /etc/apt/sources.list; \
    fi \
    && rm -f /tmp/sources.list /tmp/sources.list_x64 \
    && apt-get update

# Install development tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    bash  curl git tar unzip zip \
    sudo procps rsync tree vim tmux

# Install Node.js and AI tools
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @github/copilot @anthropic-ai/claude-code @openai/codex


# Update and install essential dev packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked  \
    apt-get update && apt-get install -y \
    # Shell & basic utils
    bash \
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    tree \
    unzip \
    zip \
    tar \
    # Build tools
    build-essential \
    cmake \
    make \
    gcc \
    g++ \
    # Python
    python3-venv \
    # Process & system
    lsof \
    procps \
    rsync \
    tmux \
    shfmt \
    pre-commit

# create python environment
RUN python3 -m venv --system-site-packages /workspaces/.venv

# build flash-attn
ARG FLASH_ATTN_CUDA_ARCHS
RUN if [ -n "${FLASH_ATTN_CUDA_ARCHS}" ]; then \
        export FLASH_ATTN_CUDA_ARCHS; \
        echo "Building flash-attn for CUDA architectures: ${FLASH_ATTN_CUDA_ARCHS}"; \
    else \
        echo "FLASH_ATTN_CUDA_ARCHS is not set. Using default architectures."; \
        unset FLASH_ATTN_CUDA_ARCHS; \
    fi \
    && FLASH_ATTENTION_FORCE_BUILD="TRUE" \
    FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" \
    MAX_JOBS=16 /workspaces/.venv/bin/pip install flash-attn --no-build-isolation

# Install Python dependencies
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
COPY requirements.txt /root/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /workspaces/.venv/bin/pip install -r /root/requirements.txt


RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked  \
    apt-get update && apt-get install -y gnome-keyring libsecret-1-0 dbus-x11

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*