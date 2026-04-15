FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    docker.io \
    iptables \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

RUN pip3 install --no-cache-dir \
    numpy scipy sympy \
    jax optax \
    scikit-learn numba \
    pandas matplotlib plotly \
    networkx cvxpy autograd pymoo \
    PyWavelets \
 && pip3 install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

# Non-root user: Claude Code refuses --dangerously-skip-permissions as root.
# Added to docker group for DinD mode.
RUN useradd -m -s /bin/bash -G docker claude

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
