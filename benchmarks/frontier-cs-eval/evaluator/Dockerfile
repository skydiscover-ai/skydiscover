FROM python:3.12-slim

# Install Docker CLI (needed for frontier-cs judge)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    docker.io \
    docker-compose-plugin \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /benchmark

# Clone and install Frontier-CS
RUN git clone --depth 1 https://github.com/FrontierCS/Frontier-CS.git /benchmark/Frontier-CS
RUN pip install --no-cache-dir -e /benchmark/Frontier-CS/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x evaluate.sh
ENTRYPOINT ["./evaluate.sh"]
