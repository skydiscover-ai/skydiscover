FROM python:3.10-slim
WORKDIR /benchmark

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# wrapper.py provides backwards compatibility for old Python-based evaluators
# that define evaluate(program_path) -> dict.  Bridges them to the container
# JSON protocol.  Source of truth: skydiscover/evaluation/wrapper.py
COPY . .

# Download run_and_check.py from KernelBench repository (pinned to specific commit)
ARG KERNELBENCH_COMMIT=423217d
RUN curl -o run_and_check.py \
    "https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/${KERNELBENCH_COMMIT}/scripts/run_and_check.py"

RUN chmod +x evaluate.sh

ENTRYPOINT ["./evaluate.sh"]