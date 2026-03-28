FROM python:3.12-slim
WORKDIR /benchmark

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# wrapper.py provides backwards compatibility for old Python-based evaluators
# that define evaluate(program_path) -> dict.  Bridges them to the container
# JSON protocol.  Source of truth: skydiscover/evaluation/wrapper.py
COPY . .
RUN chmod +x evaluate.sh

ENTRYPOINT ["./evaluate.sh"]
