FROM python:3.12-slim
WORKDIR /benchmark

RUN pip install --no-cache-dir scipy

COPY evaluator.py pairs.json evaluate.sh ./
RUN chmod +x evaluate.sh

ENTRYPOINT ["./evaluate.sh"]
