FROM python:3.12-slim
WORKDIR /benchmark

COPY run.py .
COPY data/ ./data/
COPY evaluate.sh .
RUN chmod +x evaluate.sh

ENTRYPOINT ["./evaluate.sh"]
