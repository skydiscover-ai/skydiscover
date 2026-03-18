FROM python:3.12-slim

WORKDIR /benchmark
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x evaluate.sh
ENTRYPOINT ["./evaluate.sh"]
