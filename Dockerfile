FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY artifacts/model.pkl ./artifacts/model.pkl
EXPOSE 8080
ENV MODEL_PATH=/app/artifacts/model.pkl
CMD ["python", "-m", "src.main"]
