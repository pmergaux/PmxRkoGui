# Dockerfile
FROM python:3.11
RUN pip install PyQt6 numpy pandas tensorflow numba ray[tune] optuna
COPY . /app
WORKDIR /app
CMD ["python", "src/main.py"]
