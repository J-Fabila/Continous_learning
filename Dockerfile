# Imagen base ligera
FROM python:3.11-slim

# Evita archivos .pyc y activa stdout sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para scipy/pandas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (mejor cache)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Comando por defecto
ENTRYPOINT ["python", "drift.py"]
