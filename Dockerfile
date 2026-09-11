FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Exposition du port de l'API
EXPOSE 8000

# Commande pour lancer l'API FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
