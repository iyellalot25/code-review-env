FROM python:3.11-slim

WORKDIR /app

# Install server dependencies first (cached layer)
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Install inference dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy full project
COPY . /app

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
