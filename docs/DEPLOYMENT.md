# Deployment Guide - LLM Oil & Gas Assistant

## Docker Deployment

### Build the Image

```bash
cd llm-oil-gas-assistant
docker build -t llm-oil-gas-assistant .
```

### Run the Container

```bash
docker run -p 5015:5015 llm-oil-gas-assistant
```

### With Model Training

```bash
docker run -p 5015:5015 llm-oil-gas-assistant bash -c "python train.py && python app.py"
```

## Docker Compose

```yaml
version: '3.8'
services:
  llm-assistant:
    build: .
    ports:
      - "5015:5015"
    volumes:
      - ./outputs:/app/outputs
    environment:
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PYTHONUNBUFFERED | Disable Python output buffering | 1 |
| PORT | Server port (hardcoded in app.py) | 5015 |

## Manual Deployment

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

### Run with Gunicorn (Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5015 app:app
```

### Run with Flask Development Server

```bash
python app.py
```

## Production Considerations

- Use **gunicorn** with multiple workers for production deployments
- Set `debug=False` in `app.py` (change `debug=True` to `debug=False`)
- Configure proper logging for request/error tracking
- Place behind a reverse proxy (nginx/Apache) for SSL termination
- Monitor memory usage as TF-IDF matrices can be large with many documents
- Pre-train models before starting the server for faster first request

## Health Check

```bash
curl http://localhost:5015/api/health
```

Expected response:
```json
{"status": "healthy", "rag_ready": true, "models_loaded": [...], "version": "1.0.0"}
```

## Ports

| Service | Port |
|---------|------|
| LLM Oil & Gas Assistant | 5015 |
