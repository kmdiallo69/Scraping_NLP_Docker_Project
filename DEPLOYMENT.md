# 🚀 Deployment Guide

This guide covers different deployment options for ToxiGuard French Tweets.

## 📋 Prerequisites

- Python 3.8+
- Docker (optional)
- Git
- Chrome browser (for scraping)

## 🏠 Local Development

### Quick Start

1. **Clone and setup**
   ```bash
   git clone https://github.com/yourusername/ToxiGuard-French-Tweets.git
   cd ToxiGuard-French-Tweets
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Activate environment**
   ```bash
   source venv/bin/activate
   ```

3. **Collect data and train model**
   ```bash
   python scraping.py
   ```

4. **Start API**
   ```bash
   cd fastapi
   python main.py
   ```

### Manual Setup

1. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create directories**
   ```bash
   mkdir -p data fastapi/model
   ```

## 🐳 Docker Deployment

### Single Container

```bash
cd fastapi
docker build -t toxiguard-api .
docker run -d -p 8000:80 --name toxiguard toxiguard-api
```

### Docker Compose

```bash
# Start API service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Development with Docker

```bash
# Start development environment
docker-compose --profile development up -d

# Access development container
docker exec -it toxiguard-dev bash
```

## ☁️ Cloud Deployment

### Heroku

1. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Set buildpacks**
   ```bash
   heroku buildpacks:set heroku/python
   heroku buildpacks:add https://github.com/heroku/heroku-buildpack-google-chrome
   heroku buildpacks:add https://github.com/heroku/heroku-buildpack-chromedriver
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

### DigitalOcean App Platform

1. Create `app.yaml`:
   ```yaml
   name: toxiguard-api
   services:
   - name: api
     source_dir: /fastapi
     dockerfile_path: fastapi/Dockerfile
     routes:
     - path: /
     http_port: 80
     run_command: uvicorn main:app --host 0.0.0.0 --port 80
   ```

### AWS ECS

1. **Build and push to ECR**
   ```bash
   aws ecr create-repository --repository-name toxiguard-api
   docker build -t toxiguard-api ./fastapi
   docker tag toxiguard-api:latest <account>.dkr.ecr.region.amazonaws.com/toxiguard-api:latest
   docker push <account>.dkr.ecr.region.amazonaws.com/toxiguard-api:latest
   ```

2. **Create ECS task definition**
   ```json
   {
     "family": "toxiguard-api",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "256",
     "memory": "512",
     "containerDefinitions": [
       {
         "name": "api",
         "image": "<account>.dkr.ecr.region.amazonaws.com/toxiguard-api:latest",
         "portMappings": [
           {
             "containerPort": 80,
             "protocol": "tcp"
           }
         ]
       }
     ]
   }
   ```

## 🔧 Production Configuration

### Environment Variables

```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=80
export MODEL_PATH=./model/

# Logging
export LOG_LEVEL=INFO
export LOG_FORMAT=json

# Security
export CORS_ORIGINS=["https://yourdomain.com"]
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoring

### Health Checks

The API includes health check endpoints:
- `GET /`: Basic health check
- `GET /health`: Detailed health status

### Logging

```python
# Add to main.py for structured logging
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
```

### Metrics (Optional)

Add Prometheus metrics:
```bash
pip install prometheus-fastapi-instrumentator
```

```python
# In main.py
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
```

## 🔒 Security Considerations

### API Security

1. **Rate Limiting**
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
   
   @app.get("/predict/{message}")
   @limiter.limit("10/minute")
   async def predict(request: Request, message: str):
       # ... existing code
   ```

2. **CORS Configuration**
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],
       allow_credentials=True,
       allow_methods=["GET"],
       allow_headers=["*"],
   )
   ```

### Infrastructure Security

- Use HTTPS in production
- Implement API key authentication
- Regular security updates
- Monitor for suspicious activity
- Use secrets management (AWS Secrets Manager, etc.)

## 🚨 Troubleshooting

### Common Issues

1. **Model not found**
   ```bash
   # Ensure model files exist
   ls -la fastapi/model/
   
   # Retrain if needed
   python scraping.py
   ```

2. **Chrome driver issues**
   ```bash
   # Update webdriver-manager
   pip install --upgrade webdriver-manager
   ```

3. **Memory issues**
   - Increase Docker memory limits
   - Use smaller model variants
   - Implement model caching

### Debug Mode

```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
python fastapi/main.py
```

## 📈 Scaling

### Horizontal Scaling

1. **Load Balancer Configuration**
2. **Multiple API instances**
3. **Shared model storage (S3, GCS)**
4. **Redis for caching**

### Performance Optimization

1. **Model quantization**
2. **Response caching**
3. **Async processing**
4. **Connection pooling**

---

For more detailed information, see:
- [README.md](README.md) - Project overview
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
- [examples/](examples/) - Usage examples 