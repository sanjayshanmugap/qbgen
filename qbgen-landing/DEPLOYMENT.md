# QBGen Hybrid Deployment Guide

This guide explains how to deploy the QBGen application with the new landing page alongside your existing Flask + Vite setup.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐
│   Landing Page  │    │   Flask Backend │
│   (Next.js)     │    │   + Vite App    │
│   Port: 3000    │    │   Port: 8080    │
└─────────────────┘    └─────────────────┘
         │                       │
         └─────── Links to ──────┘
```

## Deployment Options

### Option 1: Separate Containers (Recommended)

1. **Deploy Landing Page**:
   ```bash
   cd qbgen-landing
   docker build -t qbgen-landing .
   docker run -p 3000:3000 qbgen-landing
   ```

2. **Deploy Backend** (your existing setup):
   ```bash
   cd ..
   docker build -t qbgen-backend .
   docker run -p 8080:8080 qbgen-backend
   ```

3. **Configure Reverse Proxy** (nginx/apache):
   - Route `/` to landing page (port 3000)
   - Route `/static/*` and `/api/*` to backend (port 8080)

### Option 2: Docker Compose

```bash
cd qbgen-landing
docker-compose up -d
```

### Option 3: GCP Cloud Build

Update your existing Cloud Build configuration to include the landing page:

```yaml
steps:
  # Build landing page
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/qbgen-landing', './qbgen-landing']
  
  # Build backend (your existing step)
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/qbgen-backend', '.']
  
  # Push images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/qbgen-landing']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/qbgen-backend']
```

## Environment Configuration

### Landing Page Environment Variables
```bash
NODE_ENV=production
NEXT_PUBLIC_API_URL=http://localhost:8080
```

### Backend Environment Variables
```bash
FLASK_ENV=production
PORT=8080
```

## URL Structure

- **Landing Page**: `https://yourdomain.com/`
- **Unique Clues**: `https://yourdomain.com/static/`
- **Set Carding**: `https://yourdomain.com/static/`
- **API Endpoints**: `https://yourdomain.com/api/*`

## Development Setup

1. **Start Landing Page**:
   ```bash
   cd qbgen-landing
   pnpm install
   pnpm dev
   ```

2. **Start Backend** (your existing setup):
   ```bash
   cd backend
   python app.py
   ```

3. **Build Frontend** (your existing setup):
   ```bash
   cd frontend
   npm run build
   ```

## Testing

1. Visit `http://localhost:3000` for landing page
2. Click navigation links to test integration
3. Verify API calls work through the proxy

## Troubleshooting

### Common Issues

1. **CORS Issues**: Ensure your Flask backend allows requests from the landing page domain
2. **Routing Issues**: Check that the reverse proxy is correctly routing requests
3. **Build Issues**: Verify all dependencies are installed correctly

### Debug Commands

```bash
# Check if containers are running
docker ps

# View logs
docker logs <container-name>

# Test API connectivity
curl http://localhost:8080/get_sets
``` 