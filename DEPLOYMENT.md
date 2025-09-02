# 🚀 Deployment Guide for Google Cloud Run

## Prerequisites

1. **Google Cloud CLI (gcloud)** installed and configured
2. **Docker** installed and running
3. **Google Cloud Project** with billing enabled
4. **Container Registry API** enabled

## Quick Setup

### 1. Configure Google Cloud
```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2. Update Configuration
Edit `deploy.sh` (Linux/Mac) or `deploy.ps1` (Windows):
- Replace `your-project-id` with your actual Google Cloud Project ID
- Adjust `REGION` if needed (default: `us-central1`)

### 3. Deploy
```bash
# Linux/Mac
chmod +x deploy.sh
./deploy.sh

# Windows PowerShell
.\deploy.ps1
```

## Manual Deployment Steps

### 1. Build Docker Image
```bash
docker build -t gcr.io/YOUR_PROJECT_ID/qbgen-app .
```

### 2. Push to Container Registry
```bash
docker push gcr.io/YOUR_PROJECT_ID/qbgen-app
```

### 3. Deploy to Cloud Run
```bash
gcloud run deploy qbgen-app \
  --image gcr.io/YOUR_PROJECT_ID/qbgen-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 80 \
  --max-instances 10
```

## Architecture

- **Frontend**: Next.js app served on port 8080
- **Backend**: Python Flask API on the same port
- **Static Files**: Served by Next.js
- **API Routes**: Proxied to Flask backend

## Environment Variables

The app will use these default settings:
- `PORT=8080` (Cloud Run requirement)
- `PYTHONUNBUFFERED=1` (for proper logging)

## Scaling & Performance

- **Memory**: 2Gi (adjustable)
- **CPU**: 2 vCPU (adjustable)
- **Concurrency**: 80 requests per instance
- **Max Instances**: 10 (auto-scaling)
- **Timeout**: 300 seconds

## Monitoring

- **Cloud Run Console**: View logs and metrics
- **Cloud Monitoring**: Set up alerts
- **Cloud Logging**: Centralized logging

## Troubleshooting

### Common Issues

1. **Build fails**: Check Dockerfile syntax and dependencies
2. **Image push fails**: Verify gcloud authentication
3. **Deployment fails**: Check Cloud Run quotas and permissions
4. **App doesn't start**: Check container logs in Cloud Run console

### Debug Commands

```bash
# Check container logs
gcloud run logs read --service qbgen-app --region us-central1

# View service details
gcloud run services describe qbgen-app --region us-central1

# Test locally
docker run -p 8080:8080 gcr.io/YOUR_PROJECT_ID/qbgen-app
```

## Cost Optimization

- **Min Instances**: 0 (scale to zero when not in use)
- **Max Instances**: 10 (limit maximum cost)
- **Memory**: Start with 2Gi, adjust based on usage
- **Region**: Choose closest to your users

## Security

- **Public Access**: Currently allows unauthenticated access
- **HTTPS**: Automatically provided by Cloud Run
- **Container Security**: Runs in isolated environment
- **API Keys**: Consider adding authentication for production

## Updates & Rollbacks

### Update Deployment
```bash
# Rebuild and redeploy
./deploy.sh
```

### Rollback
```bash
# List revisions
gcloud run revisions list --service qbgen-app --region us-central1

# Rollback to specific revision
gcloud run services update-traffic qbgen-app \
  --to-revisions=REVISION_NAME=100 \
  --region us-central1
```

## Support

For issues:
1. Check Cloud Run logs
2. Verify Docker build locally
3. Check Google Cloud quotas
4. Review this deployment guide
