# 🚀 Deploy from GitHub to Google Cloud Run (GUI Method)

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **GitHub repository** with your code
3. **Google Cloud CLI (gcloud)** installed locally (for initial setup)

## Method 1: GitHub Actions + Cloud Run (Recommended)

### Step 1: Set up Google Cloud Service Account

1. **Create Service Account:**
   ```bash
   gcloud iam service-accounts create qbgen-deployer \
     --display-name="qbgen Deployer" \
     --description="Service account for deploying qbgen app"
   ```

2. **Grant necessary permissions:**
   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:qbgen-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/run.admin"
   
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:qbgen-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.admin"
   
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:qbgen-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/iam.serviceAccountUser"
   ```

3. **Create and download key:**
   ```bash
   gcloud iam service-accounts keys create key.json \
     --iam-account=qbgen-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

### Step 2: Configure GitHub Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Add these secrets:
   - `GCP_PROJECT_ID`: Your Google Cloud Project ID
   - `GCP_SA_KEY`: The entire content of the `key.json` file

### Step 3: Push to GitHub

The GitHub Actions workflow will automatically:
- Build your Next.js frontend
- Install Python dependencies
- Build Docker image
- Deploy to Cloud Run

## Method 2: Cloud Run GUI Deployment

### Step 1: Build and Push Docker Image Locally

1. **Build the image:**
   ```bash
   docker build -t gcr.io/YOUR_PROJECT_ID/qbgen-app .
   ```

2. **Push to Container Registry:**
   ```bash
   docker push gcr.io/YOUR_PROJECT_ID/qbgen-app
   ```

### Step 2: Deploy via Cloud Run Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Cloud Run**
3. Click **Create Service**
4. Configure the service:

   **Basic Settings:**
   - **Service name**: `qbgen-app`
   - **Region**: Choose your preferred region
   - **CPU allocation**: CPU is only allocated during request processing
   - **Memory**: 2 GiB
   - **Request timeout**: 300 seconds
   - **Maximum number of requests per container**: 80

   **Container:**
   - **Container image URL**: `gcr.io/YOUR_PROJECT_ID/qbgen-app`
   - **Port**: 8080

   **Advanced Settings:**
   - **Concurrency**: 80
   - **Maximum instances**: 10
   - **Minimum instances**: 0

5. Click **Create**

## Method 3: Using service.yaml with gcloud

1. **Update service.yaml:**
   - Replace `PROJECT_ID` with your actual project ID
   - Adjust region if needed

2. **Deploy:**
   ```bash
   gcloud run services replace service.yaml --region=us-central1
   ```

## Environment Variables

Your app will automatically use:
- `PORT=8080` (Cloud Run requirement)
- `PYTHONUNBUFFERED=1` (for proper logging)

## Monitoring & Updates

### View Logs
- **Cloud Run Console** → Select service → **Logs**
- **Cloud Logging** → Filter by resource type: Cloud Run

### Update Deployment
- **GitHub Actions**: Push to main/master branch
- **Manual**: Rebuild and push Docker image, then update service

### Rollback
- **Cloud Run Console** → **Revisions** → Select previous revision → **Traffic** → Set to 100%

## Troubleshooting

### Common Issues

1. **Build fails in GitHub Actions:**
   - Check Node.js and Python versions
   - Verify dependencies in package.json and requirements.txt

2. **Deployment fails:**
   - Check service account permissions
   - Verify project ID in secrets

3. **App doesn't start:**
   - Check container logs in Cloud Run console
   - Verify PORT environment variable

4. **CORS issues:**
   - Backend already has CORS enabled
   - Check if requests are reaching the correct endpoint

### Debug Commands

```bash
# Check service status
gcloud run services describe qbgen-app --region=us-central1

# View logs
gcloud run logs read --service qbgen-app --region=us-central1

# Test locally
docker run -p 8080:8080 gcr.io/YOUR_PROJECT_ID/qbgen-app
```

## Cost Optimization

- **Scale to zero**: Min instances = 0
- **Max instances**: 10 (prevents runaway costs)
- **Memory**: Start with 2Gi, adjust based on usage
- **Region**: Choose closest to your users

## Security Notes

- **Public access**: Currently allows unauthenticated access
- **HTTPS**: Automatically provided by Cloud Run
- **Container isolation**: Runs in secure environment
- **Consider adding authentication** for production use

## Next Steps

1. **Set up monitoring alerts** in Cloud Monitoring
2. **Configure custom domain** if needed
3. **Set up CI/CD pipeline** with GitHub Actions
4. **Add authentication** for production use
5. **Set up backup and disaster recovery**

Your app will be available at: `https://qbgen-app-YOUR_PROJECT_ID.run.app`
