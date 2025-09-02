# Deploy to Google Cloud Run (PowerShell)
# Make sure you have gcloud CLI installed and configured

# Configuration
$PROJECT_ID = "your-project-id"  # Replace with your actual project ID
$SERVICE_NAME = "qbgen-app"
$REGION = "us-central1"  # Replace with your preferred region
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

Write-Host "🚀 Starting deployment to Google Cloud Run..." -ForegroundColor Green

# Build the Docker image
Write-Host "📦 Building Docker image..." -ForegroundColor Yellow
docker build -t $IMAGE_NAME .

# Push to Google Container Registry
Write-Host "📤 Pushing image to Google Container Registry..." -ForegroundColor Yellow
docker push $IMAGE_NAME

# Deploy to Cloud Run
Write-Host "🚀 Deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $SERVICE_NAME `
  --image $IMAGE_NAME `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --port 8080 `
  --memory 2Gi `
  --cpu 2 `
  --timeout 300 `
  --concurrency 80 `
  --max-instances 10

Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host "🌐 Your app is available at: https://$SERVICE_NAME-$(gcloud config get-value project).run.app" -ForegroundColor Cyan
