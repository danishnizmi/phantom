#!/bin/bash
set -e # Exit on error

# Configuration
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID not set in gcloud config."
    exit 1
fi

REGION="us-central1"
REPO_NAME="tech-influencer-repo"
IMAGE_NAME="tech-influencer-agent"
JOB_NAME="tech-influencer-job"

echo "Deploying to Project: $PROJECT_ID in Region: $REGION"

# 1. Enable Services (idempotent)
gcloud services enable artifactregistry.googleapis.com run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com aiplatform.googleapis.com firestore.googleapis.com

# 2. Create Artifact Registry Repo if not exists
if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION > /dev/null 2>&1; then
    gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=$REGION --description="Docker repository for Tech Influencer Agent"
fi

# 3. Build and Push Container
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME .

# 4. Create/Update Cloud Run Job
# Check if job exists
if gcloud run jobs describe $JOB_NAME --region $REGION > /dev/null 2>&1; then
    echo "Updating existing job..."
    gcloud run jobs update $JOB_NAME \
        --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME \
        --region $REGION \
        --set-env-vars PROJECT_ID=$PROJECT_ID,REGION=$REGION,BUDGET_MODE=False \
        --max-retries 1 \
        --task-timeout 10m
else
    echo "Creating new job..."
    gcloud run jobs create $JOB_NAME \
        --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME \
        --region $REGION \
        --set-env-vars PROJECT_ID=$PROJECT_ID,REGION=$REGION,BUDGET_MODE=False \
        --max-retries 1 \
        --task-timeout 10m
fi

# 5. Create/Update Cloud Scheduler (Runs daily at 9am)
# Use a specific service account for the scheduler
# For this script, we'll try to find the default compute service account if not manually specified
SERVICE_ACCOUNT_EMAIL=$(gcloud iam service-accounts list --filter="displayName:Compute Engine default service account" --format="value(email)" --limit=1)

if [ -z "$SERVICE_ACCOUNT_EMAIL" ]; then
    SERVICE_ACCOUNT_EMAIL=$(gcloud config get-value account)
    echo "WARNING: Using active account $SERVICE_ACCOUNT_EMAIL. In production, use a service account."
fi

echo "Using Service Account: $SERVICE_ACCOUNT_EMAIL"

# Check if scheduler job exists
if gcloud scheduler jobs describe tech-influencer-schedule --location $REGION > /dev/null 2>&1; then
    gcloud scheduler jobs update http tech-influencer-schedule \
        --schedule "0 9 * * *" \
        --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/$JOB_NAME:run" \
        --http-method POST \
        --oauth-service-account-email $SERVICE_ACCOUNT_EMAIL \
        --location $REGION
else
    gcloud scheduler jobs create http tech-influencer-schedule \
        --schedule "0 9 * * *" \
        --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/$JOB_NAME:run" \
        --http-method POST \
        --oauth-service-account-email $SERVICE_ACCOUNT_EMAIL \
        --location $REGION
fi

echo "Deployment Complete! You can manually trigger the job with:"
echo "gcloud run jobs execute $JOB_NAME --region $REGION"
