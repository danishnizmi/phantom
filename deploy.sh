#!/bin/bash

# Configuration
PROJECT_ID=$(gcloud config get-value project)
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
gcloud run jobs update $JOB_NAME \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME \
    --region $REGION \
    --set-env-vars PROJECT_ID=$PROJECT_ID,REGION=$REGION,BUDGET_MODE=False \
    --max-retries 0 \
    --task-timeout 10m

# 5. Create/Update Cloud Scheduler (Runs daily at 9am)
# Note: Requires a service account with permission to invoke the job. 
# We'll assume the default compute service account has permissions or user will configure IAM.
gcloud scheduler jobs create http tech-influencer-schedule \
    --schedule "0 9 * * *" \
    --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/$JOB_NAME:run" \
    --http-method POST \
    --oauth-service-account-email $(gcloud config get-value account) \
    --location $REGION \
    || gcloud scheduler jobs update http tech-influencer-schedule \
    --schedule "0 9 * * *" \
    --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/$JOB_NAME:run" \
    --http-method POST \
    --oauth-service-account-email $(gcloud config get-value account) \
    --location $REGION

echo "Deployment Complete! You can manually trigger the job with:"
echo "gcloud run jobs execute $JOB_NAME --region $REGION"
