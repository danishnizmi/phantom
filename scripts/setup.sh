#!/bin/bash
set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_ACCOUNT_NAME="tech-influencer-sa"
SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

echo "Setting up resources for Project: $PROJECT_ID"

# 0. Enable Required APIs
echo "Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    aiplatform.googleapis.com \
    firestore.googleapis.com \
    cloudscheduler.googleapis.com

# 1. Create Service Account
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL > /dev/null 2>&1; then
    echo "Creating Service Account: $SERVICE_ACCOUNT_NAME"
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="Tech Influencer Agent Service Account"
else
    echo "Service Account $SERVICE_ACCOUNT_NAME already exists."
fi

# 2. Grant Permissions
echo "Granting permissions..."
# Cloud Run Invoker (for Scheduler)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/run.invoker" \
    --condition=None

# Secret Manager Access (for Agent)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None

# Vertex AI User (for Agent)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/aiplatform.user" \
    --condition=None

# Firestore User (for Agent)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/datastore.user" \
    --condition=None

# 2.5 Create Firestore Database
echo "Creating Firestore Database..."
if ! gcloud firestore databases list --format="value(name)" | grep -q "$PROJECT_ID/databases/(default)"; then
    gcloud firestore databases create --location=$REGION
else
    echo "Firestore database already exists."
fi

# 3. Create Secrets (Placeholders)
SECRETS=("TWITTER_CONSUMER_KEY" "TWITTER_CONSUMER_SECRET" "TWITTER_ACCESS_TOKEN" "TWITTER_ACCESS_TOKEN_SECRET")

for SECRET in "${SECRETS[@]}"; do
    if ! gcloud secrets describe $SECRET > /dev/null 2>&1; then
        echo "Creating secret: $SECRET"
        gcloud secrets create $SECRET --replication-policy="automatic"
        echo "Please add value for $SECRET using: echo -n 'VALUE' | gcloud secrets versions add $SECRET --data-file=-"
    else
        echo "Secret $SECRET already exists."
    fi
done

echo "Setup Complete!"
echo "IMPORTANT: Update your Cloud Run Job to use this service account:"
echo "gcloud run jobs update tech-influencer-job --service-account $SERVICE_ACCOUNT_EMAIL"
