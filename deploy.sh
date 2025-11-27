#!/bin/bash
set -e # Exit on error

# ============================================================================
# Phantom Tech Influencer - Deployment Script
# ============================================================================
# This script builds and deploys the container image.
#
# For INFRASTRUCTURE management (Cloud Run Job, Schedulers, IAM), use Terraform:
#   cd terraform && terraform apply
#
# This script is for CONTAINER BUILDS only after Terraform has set up the infra.
# ============================================================================

# Configuration - Updated for AWST (Australia/Perth)
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID not set in gcloud config."
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

REGION="${REGION:-us-central1}"
TIMEZONE="Australia/Perth"  # AWST (UTC+8)
CURRENCY="AUD"

# Updated names to match Terraform configuration
REPO_NAME="phantom-influencer"
IMAGE_NAME="phantom-influencer"
JOB_NAME="phantom-influencer-job"

# Check for Terraform mode
USE_TERRAFORM=false
if [ -f "terraform/terraform.tfvars" ] || [ "$1" == "--terraform" ]; then
    USE_TERRAFORM=true
fi

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         Phantom Tech Influencer - Deployment                      ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Project:  $PROJECT_ID"
echo "║  Region:   $REGION"
echo "║  Timezone: $TIMEZONE (AWST)"
echo "║  Currency: $CURRENCY"
if [ "$USE_TERRAFORM" = true ]; then
echo "║  Mode:     Terraform (container build only)"
else
echo "║  Mode:     Legacy (full deployment)"
fi
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# If Terraform is managing infrastructure, just build and push the container
# ============================================================================
if [ "$USE_TERRAFORM" = true ]; then
    echo "Terraform mode: Building and pushing container only..."
    echo "Infrastructure is managed by: cd terraform && terraform apply"
    echo ""

    # Ensure Artifact Registry exists (Terraform should have created it)
    if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION > /dev/null 2>&1; then
        echo "Warning: Artifact Registry '$REPO_NAME' not found."
        echo "Run 'cd terraform && terraform apply' first to create infrastructure."
        exit 1
    fi

    # Build and Push Container
    echo "Building and pushing container image..."
    gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME .

    # Update Cloud Run Job with new image
    if gcloud run jobs describe $JOB_NAME --region $REGION > /dev/null 2>&1; then
        echo "Updating Cloud Run Job with new image..."
        gcloud run jobs update $JOB_NAME \
            --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME \
            --region $REGION
    else
        echo "Warning: Cloud Run Job '$JOB_NAME' not found."
        echo "Run 'cd terraform && terraform apply' first to create infrastructure."
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    Container Build Complete!                      ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  Image: $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME"
    echo "║                                                                   ║"
    echo "║  Test the job:                                                    ║"
    echo "║    gcloud run jobs execute $JOB_NAME --region $REGION             ║"
    echo "║                                                                   ║"
    echo "║  Force a post:                                                    ║"
    echo "║    gcloud run jobs execute $JOB_NAME --region $REGION \\           ║"
    echo "║      --update-env-vars FORCE_POST=true                            ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    exit 0
fi

# ============================================================================
# Legacy Mode: Full deployment without Terraform
# ============================================================================
echo "Legacy mode: Full deployment (consider using Terraform instead)"
echo ""

# 1. Enable Services (idempotent)
echo "Enabling required GCP services..."
gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    aiplatform.googleapis.com \
    firestore.googleapis.com \
    cloudscheduler.googleapis.com

# 2. Create Artifact Registry Repo if not exists
if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION > /dev/null 2>&1; then
    echo "Creating Artifact Registry repository..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="Docker repository for Phantom Tech Influencer"
fi

# 3. Build and Push Container
echo "Building and pushing container image..."
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME .

# 4. Create/Update Cloud Run Job with AWST timezone settings
if gcloud run jobs describe $JOB_NAME --region $REGION > /dev/null 2>&1; then
    echo "Updating existing Cloud Run Job..."
    gcloud run jobs update $JOB_NAME \
        --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME \
        --region $REGION \
        --set-env-vars PROJECT_ID=$PROJECT_ID,REGION=$REGION,TIMEZONE=$TIMEZONE,CURRENCY=$CURRENCY,BUDGET_MODE=False \
        --max-retries 1 \
        --task-timeout 15m \
        --memory 1Gi
else
    echo "Creating new Cloud Run Job..."
    gcloud run jobs create $JOB_NAME \
        --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME \
        --region $REGION \
        --set-env-vars PROJECT_ID=$PROJECT_ID,REGION=$REGION,TIMEZONE=$TIMEZONE,CURRENCY=$CURRENCY,BUDGET_MODE=False \
        --max-retries 1 \
        --task-timeout 15m \
        --memory 1Gi
fi

# 5. Create/Update Cloud Schedulers
# Try phantom service account first, then fallback to default
SERVICE_ACCOUNT_EMAIL="phantom-influencer-sa@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL > /dev/null 2>&1; then
    # Try legacy name
    SERVICE_ACCOUNT_EMAIL="tech-influencer-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL > /dev/null 2>&1; then
        # Fallback to default compute
        COMPUTE_SA=$(gcloud iam service-accounts list --filter="email:compute@developer.gserviceaccount.com" --format="value(email)" --limit=1)
        if [ -n "$COMPUTE_SA" ]; then
            SERVICE_ACCOUNT_EMAIL=$COMPUTE_SA
        fi
    fi
fi

echo "Using Service Account: $SERVICE_ACCOUNT_EMAIL"

# Define schedules in AWST (Australia/Perth) timezone
SCHEDULES=(
    "phantom-morning:30 7 * * *"      # 7:30 AM AWST - Morning coffee
    "phantom-midmorning:15 10 * * *"  # 10:15 AM AWST - Mid-morning
    "phantom-lunch:45 12 * * *"       # 12:45 PM AWST - Lunch break
    "phantom-afternoon:30 15 * * *"   # 3:30 PM AWST - Afternoon
    "phantom-evening:0 18 * * *"      # 6:00 PM AWST - Evening (peak)
    "phantom-night:30 20 * * *"       # 8:30 PM AWST - Night scroll
    "phantom-late:15 22 * * *"        # 10:15 PM AWST - Late night
)

echo "Creating/updating ${#SCHEDULES[@]} scheduler triggers (AWST timezone)..."

for schedule_entry in "${SCHEDULES[@]}"; do
    SCHEDULE_NAME="${schedule_entry%%:*}"
    CRON_EXPR="${schedule_entry##*:}"

    echo "  - $SCHEDULE_NAME: $CRON_EXPR ($TIMEZONE)"

    if gcloud scheduler jobs describe $SCHEDULE_NAME --location $REGION > /dev/null 2>&1; then
        gcloud scheduler jobs update http $SCHEDULE_NAME \
            --schedule "$CRON_EXPR" \
            --time-zone "$TIMEZONE" \
            --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/$JOB_NAME:run" \
            --http-method POST \
            --oauth-service-account-email $SERVICE_ACCOUNT_EMAIL \
            --location $REGION \
            --quiet
    else
        gcloud scheduler jobs create http $SCHEDULE_NAME \
            --schedule "$CRON_EXPR" \
            --time-zone "$TIMEZONE" \
            --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/$JOB_NAME:run" \
            --http-method POST \
            --oauth-service-account-email $SERVICE_ACCOUNT_EMAIL \
            --location $REGION \
            --quiet
    fi
done

# Clean up old scheduler names if they exist
OLD_SCHEDULES=("tech-influencer-morning" "tech-influencer-midmorning" "tech-influencer-lunch" "tech-influencer-afternoon" "tech-influencer-evening" "tech-influencer-night" "tech-influencer-late" "tech-influencer-schedule")
for old_name in "${OLD_SCHEDULES[@]}"; do
    if gcloud scheduler jobs describe $old_name --location $REGION > /dev/null 2>&1; then
        echo "Removing old scheduler: $old_name"
        gcloud scheduler jobs delete $old_name --location $REGION --quiet || true
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    Deployment Complete!                           ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  ${#SCHEDULES[@]} scheduler triggers created (AWST timezone)                 ║"
echo "║  The app uses probabilistic posting for human-like behavior.      ║"
echo "║                                                                   ║"
echo "║  Commands:                                                        ║"
echo "║    Force post: gcloud run jobs execute $JOB_NAME \\                ║"
echo "║                  --region $REGION --update-env-vars FORCE_POST=true║"
echo "║    Normal run: gcloud run jobs execute $JOB_NAME --region $REGION ║"
echo "║    View logs:  gcloud logging read 'resource.type=cloud_run_job'  ║"
echo "║                                                                   ║"
echo "║  Tip: Consider using Terraform for infrastructure management:     ║"
echo "║       cd terraform && terraform init && terraform apply           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
